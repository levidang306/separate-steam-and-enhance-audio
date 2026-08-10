"""Decide whether each restoration step should run, before it runs.

Every step in this pipeline used to run unconditionally, and the damage that
caused was not subtle: tempo correction warped a solo guitar by 375ms using a
beat grid that described 41% of the track, and bandwidth extension synthesised
a high band for a source whose content stopped at 2.8kHz. Neither step was
wrong in itself. Both were wrong *for that material*, and nothing in the design
was in a position to notice.

So each step now has to make its case first. A decision carries the evidence it
was made from, which means a skip can be argued with and a run can be audited.

The default answer is no. A step that cannot demonstrate it should run does not
run -- not because processing is bad, but because the cost of processing
material that did not need it is silent, permanent damage, while the cost of
skipping is only a missed improvement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .audio_io import to_mono
from .config import settings as default_settings
from .metrics import spectral_cliff_hz

# Percussive stems give a beat tracker clear onsets. Guitar and piano were
# removed: they were the last resort in the old priority list, which meant a
# solo guitar became its own tempo reference and got warped to a grid derived
# from its own uneven picking.
REFERENCE_PRIORITY = ("drum", "percussion", "bass", "instrumental")

# Analysis cost is bounded: HPSS over a five-minute stem is minutes of CPU, and
# the answer does not change once there is enough material to judge.
ANALYSIS_SECONDS = 45.0

# A gap spanning more beats than this is a hole, not a long beat. Beat trackers
# drop beats wherever a stem falls quiet -- the reference set had gaps of up to
# 28 seconds -- and a 28-second gap that happens to divide evenly by the beat
# interval is still 28 seconds the grid says nothing about. Allowing 2 covers a
# single missed beat without letting a silent passage count as described.
MAX_BEATS_PER_GAP = 2

EPS = 1e-12

# Rate thresholds are compared with a tolerance: `abs(1 - 0.98)` is
# 0.020000000000000018 in binary floating point, and an exact comparison against
# 0.02 turns a deliberately empty window into a window of width 1.8e-17.
RATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StepDecision:
    """Whether a step runs, how sure we are, and what that was based on.

    `confidence` is not decoration. Downstream it scales the wet contribution,
    so a marginal yes produces a gentle application rather than a full one.
    """

    run: bool
    confidence: float
    reason: str
    measurements: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        verb = "RUN " if self.run else "SKIP"
        return f"{verb} (confidence {self.confidence:.2f}): {self.reason}"


def _skip(reason: str, measurements: dict | None = None) -> StepDecision:
    return StepDecision(run=False, confidence=0.0, reason=reason,
                        measurements=measurements or {})


def _excerpt(audio: np.ndarray, sr: int, seconds: float = ANALYSIS_SECONDS) -> np.ndarray:
    """Mono excerpt from the middle of the signal, bounded in length."""
    mono = to_mono(audio).astype(np.float32)
    span = int(seconds * sr)
    if mono.size <= span:
        return mono
    start = (mono.size - span) // 2
    return mono[start:start + span]


# --- Tempo ---------------------------------------------------------------


def rank_reference_stems(stem_names: list[str]) -> list[str]:
    """Candidate stems to beat-track, best first; empty if none qualify.

    Unlike the old version this does not fall back to a melodic stem, and does
    not raise. An empty list is a legitimate answer meaning "nothing here is a
    usable tempo reference", which the gate turns into a skip.
    """
    if not stem_names:
        return []
    lowered = {name: name.lower() for name in stem_names}
    ranked: list[str] = []
    for keyword in REFERENCE_PRIORITY:
        for name, low in lowered.items():
            if keyword in low and name not in ranked:
                ranked.append(name)
    return ranked


def measure_percussive_score(audio: np.ndarray, sr: int) -> float:
    """Share of energy in the percussive component, 0 to 1.

    Beat tracking needs onsets it can trust. On the corpus this separates
    cleanly: a kit or a full mix scores around 0.37-0.44, a solo guitar 0.17, a
    piano 0.007, a voice 0.002.
    """
    import librosa

    mono = _excerpt(audio, sr)
    if mono.size < sr:
        return 0.0
    harmonic, percussive = librosa.effects.hpss(mono)
    harmonic_energy = float(np.sum(harmonic.astype(np.float64) ** 2))
    percussive_energy = float(np.sum(percussive.astype(np.float64) ** 2))
    total = harmonic_energy + percussive_energy
    return float(percussive_energy / total) if total > EPS else 0.0


def detect_beats_fallback(audio: np.ndarray, sr: int) -> np.ndarray:
    """Beat times from librosa, for when a tracker's output was not supplied.

    Production uses Beat This!, which lives in the V3 image. The gate must be
    able to reach a decision without it -- and must reach the same *kind* of
    decision -- so it accepts beats from either source.
    """
    import librosa

    mono = _excerpt(audio, sr)
    if mono.size < sr:
        return np.zeros(0)
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    return np.asarray(beats, dtype=np.float64)


def measure_grid(
    beat_times: np.ndarray,
    duration_s: float,
    min_correction: float = 0.02,
    max_deviation: float = 0.02,
) -> dict:
    """How much of the timeline a beat grid actually describes, and how steady.

    `coverage` counts only the inter-beat gaps that sit within tolerance of a
    whole number of median intervals. A tracker that drops beats through a quiet
    passage leaves a 28-second hole, and a grid with a hole in it cannot be used
    to warp the material inside the hole -- that region gets copied, and then
    inherits whatever drift the corrected regions accumulated.

    A segment is corrected only when its required rate is at least
    `min_correction` away from 1.0 (below that, stretching costs more smearing
    than the drift is worth) and at most `max_deviation` away (beyond that it is
    a mis-detected beat, not drift). Note that with both set to 0.02 -- the
    values the plan specifies -- that window is empty by construction, and
    `correctable_window_empty` says so rather than the caller having to work it
    out from a silent zero.
    """
    empty = {
        "n_beats": float(len(beat_times)),
        "coverage": 0.0,
        "median_bpm": float("nan"),
        "bpm_std_pct": float("inf"),
        "max_rate_deviation": 0.0,
        "median_rate_deviation": 0.0,
        "out_of_range_fraction": 1.0,
        "segments_needing_correction": 0,
        "segments_total": 0,
        "cumulative_drift_ms": 0.0,
        "interval_s": float("nan"),
        "correctable_window_empty": min_correction >= max_deviation - RATE_TOLERANCE,
    }
    if len(beat_times) < 4 or duration_s <= 0:
        return empty

    gaps = np.diff(beat_times)
    interval = float(np.median(gaps))
    if interval <= 0:
        return empty

    spanned = np.maximum(1.0, np.round(gaps / interval))
    per_beat = gaps / spanned
    # A gap only counts as covered if it is both consistent with the grid *and*
    # short enough to be a real beat. Without the second condition a silent
    # passage that happens to last a whole number of beats reads as perfectly
    # covered, which is precisely the case the coverage test exists to catch.
    consistent = np.abs(per_beat - interval) < 0.05 * interval
    covered = consistent & (spanned <= MAX_BEATS_PER_GAP)
    coverage = float(np.sum(gaps[covered]) / duration_s)

    bpm = 60.0 / per_beat
    median_bpm = float(np.median(bpm))
    bpm_std_pct = float(np.std(bpm) / median_bpm * 100) if median_bpm > 0 else float("inf")

    # The correction the step would apply: each gap stretched to a whole number
    # of median intervals.
    targets = spanned * interval
    rates = gaps / targets
    deviation = np.abs(rates - 1.0)

    # A segment whose required rate is wildly off is a mis-detected beat, and
    # the warp leaves those alone rather than stretching to match. So they are
    # counted, not treated as drift -- but a grid where *most* segments are
    # mis-detected is not describing the material, which is what the gate cares
    # about. The tracker's own quantisation lands here too: a 10.7ms frame hop
    # is already 2% of a 0.5s beat, so this is not a rare edge case.
    out_of_range = deviation > max_deviation
    correctable = (deviation >= min_correction) & ~out_of_range
    drift_s = float(np.sum((targets - gaps)[correctable]))

    return {
        "n_beats": float(len(beat_times)),
        "coverage": coverage,
        "median_bpm": median_bpm,
        "bpm_std_pct": bpm_std_pct,
        "max_rate_deviation": float(np.max(deviation)),
        "median_rate_deviation": float(np.median(deviation)),
        "out_of_range_fraction": float(np.mean(out_of_range)),
        "segments_needing_correction": int(np.sum(correctable)),
        "segments_total": int(len(gaps)),
        "cumulative_drift_ms": drift_s * 1000.0,
        "interval_s": interval,
        "correctable_window_empty": min_correction >= max_deviation - RATE_TOLERANCE,
    }


def decide_tempo(
    audio: np.ndarray,
    sr: int,
    *,
    stem_name: str = "",
    available_stems: list[str] | None = None,
    is_separated: bool = False,
    beat_times: np.ndarray | None = None,
    settings=None,
) -> StepDecision:
    """Should tempo correction run on this material?

    Every condition here corresponds to a way the step damaged the reference
    guitar. It is deliberately hard to satisfy: tempo correction is the only
    stage that touches the time axis, its damage is unrecoverable, and it helps
    only material with a real, steady, percussive pulse.
    """
    settings = settings or default_settings
    measurements: dict = {"stem": stem_name, "is_separated": is_separated}

    if not settings.tempo_enabled:
        return _skip("tempo correction is disabled by default", measurements)

    if is_separated and not settings.tempo_allow_on_separated:
        return _skip(
            "refusing to beat-track a separated stem; separation reshapes the "
            "transients a tracker depends on",
            measurements,
        )

    if available_stems is not None and not rank_reference_stems(available_stems):
        measurements["available_stems"] = sorted(available_stems)
        return _skip(
            f"no percussive reference stem; need one of {REFERENCE_PRIORITY}", measurements
        )

    percussive = measure_percussive_score(audio, sr)
    measurements["percussive_score"] = percussive
    if percussive < settings.tempo_min_percussive_score:
        return _skip(
            f"material is not percussive enough to beat-track "
            f"({percussive:.3f} < {settings.tempo_min_percussive_score})",
            measurements,
        )

    beats = detect_beats_fallback(audio, sr) if beat_times is None else np.asarray(beat_times)
    duration = min(audio.shape[0] / sr, ANALYSIS_SECONDS) if beat_times is None else (
        audio.shape[0] / sr
    )
    low, high = settings.tempo_rate_bounds
    max_deviation = max(abs(1.0 - low), abs(high - 1.0))
    grid = measure_grid(beats, duration, settings.tempo_min_correction, max_deviation)
    measurements.update(grid)

    if grid["n_beats"] < 4:
        return _skip(f"only {int(grid['n_beats'])} beats detected", measurements)

    if grid["coverage"] < settings.tempo_min_grid_coverage:
        return _skip(
            f"beat grid covers {grid['coverage']:.0%} of the track, below "
            f"{settings.tempo_min_grid_coverage:.0%}; the uncovered regions would be "
            f"copied and would inherit the drift corrected elsewhere",
            measurements,
        )

    if grid["bpm_std_pct"] > settings.tempo_max_bpm_std_pct:
        return _skip(
            f"tempo is unsteady ({grid['bpm_std_pct']:.2f}% spread > "
            f"{settings.tempo_max_bpm_std_pct}%); there is no single correct tempo to warp to",
            measurements,
        )

    # Most segments being out of range means the grid is not describing the
    # material; a few means a few mis-detected beats, which the warp skips.
    if grid["out_of_range_fraction"] > 0.5:
        return _skip(
            f"{grid['out_of_range_fraction']:.0%} of segments would need a rate outside "
            f"({low}, {high}); the grid is not describing this material",
            measurements,
        )

    if grid["correctable_window_empty"]:
        return _skip(
            f"no segment can qualify: RESTORE_TEMPO_MIN_CORRECTION "
            f"({settings.tempo_min_correction}) is not below the rate bound "
            f"({max_deviation}), so the correctable window is empty",
            measurements,
        )

    if grid["segments_needing_correction"] == 0:
        return _skip(
            f"no segment deviates by more than {settings.tempo_min_correction:.0%}; "
            f"stretching would cost smearing to fix rounding error",
            measurements,
        )

    drift = abs(grid["cumulative_drift_ms"])
    if drift > settings.tempo_max_cumulative_drift_ms:
        return _skip(
            f"correction would move the timeline by {drift:.0f} ms, over the "
            f"{settings.tempo_max_cumulative_drift_ms:.0f} ms limit",
            measurements,
        )

    # Confidence is the weakest piece of evidence, not an average of them: a
    # grid that barely covers the track is not rescued by a steady BPM.
    confidence = min(
        1.0,
        grid["coverage"] / max(settings.tempo_min_grid_coverage, EPS),
        settings.tempo_max_bpm_std_pct / max(grid["bpm_std_pct"], EPS),
        percussive / max(settings.tempo_min_percussive_score, EPS),
    )
    if confidence < settings.tempo_min_confidence:
        return _skip(
            f"confidence {confidence:.2f} below {settings.tempo_min_confidence}", measurements
        )

    return StepDecision(
        run=True,
        confidence=float(confidence),
        reason=(
            f"steady {grid['median_bpm']:.1f} BPM pulse, grid covers "
            f"{grid['coverage']:.0%}, {grid['segments_needing_correction']} segments drift "
            f"by up to {grid['max_rate_deviation'] * 100:.1f}%"
        ),
        measurements=measurements,
    )


# --- Denoise -------------------------------------------------------------


def measure_noise(audio: np.ndarray, sr: int) -> dict:
    """Level and character of the stationary bed sitting under the music.

    The obvious approach -- take the quietest frames and call that the noise
    floor -- does not work on real material, because on most music there are no
    frames where everything has stopped. A guitar with 2.5-second decays never
    drops below about -50dBFS inside a phrase, so a temporal percentile reports
    the tail of the last note and is completely insensitive to the actual floor:
    the same number comes back for a -70dB room tone and a -95dB one.

    So the estimate is spectral instead. For each frequency bin, the level it
    falls to across time is what is always present at that frequency -- a
    minimum-statistics noise estimate. Music moves; a noise bed does not. That
    reads a real floor even while notes are ringing, and it also gives the
    noise's *shape*, which is what separates hiss from reverb tail.
    """
    import librosa

    mono = _excerpt(audio, sr)
    if mono.size < 4096:
        return {"noise_floor_db": -np.inf, "hiss_db": -np.inf, "noise_flatness": 0.0,
                "program_db": -np.inf, "snr_db": float("inf")}

    spec = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    if spec.shape[1] < 8:
        return {"noise_floor_db": -np.inf, "hiss_db": -np.inf, "noise_flatness": 0.0,
                "program_db": -np.inf, "snr_db": float("inf")}

    # Per-bin floor across time: what is present at this frequency even at its
    # quietest. The 10th percentile rather than the minimum, so one unusually
    # quiet frame cannot set the estimate.
    profile = np.percentile(spec, 10, axis=1)

    # Convert to honest dBFS by scaling against the time-domain level: the STFT
    # carries a window-dependent factor that an absolute threshold cannot ignore.
    total_power = float(np.mean(np.sum(spec**2, axis=0)))
    noise_power = float(np.sum(profile**2))
    signal_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    signal_db = 20 * np.log10(signal_rms + EPS)
    noise_floor = (
        signal_db + 10 * np.log10(noise_power / total_power)
        if total_power > EPS and noise_power > EPS
        else -np.inf
    )

    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    program = float(np.percentile(20 * np.log10(rms + EPS), 90))

    high = freqs >= 10000.0
    if high.any() and noise_power > EPS:
        high_power = float(np.sum(profile[high] ** 2))
        hiss = (
            signal_db + 10 * np.log10(high_power / total_power) - program
            if high_power > EPS
            else -np.inf
        )
    else:
        hiss = -np.inf

    # Flatness of the noise *profile*: broadband hiss is flat, a reverb tail or
    # a sustained note is peaky. This is what stops the gate from treating decay
    # as something to remove.
    positive = profile[profile > EPS]
    flatness = (
        float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
        if positive.size > 8
        else 0.0
    )

    return {
        "noise_floor_db": float(noise_floor),
        "program_db": program,
        "hiss_db": float(hiss),
        "noise_flatness": flatness,
        "snr_db": float(program - noise_floor),
    }


def decide_denoise(audio: np.ndarray, sr: int, *, settings=None) -> StepDecision:
    """Should denoising run, and how hard?

    Unlike tempo, this one usually says yes -- but the confidence it says it
    with is what keeps it gentle. A stem with a barely-there noise floor gets a
    correspondingly small wet contribution rather than a full-strength pass that
    strips its room tone.
    """
    settings = settings or default_settings
    if not settings.denoise_enabled:
        return _skip("denoise is disabled")

    measurements = measure_noise(audio, sr)
    floor = measurements["noise_floor_db"]

    if not np.isfinite(floor):
        return _skip("stem is silent or too short to measure", measurements)

    if floor <= settings.denoise_min_noise_floor_db:
        return _skip(
            f"noise floor is already at {floor:.1f} dBFS, below "
            f"{settings.denoise_min_noise_floor_db:.0f}; nothing to remove",
            measurements,
        )

    hiss = measurements["hiss_db"]
    if hiss <= settings.denoise_min_hiss_db:
        return _skip(
            f"stationary high-band bed is {hiss:.1f} dB below program, under "
            f"{settings.denoise_min_hiss_db:.0f}; there is no audible hiss to remove",
            measurements,
        )

    # How far above "already clean" the hiss sits, over a 30dB span. 30dB above
    # the threshold is unambiguously noisy; just above it is not.
    level_score = float(np.clip((hiss - settings.denoise_min_hiss_db) / 30.0, 0.0, 1.0))
    # Flatness above ~0.10 means a broadband bed rather than a ringing tail. It
    # is the term that stops the gate treating natural decay as noise.
    flatness_score = float(np.clip(measurements["noise_flatness"] / 0.10, 0.0, 1.0))
    confidence = float(np.clip(min(level_score, flatness_score), 0.0, 1.0))
    measurements["level_score"] = level_score
    measurements["flatness_score"] = flatness_score

    if confidence < settings.denoise_min_confidence:
        return _skip(
            f"confidence {confidence:.3f} is below {settings.denoise_min_confidence}: the bed "
            f"under this stem is not broadband (flatness "
            f"{measurements['noise_flatness']:.3f}), so that is decay or coding artifact, "
            f"not hiss",
            measurements,
        )

    return StepDecision(
        run=True,
        confidence=confidence,
        reason=(
            f"noise floor at {floor:.1f} dBFS ({measurements['snr_db']:.0f} dB below program), "
            f"hiss {measurements['hiss_db']:.1f} dB down, flatness "
            f"{measurements['noise_flatness']:.3f}"
        ),
        measurements=measurements,
    )


# --- Bandwidth -----------------------------------------------------------


def measure_bandwidth(audio: np.ndarray, sr: int) -> dict:
    """Is there a codec-shaped cliff, and does the signal actually reach it?

    Both halves matter. The reference guitar had no 8-20kHz cliff at all -- its
    content simply stopped at 2.8kHz -- so there was nothing to restore, and
    what got synthesised had 0.012 coherence with the source.
    """
    import librosa

    mono = _excerpt(audio, sr)
    if mono.size < 4096:
        return {"cliff_hz": None, "rolloff85_hz": 0.0, "rolloff99_hz": 0.0,
                "energy_below_cliff_pct": 0.0}

    cliff = spectral_cliff_hz(audio, sr)
    rolloff85 = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.85)))
    rolloff99 = float(np.mean(librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.99)))

    energy_below = 0.0
    if cliff:
        spec = np.abs(librosa.stft(mono, n_fft=8192, hop_length=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        profile = np.percentile(spec, 95, axis=1)
        total = float(np.sum(profile))
        # The octave immediately below the cliff. If that is empty the signal
        # never reached the cliff, and there is nothing to extend upward from.
        shoulder = (freqs >= cliff * 0.5) & (freqs < cliff)
        if total > EPS and shoulder.any():
            energy_below = float(np.sum(profile[shoulder]) / total * 100)

    return {
        "cliff_hz": float(cliff) if cliff else None,
        "rolloff85_hz": rolloff85,
        "rolloff99_hz": rolloff99,
        "energy_below_cliff_pct": energy_below,
    }


def decide_bandwidth(audio: np.ndarray, sr: int, *, settings=None) -> StepDecision:
    """Should bandwidth extension run?

    Only for a real cutoff. Generating a high band for material that never had
    one is not restoration -- it is invention, and it measures as invention.
    """
    settings = settings or default_settings
    if not settings.bandwidth_enabled:
        return _skip("bandwidth extension is disabled")

    measurements = measure_bandwidth(audio, sr)
    cliff = measurements["cliff_hz"]

    if cliff is None:
        return _skip(
            f"no spectral cliff found; content rolls off gradually at "
            f"{measurements['rolloff85_hz']:.0f} Hz (85%), so there is no cutoff to restore",
            measurements,
        )

    if not settings.bandwidth_min_cliff_hz <= cliff <= settings.bandwidth_max_cliff_hz:
        return _skip(
            f"cliff at {cliff:.0f} Hz is outside "
            f"{settings.bandwidth_min_cliff_hz:.0f}-{settings.bandwidth_max_cliff_hz:.0f} Hz; "
            f"that is tonal balance, not a codec cutoff",
            measurements,
        )

    if measurements["energy_below_cliff_pct"] < settings.bandwidth_min_energy_below_cliff_pct:
        return _skip(
            f"only {measurements['energy_below_cliff_pct']:.2f}% of energy sits in the octave "
            f"below the cliff; the signal never reaches it, so there is nothing to extend from",
            measurements,
        )

    headroom = (sr / 2) - cliff
    if headroom < 2000:
        return _skip(
            f"cliff at {cliff:.0f} Hz leaves only {headroom:.0f} Hz below Nyquist", measurements
        )

    # More confident the lower the cutoff sits and the more material feeds it.
    span_score = float(np.clip(headroom / 8000.0, 0.0, 1.0))
    feed_score = float(
        np.clip(
            measurements["energy_below_cliff_pct"]
            / max(settings.bandwidth_min_energy_below_cliff_pct * 3, EPS),
            0.0,
            1.0,
        )
    )
    confidence = float(np.clip(min(span_score, feed_score), 0.0, 1.0))
    measurements["span_score"] = span_score
    measurements["feed_score"] = feed_score

    return StepDecision(
        run=True,
        confidence=confidence,
        reason=(
            f"spectral cliff at {cliff:.0f} Hz with "
            f"{measurements['energy_below_cliff_pct']:.1f}% of energy feeding it from below"
        ),
        measurements=measurements,
    )


def decide_all(
    audio: np.ndarray,
    sr: int,
    *,
    stem_name: str = "",
    available_stems: list[str] | None = None,
    is_separated: bool = False,
    settings=None,
) -> dict[str, StepDecision]:
    """Every decision for one stem, taken before any of them is acted on."""
    return {
        "tempo": decide_tempo(
            audio, sr, stem_name=stem_name, available_stems=available_stems,
            is_separated=is_separated, settings=settings,
        ),
        "denoise": decide_denoise(audio, sr, settings=settings),
        "bandwidth": decide_bandwidth(audio, sr, settings=settings),
    }


def decisions_to_config(decisions: dict[str, StepDecision]) -> dict:
    """Serialise decisions for `engine_config`, so a run can be explained later."""
    return {name: decision.to_dict() for name, decision in decisions.items()}


__all__ = [
    "REFERENCE_PRIORITY",
    "StepDecision",
    "decide_all",
    "decide_bandwidth",
    "decide_denoise",
    "decide_tempo",
    "decisions_to_config",
    "measure_bandwidth",
    "measure_grid",
    "measure_noise",
    "measure_percussive_score",
    "rank_reference_stems",
]
