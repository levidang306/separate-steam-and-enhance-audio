"""Step 1 -- tempo correction.

Suno can let the tempo wander across a song. This detects the beat grid once, on
one reference stem, and applies the resulting time-warp identically to every
stem. Beat-tracking each stem separately would give each its own warp and drift
the stems apart, which defeats the point of exporting stems at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from .audio_io import to_mono
from .metrics import tempo_stability

# Percussive stems give the beat tracker the clearest onsets; melodic stems are
# the last resort.
REFERENCE_PRIORITY = ("drum", "bass", "instrumental", "guitar", "piano")

# Below this deviation a segment is copied rather than stretched. Phase-vocoder
# stretching always costs some smearing, so it should not be spent undoing
# rounding-level differences.
MIN_CORRECTION = 0.005

# Segments needing more than this are treated as beat-tracking failures rather
# than tempo drift, and left alone.
MAX_RATE_DEVIATION = (0.80, 1.25)

CROSSFADE_S = 0.005


class NoReferenceStem(RuntimeError):
    pass


@dataclass
class TempoAnalysis:
    reference_stem: str
    beat_times: np.ndarray
    target_interval_s: float
    before: dict = field(default_factory=dict)
    applied: dict = field(default_factory=dict)

    @property
    def target_bpm(self) -> float:
        return 60.0 / self.target_interval_s if self.target_interval_s else float("nan")

    def summary(self) -> str:
        if not self.applied or not self.applied.get("segments_corrected"):
            return (
                f"{self.target_bpm:.1f} BPM on '{self.reference_stem}', "
                f"spread ±{self.before.get('std_bpm', float('nan')):.2f} BPM — "
                "no segment exceeded the correction threshold, audio unchanged"
            )
        return (
            f"{self.target_bpm:.1f} BPM on '{self.reference_stem}': corrected "
            f"{self.applied['segments_corrected']}/{self.applied['segments_total']} segments, "
            f"median {self.applied['median_correction_pct']:.2f}%, "
            f"max {self.applied['max_correction_pct']:.2f}%, "
            f"length {self.applied['length_change_pct']:+.2f}%"
        )


def rank_reference_stems(stem_names: list[str]) -> list[str]:
    """Order candidate stems for beat-tracking, best first.

    More than one candidate is returned because the best instrument is not
    always usable: a drum stem that stays silent through the section being
    processed yields no beats at all, and the next candidate should be tried
    rather than failing the step.
    """
    # With a single stem the priority list is moot: it is the only candidate, and
    # there is nothing else for it to stay in sync with.
    if len(stem_names) == 1:
        return list(stem_names)

    lowered = {name: name.lower() for name in stem_names}
    ranked = []
    for keyword in REFERENCE_PRIORITY:
        for name, low in lowered.items():
            if keyword in low and name not in ranked:
                ranked.append(name)
    if not ranked:
        raise NoReferenceStem(
            "No drums/bass/instrumental stem found to beat-track. Tempo correction "
            f"needs one of {REFERENCE_PRIORITY}; got: {sorted(stem_names)}"
        )
    return ranked


def pick_reference_stem(stem_names: list[str]) -> str:
    """Highest-priority stem to beat-track."""
    return rank_reference_stems(stem_names)[0]


def detect_beats(audio: np.ndarray, sr: int, device: str = "cpu") -> np.ndarray:
    """Return beat positions in seconds using Beat This!."""
    from beat_this.inference import Audio2Beats

    audio2beats = Audio2Beats(checkpoint_path="final0", device=device)
    beats, _downbeats = audio2beats(to_mono(audio).astype(np.float32), sr)
    return np.asarray(beats, dtype=np.float64)


def analyse(audio: np.ndarray, sr: int, reference_stem: str, device: str = "cpu") -> TempoAnalysis:
    """Detect the beat grid and the uniform interval to warp it onto."""
    beat_times = detect_beats(audio, sr, device=device)
    if len(beat_times) < 4:
        raise NoReferenceStem(
            f"Only {len(beat_times)} beats detected in '{reference_stem}' -- "
            "not enough to establish a tempo grid."
        )
    target_interval = float(np.median(np.diff(beat_times)))
    return TempoAnalysis(
        reference_stem=reference_stem,
        beat_times=beat_times,
        target_interval_s=target_interval,
        before=tempo_stability(beat_times, target_interval),
    )


def _segment_bounds(
    analysis: TempoAnalysis, n_samples: int, sr: int
) -> list[tuple[int, int, float]]:
    """Split the timeline into (start, end, stretch_rate) segments.

    Rate follows librosa's convention: >1 shortens a segment, <1 lengthens it.

    Consecutive detected beats are not necessarily one beat apart. Beat This!
    drops beats wherever the reference stem falls quiet -- on the reference set
    it found 188 beats where a steady 68 BPM would give ~333, leaving gaps of up
    to 28 seconds. Each gap is therefore measured in whole beats before being
    corrected; forcing every gap to a single beat collapsed the track to 79% of
    its length.
    """
    beats = analysis.beat_times
    target_samples = int(round(analysis.target_interval_s * sr))

    edges = [0]
    edges.extend(int(round(t * sr)) for t in beats)
    edges.append(n_samples)
    edges = sorted({min(max(e, 0), n_samples) for e in edges})

    segments = []
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        length = end - start
        if length <= 0:
            continue

        # Head and tail sit outside the beat grid, so there is nothing to align
        # them to; only inter-beat segments get corrected.
        inside_grid = 0 < index < len(edges) - 2
        rate = 1.0
        if inside_grid and target_samples > 0:
            beats_spanned = max(1, round(length / target_samples))
            # Each crossfaded join consumes CROSSFADE_S of material, so segments
            # are stretched that much longer to compensate. Without this the
            # track loses ~0.3% of its length across ~190 joins -- more than a
            # tenth of the drift the step exists to correct.
            target = beats_spanned * target_samples + int(CROSSFADE_S * sr)
            rate = length / target
            # A rate this far off is a tracking failure, not drift; stretching to
            # match it would do more damage than the drift it claims to fix.
            if not MAX_RATE_DEVIATION[0] <= rate <= MAX_RATE_DEVIATION[1]:
                rate = 1.0

        if abs(rate - 1.0) < MIN_CORRECTION:
            rate = 1.0
        segments.append((start, end, rate))
    return segments


def _stretch(segment: np.ndarray, rate: float) -> np.ndarray:
    if rate == 1.0 or segment.shape[0] < 4096:
        return segment
    if segment.ndim == 1:
        return librosa.effects.time_stretch(segment, rate=rate)
    # librosa expects channels-first for multichannel input.
    return librosa.effects.time_stretch(segment.T, rate=rate).T


def _concat_with_crossfade(pieces: list[np.ndarray], sr: int) -> np.ndarray:
    """Join stretched segments, crossfading the seams to suppress clicks."""
    pieces = [p for p in pieces if p.shape[0] > 0]
    if not pieces:
        return np.zeros(0, dtype=np.float32)

    fade = int(CROSSFADE_S * sr)
    out = pieces[0]
    for piece in pieces[1:]:
        n = min(fade, out.shape[0], piece.shape[0])
        if n <= 0:
            out = np.concatenate([out, piece], axis=0)
            continue
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if out.ndim == 2:
            ramp = ramp[:, None]
        blended = out[-n:] * (1 - ramp) + piece[:n] * ramp
        out = np.concatenate([out[:-n], blended, piece[n:]], axis=0)
    return out.astype(np.float32)


def apply_warp(audio: np.ndarray, sr: int, analysis: TempoAnalysis) -> np.ndarray:
    """Apply the shared time-warp to one stem."""
    segments = _segment_bounds(analysis, audio.shape[0], sr)
    if all(rate == 1.0 for _, _, rate in segments):
        return audio
    pieces = [_stretch(audio[start:end], rate) for start, end, rate in segments]
    return _concat_with_crossfade(pieces, sr)


def describe_warp(analysis: TempoAnalysis, n_samples: int, sr: int) -> dict:
    """Report exactly what the warp does, without re-running beat detection.

    Re-detecting beats on warped audio and comparing is unreliable: the tracker
    flips octaves (reporting 136 BPM for a 68 BPM track) and finds a different
    number of beats each time, so the comparison measures tracker noise rather
    than the correction.
    """
    segments = _segment_bounds(analysis, n_samples, sr)
    corrections = [abs(rate - 1.0) * 100 for _, _, rate in segments if rate != 1.0]
    fade = int(CROSSFADE_S * sr)
    warped_length = sum(
        (end - start) if rate == 1.0 else (end - start) / rate for start, end, rate in segments
    ) - fade * max(0, len(segments) - 1)
    return {
        "segments_total": len(segments),
        "segments_corrected": len(corrections),
        "median_correction_pct": float(np.median(corrections)) if corrections else 0.0,
        "max_correction_pct": float(np.max(corrections)) if corrections else 0.0,
        "length_change_pct": (warped_length / n_samples - 1) * 100 if n_samples else 0.0,
    }


def correct_tempo(
    stems: dict[str, tuple[np.ndarray, int]], device: str = "cpu"
) -> tuple[dict[str, tuple[np.ndarray, int]], TempoAnalysis]:
    """Beat-track one stem, then warp every stem with that single result."""
    candidates = rank_reference_stems(list(stems))
    analysis = None
    failures = []
    for reference_stem in candidates:
        ref_audio, ref_sr = stems[reference_stem]
        try:
            analysis = analyse(ref_audio, ref_sr, reference_stem, device=device)
            break
        except NoReferenceStem as error:
            failures.append(str(error))
    if analysis is None:
        raise NoReferenceStem(
            "No stem yielded a usable beat grid:\n  " + "\n  ".join(failures)
        )

    ref_audio, ref_sr = stems[analysis.reference_stem]
    analysis.applied = describe_warp(analysis, ref_audio.shape[0], ref_sr)

    corrected = {
        name: (apply_warp(audio, sr, analysis), sr) for name, (audio, sr) in stems.items()
    }
    return corrected, analysis
