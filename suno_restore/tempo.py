"""Step 1 -- tempo correction.

Suno can let the tempo wander across a song. This detects the beat grid once, on
one reference stem, and applies the resulting time-warp identically to every
stem. Beat-tracking each stem separately would give each its own warp and drift
the stems apart, which defeats the point of exporting stems at all.

The warp is applied as a *single continuous pass* over the whole signal. The
previous version cut the timeline at every beat, time-stretched each piece with
an independent `librosa.effects.time_stretch` call, and glued the results back
together with a 5ms linear crossfade. Every one of those joins was audible, for
three separate reasons, all of them measured on a pure 440Hz tone -- a signal
whose envelope is flat by construction, so any dip in it is manufactured:

  * `librosa.effects.time_stretch` runs an STFT and an inverse STFT. The last
    synthesis windows of a segment are not fully overlap-added, so the segment
    *decays* at its tail: 13dB down over the final milliseconds at rate 0.99.
    A 5ms crossfade cannot cover a decay that spans a whole 2048-sample window.
  * Each call ran its own phase vocoder, which starts phase accumulation from
    scratch. Phase at a segment boundary was therefore arbitrary, and
    crossfading two copies of the same partial at different phases cancels
    rather than sums.
  * Joins consumed 5ms of material each. Segments that were stretched had that
    5ms added back; segments left at rate 1.0 by `MIN_CORRECTION` did not, so
    those joins deleted 5ms of music outright.

Measured on the 440Hz tone at a realistic beat spacing, the three together put a
level dip at nearly every join, worst case 9.2dB, once per beat for the length
of the track. That is what "not fully smooth" sounds like.

There is no way to crossfade that away, because the crossfade is what causes it.
So there are no segments and no joins here any more. The beat grid becomes a
continuous, monotonic map from source time to output time, and one variable-rate
phase vocoder walks the whole signal through it. Beats still land exactly on the
grid -- they are the anchors of the map -- but between them the rate now varies
smoothly instead of stepping, and the phase accumulator never restarts.
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

# Below this deviation an interval is left alone rather than warped. Any
# resynthesis costs some smearing, so it should not be spent undoing
# rounding-level differences.
MIN_CORRECTION = 0.005

# Intervals needing more than this are treated as beat-tracking failures rather
# than tempo drift, and left alone.
MAX_RATE_DEVIATION = (0.80, 1.25)

# Phase-vocoder resolution. 2048/512 at 44.1-48kHz is the usual compromise
# between frequency resolution (which sets how cleanly partials separate) and
# time resolution (which sets how much transients smear).
N_FFT = 2048
HOP_LENGTH = 512


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
                "no interval exceeded the correction threshold, audio unchanged"
            )
        return (
            f"{self.target_bpm:.1f} BPM on '{self.reference_stem}': corrected "
            f"{self.applied['segments_corrected']}/{self.applied['segments_total']} intervals, "
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


def warp_anchors(
    analysis: TempoAnalysis, n_samples: int, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Source and output sample positions the warp must map onto each other.

    The two arrays are the same length and both strictly increasing, so the map
    between them is continuous and invertible by construction. That property is
    the whole point: it is what lets the warp be applied in one pass, and a
    one-pass warp is what has no seams in it.

    Consecutive detected beats are not necessarily one beat apart. Beat This!
    drops beats wherever the reference stem falls quiet -- on the reference set
    it found 188 beats where a steady 68 BPM would give ~333, leaving gaps of up
    to 28 seconds. Each gap is therefore measured in whole beats before being
    corrected; forcing every gap to a single beat collapsed the track to 79% of
    its length.

    The material before the first beat and after the last sits outside the grid,
    so it has nothing to align to and is carried at rate 1.0.
    """
    target_samples = int(round(analysis.target_interval_s * sr))
    beats = np.unique(np.clip(np.round(analysis.beat_times * sr), 0, n_samples).astype(np.int64))
    beats = beats[(beats > 0) & (beats < n_samples)]

    if len(beats) < 2 or target_samples <= 0:
        return np.array([0, n_samples], dtype=np.float64), np.array(
            [0, n_samples], dtype=np.float64
        )

    source = [0.0, float(beats[0])]
    output = [0.0, float(beats[0])]
    for start, end in zip(beats, beats[1:]):
        length = int(end - start)
        beats_spanned = max(1, int(round(length / target_samples)))
        target = beats_spanned * target_samples
        rate = length / target
        # A rate this far off is a tracking failure, not drift; warping to match
        # it would do more damage than the drift it claims to fix.
        if not MAX_RATE_DEVIATION[0] <= rate <= MAX_RATE_DEVIATION[1]:
            target = length
        elif abs(rate - 1.0) < MIN_CORRECTION:
            target = length
        source.append(float(end))
        output.append(output[-1] + float(max(1, target)))

    # Tail, outside the grid: carried at rate 1.0.
    if n_samples > source[-1]:
        tail = n_samples - source[-1]
        source.append(float(n_samples))
        output.append(output[-1] + tail)

    return np.asarray(source, dtype=np.float64), np.asarray(output, dtype=np.float64)


def _source_positions(
    source: np.ndarray, output: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Invert the warp: where in the source does each output position come from?

    A monotone cubic (PCHIP) interpolant is used rather than a straight line
    through the anchors. Both place the beats identically -- they agree at every
    anchor -- but the piecewise-linear version has a slope discontinuity at each
    one, which is a step change in playback rate on every beat. PCHIP is
    C1-continuous and preserves monotonicity, so the rate glides between
    intervals instead of stepping, and the map still cannot fold back on itself.
    """
    if len(output) < 3:
        return np.interp(positions, output, source)
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(output, source, extrapolate=True)(positions)
    except ImportError:  # pragma: no cover - scipy is a hard dependency elsewhere
        return np.interp(positions, output, source)


def _phase_advance(spectrum: np.ndarray, frames: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """True phase advance over one analysis hop, at each read position.

    `expected` is the phase a bin would advance by if it sat exactly on its bin
    centre. What a partial actually does is that plus a wrapped deviation, and
    the deviation is what carries its real frequency.
    """
    left = np.angle(spectrum[:, frames])
    right = np.angle(spectrum[:, np.minimum(frames + 1, spectrum.shape[1] - 1)])
    deviation = right - left - expected
    deviation -= 2.0 * np.pi * np.round(deviation / (2.0 * np.pi))
    return (expected + deviation).astype(np.float64)


def _lock_phase_to_peaks(
    magnitude: np.ndarray, rotation: np.ndarray, block: int = 512
) -> np.ndarray:
    """Give every bin the phase rotation of the spectral peak it belongs to.

    A phase vocoder advances each bin independently, so the bins that together
    describe one partial drift out of step with each other. The partial stays at
    the right frequency but loses its shape, which is the hollow, watery
    "phasiness" a vocoder is known for. Locking each bin to the peak dominating
    its neighbourhood -- Laroche and Dolson's identity locking -- keeps the
    bins of a partial rotating together, so the waveform keeps its shape.

    Only the *rotation* is locked, never the measured phase, so the stereo image
    and the fine structure of the source both survive untouched.

    Frames are independent of each other here, so they are done in blocks: the
    index arrays this needs are the same size as the spectrogram, and a whole
    track's worth of them at once is several hundred megabytes for no benefit.
    """
    if magnitude.shape[1] > block:
        for start in range(0, magnitude.shape[1], block):
            stop = start + block
            rotation[:, start:stop] = _lock_phase_to_peaks(
                magnitude[:, start:stop], rotation[:, start:stop], block
            )
        return rotation

    bins = magnitude.shape[0]
    louder_than_below = np.empty_like(magnitude, dtype=bool)
    louder_than_above = np.empty_like(magnitude, dtype=bool)
    louder_than_below[0] = True
    louder_than_below[1:] = magnitude[1:] >= magnitude[:-1]
    louder_than_above[-1] = True
    louder_than_above[:-1] = magnitude[:-1] >= magnitude[1:]
    peaks = louder_than_below & louder_than_above

    # Nearest peak at or below each bin, and at or above it; whichever is closer
    # owns the bin. `maximum.accumulate` over the peak indices does the
    # fill-forward, and the same trick reversed does the fill-backward.
    index = np.arange(bins, dtype=np.int32)[:, None]
    below = np.where(peaks, index, -1)
    np.maximum.accumulate(below, axis=0, out=below)
    above = np.where(peaks, index, bins)
    above = np.minimum.accumulate(above[::-1], axis=0)[::-1]

    below_valid = below >= 0
    above_valid = above < bins
    below_safe = np.where(below_valid, below, 0)
    above_safe = np.where(above_valid, above, bins - 1)
    take_below = below_valid & (
        ~above_valid | ((index - below_safe) <= (above_safe - index))
    )
    owner = np.where(take_below, below_safe, above_safe)

    return np.take_along_axis(rotation, owner, axis=0)


def _warp_channel(
    signal: np.ndarray | None,
    rotation: np.ndarray,
    read_frames: np.ndarray,
    read_fraction: np.ndarray,
    length: int,
    spectrum: np.ndarray | None = None,
) -> np.ndarray:
    """Resynthesise one channel at the warped positions, using a shared rotation.

    Every channel is turned by the *same* phase rotation, derived once from the
    downmix. Running an independent vocoder per channel is the usual way to do
    this and it quietly widens or collapses the image, because the phase
    relationship between left and right is exactly what stereo is made of and
    two independent accumulators do not preserve it. One shared rotation moves
    both channels through time together and leaves that relationship intact.
    """
    if spectrum is None:
        spectrum = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    right_frames = np.minimum(read_frames + 1, spectrum.shape[1] - 1)
    read = spectrum[:, read_frames]
    magnitude = (1.0 - read_fraction) * np.abs(read) + (
        read_fraction * np.abs(spectrum[:, right_frames])
    )
    # `read / |read|` is the source phase as a unit vector -- the same thing as
    # `exp(1j*angle(read))` without building the intermediate angle array.
    unit = np.divide(read, np.abs(read), out=np.ones_like(read), where=np.abs(read) > 0)
    warped = (magnitude * unit * np.exp(1j * rotation)).astype(np.complex64)
    del read, magnitude, unit
    return librosa.istft(warped, hop_length=HOP_LENGTH, n_fft=N_FFT, length=length)


def apply_warp(audio: np.ndarray, sr: int, analysis: TempoAnalysis) -> np.ndarray:
    """Apply the shared time-warp to one stem, in a single continuous pass."""
    n_samples = audio.shape[0]
    source, output = warp_anchors(analysis, n_samples, sr)
    if np.allclose(source, output):
        return audio

    out_length = int(round(output[-1]))
    if out_length < N_FFT or n_samples < N_FFT:
        return audio

    # One output frame every HOP_LENGTH samples; each reads from wherever in the
    # source that output instant came from. One frame past the end, so the final
    # samples are synthesised from a full overlap-add rather than a fading tail.
    out_frames = -(-out_length // HOP_LENGTH) + 2
    read_positions = _source_positions(
        source, output, np.arange(out_frames, dtype=np.float64) * HOP_LENGTH
    )
    read_positions = np.clip(read_positions, 0.0, float(n_samples - 1))
    read_steps = read_positions / HOP_LENGTH

    mono = to_mono(audio).astype(np.float32)
    reference = librosa.stft(mono, n_fft=N_FFT, hop_length=HOP_LENGTH)
    max_frame = reference.shape[1] - 1
    read_frames = np.clip(read_steps.astype(np.int64), 0, max_frame)
    read_fraction = np.clip(read_steps - read_frames, 0.0, 1.0)

    expected = (
        2.0 * np.pi * HOP_LENGTH * np.arange(reference.shape[0]) / N_FFT
    )[:, None].astype(np.float64)

    # The phase to synthesise with is the running sum of one hop's worth of true
    # advance per output frame -- the accumulator that the old per-segment code
    # restarted at every beat. Here it runs once, over the whole track.
    advance = _phase_advance(reference, read_frames, expected)
    rotation = np.cumsum(advance, axis=1, dtype=np.float64)
    rotation -= advance
    # Frame 0 synthesises at the source's own phase, so the accumulator starts
    # there. Dropping this constant would leave every bin turned by a fixed
    # angle for the whole track -- an all-pass filter nobody asked for, which
    # smears exactly the transients this step is trying not to damage.
    rotation += np.angle(reference[:, read_frames[0]])[:, None]
    del advance, expected
    # Only the rotation relative to the source phase matters, and it is used as
    # `exp(1j*rotation)`, so wrapping it here costs nothing and keeps a track's
    # worth of accumulated radians from needing the range that float64 was
    # holding it in.
    rotation -= np.angle(reference[:, read_frames])
    rotation -= 2.0 * np.pi * np.round(rotation / (2.0 * np.pi))
    rotation = _lock_phase_to_peaks(np.abs(reference[:, read_frames]), rotation)

    if audio.ndim == 1:
        warped = _warp_channel(
            None, rotation, read_frames, read_fraction, out_length, spectrum=reference
        )
        return warped.astype(np.float32)
    del reference

    channels = [
        _warp_channel(
            np.ascontiguousarray(audio[:, index], dtype=np.float32),
            rotation,
            read_frames,
            read_fraction,
            out_length,
        )
        for index in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def describe_warp(analysis: TempoAnalysis, n_samples: int, sr: int) -> dict:
    """Report exactly what the warp does, without re-running beat detection.

    Re-detecting beats on warped audio and comparing is unreliable: the tracker
    flips octaves (reporting 136 BPM for a 68 BPM track) and finds a different
    number of beats each time, so the comparison measures tracker noise rather
    than the correction.
    """
    source, output = warp_anchors(analysis, n_samples, sr)
    source_spans = np.diff(source)
    output_spans = np.diff(output)
    ratios = np.divide(
        source_spans, output_spans, out=np.ones_like(source_spans), where=output_spans > 0
    )
    corrections = np.abs(ratios - 1.0) * 100
    corrected = corrections[corrections > 0]
    return {
        "segments_total": int(len(source_spans)),
        "segments_corrected": int(len(corrected)),
        "median_correction_pct": float(np.median(corrected)) if corrected.size else 0.0,
        "max_correction_pct": float(np.max(corrected)) if corrected.size else 0.0,
        "length_change_pct": (output[-1] / n_samples - 1) * 100 if n_samples else 0.0,
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
