"""Objective A/B measurement between an original and a processed signal.

This module decides nothing. It only measures, so that every other stage in the
restoration chain can be argued about with numbers instead of adjectives. It is
deliberately free of any framework, model or I/O dependency beyond numpy, scipy
and librosa, so the same code can run inside a worker, inside a test, or from
the command line.

Two measurement choices here are load-bearing and are not interchangeable with
the obvious simpler versions:

  * **Offset detection is two-stage.** A single sample-domain search over a
    +/-150ms window silently returns garbage as soon as the real drift exceeds
    the window -- it locks onto whatever spurious peak sits inside the range and
    reports a confident, wrong answer. Stage 1 correlates amplitude envelopes
    over +/-15s to find the region; stage 2 refines to the sample inside
    +/-120ms of that. Local drift is then *tracked* window by window, because
    drift that accumulates past the fine window cannot be found by searching
    around the global offset.

  * **Band deltas are median per-frame dB, gated to frames above the 40th
    percentile, and normalised against measured program gain.** Summing raw
    energy per band lets a handful of loud events decide the answer for the
    whole track, and an un-normalised delta reports a level change as a
    spectral change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import load_audio, resample, to_mono

# --- Analysis resolution -------------------------------------------------

N_FFT = 2048
HOP = 512

# Envelope rate for the coarse offset search. 200Hz resolves a 5ms step, which
# is far finer than stage 1 needs, while keeping a 5-minute track under 60k
# samples so a full cross-correlation stays instant.
ENVELOPE_SR = 200.0

COARSE_SEARCH_S = 15.0
FINE_SEARCH_S = 0.120

# Material used for the sample-domain refinement. The whole track would work but
# costs seconds; the loudest stretch carries the alignment information anyway.
FINE_MATERIAL_S = 60.0

# Local offset tracking. Each window searches around the *previous* window's
# result rather than around the global offset, so cumulative drift larger than
# the search radius is still followed.
LOCAL_WINDOW_S = 2.0
LOCAL_HOP_S = 1.0
LOCAL_SEARCH_S = 0.250

# A window that correlates this badly has no offset worth reporting: inside a
# time-warped region the best-matching position wanders, and letting those
# windows into the drift statistic measures the tracker's confusion rather than
# the drift. They still count against the correlation metrics, which is where
# that damage belongs.
LOCAL_CONFIDENCE_MIN = 0.5

# Frames quieter than this percentile of the original's frame energy are dropped
# before any spectral comparison. Near-silent frames have no meaningful band
# balance, and their ratios are numerically wild.
FRAME_PERCENTILE = 40.0

BANDS: tuple[tuple[str, float, float], ...] = (
    ("20_100", 20.0, 100.0),
    ("100_250", 100.0, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1k", 500.0, 1000.0),
    ("1k_2k", 1000.0, 2000.0),
    ("2k_5k", 2000.0, 5000.0),
    ("5k_8k", 5000.0, 8000.0),
    ("8k_10k", 8000.0, 10000.0),
    ("10k_16k", 10000.0, 16000.0),
    ("16k_22k", 16000.0, 22050.0),
)

# Level bins for the gain-vs-input-level curve. A downward expander shows up as
# a monotonic ramp across these; a clean gain change shows up as a flat line.
LEVEL_BIN_EDGES = np.arange(-100.0, 1.0, 6.0)
MIN_FRAMES_PER_BIN = 12

EPS = 1e-12


# --- Report structures ---------------------------------------------------


@dataclass
class SignalInfo:
    sample_rate: int
    channels: int
    samples: int
    duration_s: float
    peak: float
    peak_db: float
    clipped_samples: int
    lufs_integrated: float
    lra: float


@dataclass
class TimingMetrics:
    duration_delta_ms: float
    global_offset_ms: float
    coarse_offset_ms: float
    fine_offset_ms: float
    local_offset_ms: list[float] = field(default_factory=list)
    local_offset_times_s: list[float] = field(default_factory=list)
    offset_drift_ms: float = 0.0
    max_local_offset_ms: float = 0.0
    confident_windows: int = 0
    correlation_windows: list[float] = field(default_factory=list)
    correlation_min: float = 0.0
    correlation_median: float = 0.0
    correlation_mean: float = 0.0
    fine_correlation_min: float = 0.0
    fine_correlation_median: float = 0.0


@dataclass
class SpectralMetrics:
    program_gain_db: float
    band_delta_db: dict[str, float] = field(default_factory=dict)
    band_level_db_original: dict[str, float] = field(default_factory=dict)
    band_level_db_processed: dict[str, float] = field(default_factory=dict)
    centroid_hz_original: float = 0.0
    centroid_hz_processed: float = 0.0
    rolloff85_hz_original: float = 0.0
    rolloff85_hz_processed: float = 0.0
    rolloff99_hz_original: float = 0.0
    rolloff99_hz_processed: float = 0.0
    frames_used: int = 0


@dataclass
class TransientMetrics:
    attack_count_original: int
    attack_count_processed: int
    attack_count_delta_pct: float
    attack_slope_db_original: float
    attack_slope_db_processed: float
    attack_slope_delta_db: float
    decay_db_original: float
    decay_db_processed: float
    decay_delta_db: float


@dataclass
class DynamicMetrics:
    gain_curve_db: dict[str, float] = field(default_factory=dict)
    expansion_spread_db: float = 0.0
    quiet_gain_db: float = 0.0
    loud_gain_db: float = 0.0


@dataclass
class StereoMetrics:
    channels_original: int
    channels_processed: int
    channels_preserved: bool
    side_to_mid_db_original: float
    side_to_mid_db_processed: float
    side_to_mid_delta_db: float
    lr_correlation_original: float
    lr_correlation_processed: float


@dataclass
class ArtifactMetrics:
    hf_coherence: float
    hf_coherence_band_hz: tuple[float, float]
    hf_modulation_period_s: float
    hf_modulation_depth_db: float
    # Music has its own slow variation in high-band level, so the processed
    # signal's modulation depth means nothing on its own. Only the increase over
    # the source is attributable to processing.
    hf_modulation_depth_db_original: float
    hf_modulation_increase_db: float
    # Coherence cannot answer the question for a genuinely band-limited source:
    # there is nothing above the cutoff to be coherent *with*, so any extension
    # scores near zero and a good one is indistinguishable from noise. This
    # asks the answerable version instead -- does the high band rise and fall
    # with the music? Content derived from the source does; invented content
    # does not.
    hf_envelope_correlation: float
    max_sample_step_original: float
    max_sample_step_processed: float


@dataclass
class QualityReport:
    original: SignalInfo
    processed: SignalInfo
    timing: TimingMetrics
    spectral: SpectralMetrics
    transient: TransientMetrics
    dynamic: DynamicMetrics
    stereo: StereoMetrics
    artifacts: ArtifactMetrics

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One-screen readout of the measurements that decide accept/reject."""
        t, s, tr = self.timing, self.spectral, self.transient
        bands = "  ".join(
            f"{name}:{s.band_delta_db.get(name, float('nan')):+.2f}" for name, _, _ in BANDS
        )
        return "\n".join(
            [
                f"duration   {self.original.duration_s:.3f}s -> {self.processed.duration_s:.3f}s "
                f"({t.duration_delta_ms:+.1f} ms)",
                f"channels   {self.stereo.channels_original} -> {self.stereo.channels_processed}"
                f"{'' if self.stereo.channels_preserved else '   *** COLLAPSED ***'}",
                f"rate       {self.original.sample_rate} -> {self.processed.sample_rate}",
                f"offset     global {t.global_offset_ms:+.1f} ms   "
                f"drift {t.offset_drift_ms:.1f} ms   max local {t.max_local_offset_ms:+.1f} ms",
                f"corr       2s min {t.correlation_min:.3f}  median {t.correlation_median:.3f}   "
                f"50ms min {t.fine_correlation_min:.3f}",
                f"loudness   {self.original.lufs_integrated:.2f} -> "
                f"{self.processed.lufs_integrated:.2f} LUFS   "
                f"LRA {self.original.lra:.2f} -> {self.processed.lra:.2f}",
                f"clipping   {self.original.clipped_samples} -> {self.processed.clipped_samples}",
                f"gain       program {s.program_gain_db:+.2f} dB",
                f"bands dB   {bands}",
                f"rolloff85  {s.rolloff85_hz_original:.0f} -> {s.rolloff85_hz_processed:.0f} Hz",
                f"attacks    {tr.attack_count_original} -> {tr.attack_count_processed} "
                f"({tr.attack_count_delta_pct:+.1f}%)   slope {tr.attack_slope_delta_db:+.2f} dB   "
                f"decay {tr.decay_delta_db:+.2f} dB",
                f"expansion  spread {self.dynamic.expansion_spread_db:.2f} dB   "
                f"quiet {self.dynamic.quiet_gain_db:+.2f} dB   "
                f"loud {self.dynamic.loud_gain_db:+.2f} dB",
                f"stereo     side/mid {self.stereo.side_to_mid_db_original:.1f} -> "
                f"{self.stereo.side_to_mid_db_processed:.1f} dB",
                f"artifacts  HF coherence {self.artifacts.hf_coherence:.3f}   "
                f"HF modulation +{self.artifacts.hf_modulation_increase_db:.2f} dB @ "
                f"{self.artifacts.hf_modulation_period_s:.1f}s",
            ]
        )


# --- Loudness (ITU-R BS.1770 / EBU R128) ---------------------------------


def _biquad_high_shelf(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 stage 1: the head-shadow high shelf."""
    fc, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    a_gain = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    root = 2 * np.sqrt(a_gain) * alpha
    b = np.array([
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + root),
        -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0),
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - root),
    ])
    a = np.array([
        (a_gain + 1) - (a_gain - 1) * cos_w0 + root,
        2 * ((a_gain - 1) - (a_gain + 1) * cos_w0),
        (a_gain + 1) - (a_gain - 1) * cos_w0 - root,
    ])
    return b / a[0], a / a[0]


def _biquad_high_pass(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 stage 2: the RLB high-pass."""
    fc, q = 38.13547087602444, 0.5003270373238773
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2])
    a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
    return b / a[0], a / a[0]


def _k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting curve along the sample axis."""
    from scipy.signal import lfilter

    out = np.asarray(audio, dtype=np.float64)
    for b, a in (_biquad_high_shelf(sr), _biquad_high_pass(sr)):
        out = lfilter(b, a, out, axis=0)
    return out


def _block_loudness(audio: np.ndarray, sr: int, block_s: float, hop_s: float) -> np.ndarray:
    """Gated-block loudness in LKFS for every block of `block_s` seconds."""
    signal = audio if audio.ndim == 2 else audio[:, None]
    weighted = _k_weight(signal, sr)
    block = int(round(block_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    if weighted.shape[0] < block:
        return np.array([])

    starts = np.arange(0, weighted.shape[0] - block + 1, hop)
    # Channel weights: L/R/C are 1.0. Surround channels would be 1.41, but stem
    # material here is never more than stereo.
    power = np.empty(len(starts))
    for i, start in enumerate(starts):
        segment = weighted[start:start + block]
        power[i] = np.sum(np.mean(np.square(segment), axis=0))
    return -0.691 + 10 * np.log10(power + EPS)


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    """EBU R128 integrated loudness, with both gates applied."""
    loudness = _block_loudness(audio, sr, block_s=0.400, hop_s=0.100)
    if loudness.size == 0:
        return float("-inf")
    above_absolute = loudness[loudness > -70.0]
    if above_absolute.size == 0:
        return float("-inf")
    power = 10 ** ((above_absolute + 0.691) / 10)
    relative_gate = -0.691 + 10 * np.log10(power.mean() + EPS) - 10.0
    gated = above_absolute[above_absolute > relative_gate]
    if gated.size == 0:
        return float(above_absolute.mean())
    power = 10 ** ((gated + 0.691) / 10)
    return float(-0.691 + 10 * np.log10(power.mean() + EPS))


def loudness_range(audio: np.ndarray, sr: int) -> float:
    """EBU Tech 3342 loudness range, in LU."""
    loudness = _block_loudness(audio, sr, block_s=3.0, hop_s=0.100)
    if loudness.size == 0:
        return 0.0
    above_absolute = loudness[loudness > -70.0]
    if above_absolute.size < 2:
        return 0.0
    power = 10 ** ((above_absolute + 0.691) / 10)
    relative_gate = -0.691 + 10 * np.log10(power.mean() + EPS) - 20.0
    gated = above_absolute[above_absolute > relative_gate]
    if gated.size < 2:
        return 0.0
    return float(np.percentile(gated, 95) - np.percentile(gated, 10))


# --- Alignment -----------------------------------------------------------


def _envelope(mono: np.ndarray, sr: int, env_sr: float = ENVELOPE_SR) -> np.ndarray:
    """Downsampled RMS envelope, mean-removed and unit-scaled for correlation."""
    hop = max(1, int(round(sr / env_sr)))
    trimmed = mono[: (len(mono) // hop) * hop]
    if trimmed.size == 0:
        return np.zeros(0)
    blocks = trimmed.reshape(-1, hop).astype(np.float64)
    env = np.sqrt(np.mean(blocks * blocks, axis=1))
    env -= env.mean()
    norm = np.linalg.norm(env)
    return env / norm if norm > 0 else env


def _sliding_correlation(short: np.ndarray, long: np.ndarray) -> np.ndarray:
    """Normalised correlation of `short` at every position it fits inside `long`.

    Each position is normalised by the norm of *that window* of `long`, not by
    the norm of the whole slice. Normalising once against the full slice makes
    every score shrink as the search range widens, which silently turns a
    perfect match into a mediocre-looking one -- the search still finds the
    right position, but the correlation it reports is wrong.
    """
    from scipy.signal import fftconvolve

    width = short.size
    count = long.size - width + 1
    if width == 0 or count <= 0:
        return np.zeros(0)

    centred = short - short.mean()
    short_norm = np.linalg.norm(centred)
    if short_norm <= 0:
        return np.zeros(count)

    # `centred` sums to zero, so correlating against the raw window is already
    # equivalent to correlating against the mean-removed window.
    numerator = fftconvolve(long, centred[::-1], mode="valid")[:count]

    cumulative = np.concatenate([[0.0], np.cumsum(long)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(long * long)])
    total = cumulative[width:] - cumulative[:-width]
    total_sq = cumulative_sq[width:] - cumulative_sq[:-width]
    variance = np.maximum(total_sq - total * total / width, 0.0)[:count]

    denominator = short_norm * np.sqrt(variance)
    return np.divide(numerator, denominator, out=np.zeros(count), where=denominator > 0)


def _best_position(short: np.ndarray, long: np.ndarray) -> tuple[int, float]:
    """Index into `long` where `short` fits best, and how well it fits."""
    scores = _sliding_correlation(short, long)
    if scores.size == 0:
        return 0, 0.0
    index = int(np.argmax(scores))
    return index, float(scores[index])


def _best_lag(reference: np.ndarray, test: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag maximising cross-correlation, for series of comparable length.

    Used only by the coarse envelope stage, where the two series cover the whole
    track and only the argmax matters.
    """
    from scipy.signal import correlate, correlation_lags

    if reference.size == 0 or test.size == 0:
        return 0, 0.0
    a = reference - reference.mean()
    b = test - test.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return 0, 0.0
    raw = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(a.size, b.size, mode="full")
    keep = np.abs(lags) <= max_lag
    if not keep.any():
        return 0, 0.0
    raw, lags = raw[keep], lags[keep]
    index = int(np.argmax(raw))
    return int(lags[index]), float(raw[index] / denominator)


@dataclass
class OffsetEstimate:
    """Positive means the processed signal is *late* relative to the original.

    The sign convention is fixed here and relied on everywhere downstream: an
    offset is the index into the processed signal at which the original's first
    sample appears.
    """

    coarse_samples: int
    fine_samples: int
    total_samples: int
    coarse_score: float
    fine_score: float

    def total_ms(self, sr: int) -> float:
        return self.total_samples / sr * 1000.0


def estimate_offset(
    reference: np.ndarray,
    test: np.ndarray,
    sr: int,
    coarse_s: float = COARSE_SEARCH_S,
    fine_s: float = FINE_SEARCH_S,
) -> OffsetEstimate:
    """Two-stage global offset: envelope search, then sample-domain refinement.

    Stage 1 finds the region -- it can see a 15-second displacement that no
    sample-domain window would reach. Stage 2 then pins the exact sample inside
    +/-`fine_s` of the stage 1 answer.
    """
    ref_mono = to_mono(reference).astype(np.float64)
    test_mono = to_mono(test).astype(np.float64)

    env_hop = max(1, int(round(sr / ENVELOPE_SR)))
    coarse_lag_env, coarse_score = _best_lag(
        _envelope(ref_mono, sr), _envelope(test_mono, sr), int(round(coarse_s * ENVELOPE_SR))
    )
    # `_best_lag` reports the shift of the reference against the test; negating
    # it gives the delay of the test, which is the convention documented on
    # OffsetEstimate and assumed by every caller.
    coarse_samples = -coarse_lag_env * env_hop

    # Refine on the loudest stretch: silence carries no alignment information
    # and only adds cost.
    # The reference slice is deliberately shorter than the material available in
    # the test signal. Taking the whole track leaves nothing to slide against --
    # the refinement silently does nothing, and the answer stays quantised to
    # the envelope hop, which is 5ms of avoidable error.
    pad = int(fine_s * sr)
    span = min(len(ref_mono), len(test_mono), int(FINE_MATERIAL_S * sr)) - 4 * pad
    fine_samples, fine_score = 0, 0.0

    if span > sr:
        lowest = max(0, pad - coarse_samples)
        highest = min(len(ref_mono) - span, len(test_mono) - coarse_samples - span - pad)
        if highest >= lowest:
            start = int(np.clip(_loudest_start(ref_mono, span), lowest, highest))
            ref_slice = ref_mono[start:start + span]
            test_start = start + coarse_samples
            lo = test_start - pad
            test_slice = test_mono[lo:lo + span + 2 * pad]
            position, fine_score = _best_position(ref_slice, test_slice)
            fine_samples = int(np.clip((lo + position) - test_start, -pad, pad))

    return OffsetEstimate(
        coarse_samples=coarse_samples,
        fine_samples=fine_samples,
        total_samples=coarse_samples + fine_samples,
        coarse_score=coarse_score,
        fine_score=fine_score,
    )


def _loudest_start(mono: np.ndarray, span: int) -> int:
    """Start index of the highest-energy `span`-sample stretch."""
    if len(mono) <= span:
        return 0
    hop = max(1, len(mono) // 512)
    energy = np.cumsum(np.square(mono, dtype=np.float64))
    starts = np.arange(0, len(mono) - span, hop)
    totals = energy[starts + span] - energy[starts]
    return int(starts[int(np.argmax(totals))])


def _aligned_pair(
    original: np.ndarray, processed: np.ndarray, offset_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trim both signals to the region where they overlap after alignment."""
    if offset_samples >= 0:
        a, b = original, processed[offset_samples:]
    else:
        a, b = original[-offset_samples:], processed
    n = min(len(a), len(b))
    return a[:n], b[:n]


def track_local_offsets(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    global_offset: int,
    window_s: float = LOCAL_WINDOW_S,
    hop_s: float = LOCAL_HOP_S,
    search_s: float = LOCAL_SEARCH_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow the local offset window by window, and correlate at each position.

    Each window searches around the *previous* window's offset. Searching around
    the global offset instead would lose the signal as soon as accumulated drift
    passed the search radius -- which is exactly the failure mode a warped region
    produces.

    Returns (times_s, offsets_samples, correlations).
    """
    ref = to_mono(original).astype(np.float64)
    test = to_mono(processed).astype(np.float64)
    window = int(window_s * sr)
    hop = int(hop_s * sr)
    search = int(search_s * sr)
    if window <= 0 or len(ref) < window:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    times, offsets, scores = [], [], []
    running = 0
    locked = False
    for start in range(0, len(ref) - window, hop):
        chunk = ref[start:start + window]
        if np.sqrt(np.mean(chunk**2)) < 1e-5:  # silence has no offset to find
            continue

        # The first window must not be seeded with the global offset. The global
        # offset is whatever the loudest stretch of the track says, which on a
        # drifting file is the value *after* the drift has accumulated -- seeding
        # with it puts the true head offset outside the search radius, and the
        # tracker then reports drift that starts at the wrong end. So the first
        # window searches wide enough to reach either end of the drift.
        radius = search if locked else max(search, abs(global_offset) + search)
        centre = start + running
        lo, hi = max(0, centre - radius), min(len(test), centre + window + radius)
        candidate = test[lo:hi]
        if candidate.size <= window:
            continue
        position, score = _best_position(chunk, candidate)
        offset = (lo + position) - start
        if locked and abs(offset - running) > search:  # lost; keep the last good value
            continue
        running, locked = offset, True
        times.append(start / sr)
        offsets.append(offset)
        scores.append(score)
    return np.array(times), np.array(offsets), np.array(scores)


def _fine_correlations(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    times: np.ndarray,
    offsets: np.ndarray,
    window_s: float = 0.050,
) -> np.ndarray:
    """Correlation over short windows, each aligned by the tracked local offset.

    A time-warp shows up here even when a 2-second window still correlates: at
    50ms the smearing a phase vocoder leaves behind cannot be hidden by getting
    the gross alignment right.
    """
    ref = to_mono(original).astype(np.float64)
    test = to_mono(processed).astype(np.float64)
    window = int(window_s * sr)
    scores = []
    for time_s, offset in zip(times, offsets):
        start = int(time_s * sr)
        chunk = ref[start:start + window]
        other = test[start + int(offset):start + int(offset) + window]
        if chunk.size < window or other.size < window:
            continue
        a, b = chunk - chunk.mean(), other - other.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator > 0:
            scores.append(float(np.dot(a, b) / denominator))
    return np.array(scores)


# --- Spectral ------------------------------------------------------------


def _stft_db(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    spec = np.abs(librosa.stft(mono.astype(np.float32), n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    return freqs, spec


def _band_frame_db(spec: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return np.full(spec.shape[1], -np.inf)
    return 20 * np.log10(np.sqrt(np.mean(np.square(spec[mask]), axis=0)) + EPS)


def measure_spectral(original: np.ndarray, processed: np.ndarray, sr: int) -> SpectralMetrics:
    """Per-band median frame delta, gated to loud frames and de-gained.

    `program_gain_db` is the median broadband frame delta over the same gated
    frames. Subtracting it is what turns "this got quieter" into "this got
    duller", which is the only version of the question worth asking.
    """
    import librosa

    ref, test = to_mono(original), to_mono(processed)
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]

    freqs, spec_ref = _stft_db(ref, sr)
    _, spec_test = _stft_db(test, sr)
    frames = min(spec_ref.shape[1], spec_test.shape[1])
    spec_ref, spec_test = spec_ref[:, :frames], spec_test[:, :frames]

    broadband_ref = 20 * np.log10(np.sqrt(np.mean(np.square(spec_ref), axis=0)) + EPS)
    broadband_test = 20 * np.log10(np.sqrt(np.mean(np.square(spec_test), axis=0)) + EPS)
    if frames == 0:
        return SpectralMetrics(program_gain_db=0.0)

    threshold = np.percentile(broadband_ref, FRAME_PERCENTILE)
    loud = broadband_ref >= threshold
    if loud.sum() < 4:
        loud = np.ones(frames, dtype=bool)

    program_gain = float(np.median(broadband_test[loud] - broadband_ref[loud]))

    deltas, level_ref, level_test = {}, {}, {}
    nyquist = sr / 2
    for name, low, high in BANDS:
        if low >= nyquist:
            continue
        band_ref = _band_frame_db(spec_ref, freqs, low, min(high, nyquist))
        band_test = _band_frame_db(spec_test, freqs, low, min(high, nyquist))
        usable = loud & np.isfinite(band_ref) & np.isfinite(band_test)
        if usable.sum() < 4:
            continue
        deltas[name] = float(np.median(band_test[usable] - band_ref[usable]) - program_gain)
        level_ref[name] = float(np.median(band_ref[usable]))
        level_test[name] = float(np.median(band_test[usable]))

    return SpectralMetrics(
        program_gain_db=program_gain,
        band_delta_db=deltas,
        band_level_db_original=level_ref,
        band_level_db_processed=level_test,
        centroid_hz_original=float(np.mean(librosa.feature.spectral_centroid(S=spec_ref, sr=sr))),
        centroid_hz_processed=float(np.mean(librosa.feature.spectral_centroid(S=spec_test, sr=sr))),
        rolloff85_hz_original=float(
            np.mean(librosa.feature.spectral_rolloff(S=spec_ref, sr=sr, roll_percent=0.85))
        ),
        rolloff85_hz_processed=float(
            np.mean(librosa.feature.spectral_rolloff(S=spec_test, sr=sr, roll_percent=0.85))
        ),
        rolloff99_hz_original=float(
            np.mean(librosa.feature.spectral_rolloff(S=spec_ref, sr=sr, roll_percent=0.99))
        ),
        rolloff99_hz_processed=float(
            np.mean(librosa.feature.spectral_rolloff(S=spec_test, sr=sr, roll_percent=0.99))
        ),
        frames_used=int(loud.sum()),
    )


# --- Transients ----------------------------------------------------------


def _frame_db(mono: np.ndarray, sr: int, start: int, length_s: float) -> float:
    length = max(1, int(length_s * sr))
    chunk = mono[max(0, start):max(0, start) + length]
    if chunk.size == 0:
        return -np.inf
    return float(20 * np.log10(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) + EPS))


def measure_transients(
    original: np.ndarray, processed: np.ndarray, sr: int
) -> TransientMetrics:
    """Attack count, attack slope and note decay, measured at the same instants.

    Onsets are detected on the original and then *reused* as sample positions in
    the processed signal. Detecting independently on each and comparing averages
    would measure the detector's own instability instead of the processing.
    """
    import librosa

    ref, test = to_mono(original), to_mono(processed)
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float32), test[:n].astype(np.float32)

    onsets_ref = librosa.onset.onset_detect(y=ref, sr=sr, units="samples", hop_length=HOP)
    onsets_test = librosa.onset.onset_detect(y=test, sr=sr, units="samples", hop_length=HOP)

    slopes_ref, slopes_test, decay_ref, decay_test = [], [], [], []
    for onset in onsets_ref:
        pre = _frame_db(ref, sr, onset - int(0.025 * sr), 0.020)
        peak = _frame_db(ref, sr, onset, 0.030)
        pre_t = _frame_db(test, sr, onset - int(0.025 * sr), 0.020)
        peak_t = _frame_db(test, sr, onset, 0.030)
        if np.isfinite(pre) and np.isfinite(peak) and np.isfinite(pre_t) and np.isfinite(peak_t):
            slopes_ref.append(peak - pre)
            slopes_test.append(peak_t - pre_t)
        tail = _frame_db(ref, sr, onset + int(0.400 * sr), 0.050)
        tail_t = _frame_db(test, sr, onset + int(0.400 * sr), 0.050)
        if np.isfinite(tail) and np.isfinite(peak) and np.isfinite(tail_t) and np.isfinite(peak_t):
            decay_ref.append(tail - peak)
            decay_test.append(tail_t - peak_t)

    slope_ref = float(np.median(slopes_ref)) if slopes_ref else 0.0
    slope_test = float(np.median(slopes_test)) if slopes_test else 0.0
    fall_ref = float(np.median(decay_ref)) if decay_ref else 0.0
    fall_test = float(np.median(decay_test)) if decay_test else 0.0
    count_ref = len(onsets_ref)

    return TransientMetrics(
        attack_count_original=count_ref,
        attack_count_processed=len(onsets_test),
        attack_count_delta_pct=(
            (len(onsets_test) - count_ref) / count_ref * 100.0 if count_ref else 0.0
        ),
        attack_slope_db_original=slope_ref,
        attack_slope_db_processed=slope_test,
        # Attack slope is already a difference of two levels, so a uniform gain
        # change cancels out of it; no program-gain correction is applied.
        attack_slope_delta_db=slope_test - slope_ref,
        decay_db_original=fall_ref,
        decay_db_processed=fall_test,
        decay_delta_db=fall_test - fall_ref,
    )


# --- Dynamics ------------------------------------------------------------


def measure_dynamics(original: np.ndarray, processed: np.ndarray, sr: int) -> DynamicMetrics:
    """Median gain applied at each input level.

    A denoiser that is really a downward expander cannot hide here: its gain
    curve slopes steeply across the quiet bins while staying near 0 dB on loud
    material. `expansion_spread_db` is the height of that slope.
    """
    ref, test = to_mono(original), to_mono(processed)
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]

    window, hop = N_FFT, HOP
    if n < window:
        return DynamicMetrics()
    starts = np.arange(0, n - window, hop)
    ref_db = np.empty(len(starts))
    gain = np.empty(len(starts))
    for i, start in enumerate(starts):
        a = ref[start:start + window].astype(np.float64)
        b = test[start:start + window].astype(np.float64)
        rms_a = np.sqrt(np.mean(a * a))
        rms_b = np.sqrt(np.mean(b * b))
        ref_db[i] = 20 * np.log10(rms_a + EPS)
        gain[i] = 20 * np.log10(rms_b + EPS) - ref_db[i]

    curve = {}
    for low, high in zip(LEVEL_BIN_EDGES[:-1], LEVEL_BIN_EDGES[1:]):
        inside = (ref_db >= low) & (ref_db < high)
        if inside.sum() >= MIN_FRAMES_PER_BIN:
            curve[f"{low:.0f}_{high:.0f}"] = float(np.median(gain[inside]))
    if not curve:
        return DynamicMetrics()

    values = list(curve.values())
    keys = sorted(curve, key=lambda k: float(k.split("_")[0]))
    return DynamicMetrics(
        gain_curve_db=curve,
        expansion_spread_db=float(max(values) - min(values)),
        quiet_gain_db=curve[keys[0]],
        loud_gain_db=curve[keys[-1]],
    )


# --- Stereo --------------------------------------------------------------


def measure_stereo(original: np.ndarray, processed: np.ndarray) -> StereoMetrics:
    def side_to_mid(audio: np.ndarray) -> float:
        if audio.ndim != 2 or audio.shape[1] < 2:
            return float("-inf")
        mid = (audio[:, 0] + audio[:, 1]) / 2
        side = (audio[:, 0] - audio[:, 1]) / 2
        mid_rms = np.sqrt(np.mean(np.square(mid, dtype=np.float64)))
        side_rms = np.sqrt(np.mean(np.square(side, dtype=np.float64)))
        return float(20 * np.log10((side_rms + EPS) / (mid_rms + EPS)))

    def lr_correlation(audio: np.ndarray) -> float:
        if audio.ndim != 2 or audio.shape[1] < 2:
            return 1.0
        left = audio[:, 0] - audio[:, 0].mean()
        right = audio[:, 1] - audio[:, 1].mean()
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator > 0 else 1.0

    channels_a = 1 if original.ndim == 1 else original.shape[1]
    channels_b = 1 if processed.ndim == 1 else processed.shape[1]
    a_ratio, b_ratio = side_to_mid(original), side_to_mid(processed)
    delta = b_ratio - a_ratio if np.isfinite(a_ratio) and np.isfinite(b_ratio) else float("-inf")
    return StereoMetrics(
        channels_original=channels_a,
        channels_processed=channels_b,
        channels_preserved=channels_a == channels_b,
        side_to_mid_db_original=a_ratio,
        side_to_mid_db_processed=b_ratio,
        side_to_mid_delta_db=delta,
        lr_correlation_original=lr_correlation(original),
        lr_correlation_processed=lr_correlation(processed),
    )


# --- Artifacts -----------------------------------------------------------


def measure_artifacts(
    original: np.ndarray, processed: np.ndarray, sr: int, hf_low: float = 10000.0
) -> ArtifactMetrics:
    """Is the added high band related to the source, and does it pulse?

    Coherence near zero above the cutoff means the model invented the high band
    rather than restoring it. A strong periodic swing in that band's level means
    the chunking is showing through: the period lands on the chunk stride.
    """
    from scipy.signal import coherence

    ref, test = to_mono(original), to_mono(processed)
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    nyquist = sr / 2
    hf_high = min(nyquist, 20000.0)

    hf_coherence = float("nan")
    if n > 8192 and hf_low < nyquist:
        freqs, values = coherence(ref, test, fs=sr, nperseg=8192)
        band = (freqs >= hf_low) & (freqs <= hf_high)
        if band.any():
            hf_coherence = float(np.mean(values[band]))

    period_s, depth_db = _hf_modulation(test, sr, hf_low)
    _, depth_reference = _hf_modulation(ref, sr, hf_low)
    envelope_correlation = _hf_envelope_correlation(ref, test, sr, hf_low)

    def max_step(audio: np.ndarray) -> float:
        if audio.size < 2:
            return 0.0
        peak = float(np.abs(audio).max())
        return float(np.abs(np.diff(audio)).max() / peak) if peak > 0 else 0.0

    return ArtifactMetrics(
        hf_coherence=hf_coherence,
        hf_coherence_band_hz=(hf_low, hf_high),
        hf_modulation_period_s=period_s,
        hf_modulation_depth_db=depth_db,
        hf_modulation_depth_db_original=depth_reference,
        hf_modulation_increase_db=max(0.0, depth_db - depth_reference),
        hf_envelope_correlation=envelope_correlation,
        max_sample_step_original=max_step(ref),
        max_sample_step_processed=max_step(test),
    )


def _hf_envelope_correlation(
    reference: np.ndarray, test: np.ndarray, sr: int, hf_low: float
) -> float:
    """Does the processed high band follow the source's own loudness contour?

    A bandwidth extender that derives its output from the material below the
    cutoff produces a high band that rises and falls with the music. One that
    invents a band produces something with its own, unrelated contour -- often
    nearly steady, since noise has a flat envelope.

    This is the test that still works when the source has nothing above the
    cutoff, which is exactly the case bandwidth extension exists for and exactly
    the case coherence cannot judge.
    """
    if hf_low >= sr / 2 or min(reference.size, test.size) < sr:
        return float("nan")

    _, spec_ref = _stft_db(reference, sr)
    freqs, spec_test = _stft_db(test, sr)
    frames = min(spec_ref.shape[1], spec_test.shape[1])
    if frames < 16:
        return float("nan")

    band = freqs >= hf_low
    if not band.any():
        return float("nan")

    high = 20 * np.log10(np.sqrt(np.mean(np.square(spec_test[band, :frames]), axis=0)) + EPS)
    broadband = 20 * np.log10(np.sqrt(np.mean(np.square(spec_ref[:, :frames]), axis=0)) + EPS)

    usable = np.isfinite(high) & np.isfinite(broadband)
    if usable.sum() < 16:
        return float("nan")
    a = high[usable] - high[usable].mean()
    b = broadband[usable] - broadband[usable].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _hf_modulation(mono: np.ndarray, sr: int, hf_low: float) -> tuple[float, float]:
    """Dominant period and depth of any slow level swing in the band above `hf_low`."""
    if mono.size < sr * 4 or hf_low >= sr / 2:
        return 0.0, 0.0
    freqs, spec = _stft_db(mono, sr)
    band = freqs >= hf_low
    if not band.any() or spec.shape[1] < 16:
        return 0.0, 0.0
    level = 20 * np.log10(np.sqrt(np.mean(np.square(spec[band]), axis=0)) + EPS)
    level = level[np.isfinite(level)]
    if level.size < 16:
        return 0.0, 0.0
    frame_rate = sr / HOP
    # Only the mean is removed. Subtracting a short moving average would
    # high-pass exactly the slow swing this function exists to find.
    detrended = level - level.mean()
    window = np.hanning(detrended.size)
    magnitude = np.abs(np.fft.rfft(detrended * window))
    periods = np.divide(
        detrended.size / frame_rate,
        np.arange(magnitude.size),
        out=np.full(magnitude.size, np.inf),
        where=np.arange(magnitude.size) > 0,
    )
    # Only slow swings are chunk artifacts; anything faster is musical.
    usable = (periods >= 2.0) & (periods <= 30.0)
    if not usable.any():
        return 0.0, 0.0
    index = int(np.argmax(np.where(usable, magnitude, 0.0)))
    depth = float(magnitude[index] * 4 / detrended.size)

    # A slow swing spans very few bins -- a 9s period in a 15s excerpt sits
    # between bin 1 and bin 2 -- so the bin index alone is a coarse answer.
    # Parabolic interpolation on the peak recovers the period between bins.
    bin_index = float(index)
    if 0 < index < magnitude.size - 1:
        left, centre, right = magnitude[index - 1], magnitude[index], magnitude[index + 1]
        denominator = left - 2 * centre + right
        if denominator != 0:
            bin_index += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    if bin_index <= 0:
        return float(periods[index]), depth
    return float(detrended.size / frame_rate / bin_index), depth


# --- Top level -----------------------------------------------------------


def describe(audio: np.ndarray, sr: int) -> SignalInfo:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return SignalInfo(
        sample_rate=sr,
        channels=1 if audio.ndim == 1 else audio.shape[1],
        samples=int(audio.shape[0]),
        duration_s=float(audio.shape[0] / sr) if sr else 0.0,
        peak=peak,
        peak_db=float(20 * np.log10(peak)) if peak > 0 else float("-inf"),
        clipped_samples=int(np.count_nonzero(np.abs(audio) >= 0.999969)),
        lufs_integrated=integrated_lufs(audio, sr),
        lra=loudness_range(audio, sr),
    )


def compare(
    original: np.ndarray,
    processed: np.ndarray,
    sr_in: int,
    sr_out: int | None = None,
) -> QualityReport:
    """Measure `processed` against `original`, reporting every axis at once.

    Differing sample rates are reconciled by resampling the processed signal to
    `sr_in` before comparison, so a rate change is reported in the signal info
    rather than smeared across every other metric.
    """
    sr_out = sr_out or sr_in
    original = np.asarray(original)
    processed = np.asarray(processed)

    info_original = describe(original, sr_in)
    info_processed = describe(processed, sr_out)

    work = resample(processed, sr_out, sr_in) if sr_out != sr_in else processed

    offset = estimate_offset(original, work, sr_in)
    aligned_ref, aligned_test = _aligned_pair(original, work, offset.total_samples)

    times, offsets, scores = track_local_offsets(original, work, sr_in, offset.total_samples)
    fine_scores = _fine_correlations(original, work, sr_in, times, offsets)
    offsets_ms = offsets / sr_in * 1000.0 if offsets.size else np.zeros(0)
    confident = offsets_ms[scores >= LOCAL_CONFIDENCE_MIN] if offsets_ms.size else np.zeros(0)
    if confident.size == 0:
        confident = offsets_ms

    timing = TimingMetrics(
        duration_delta_ms=(info_processed.duration_s - info_original.duration_s) * 1000.0,
        global_offset_ms=offset.total_ms(sr_in),
        coarse_offset_ms=offset.coarse_samples / sr_in * 1000.0,
        fine_offset_ms=offset.fine_samples / sr_in * 1000.0,
        local_offset_ms=[float(v) for v in offsets_ms],
        local_offset_times_s=[float(v) for v in times],
        offset_drift_ms=float(confident.max() - confident.min()) if confident.size else 0.0,
        max_local_offset_ms=(
            float(confident[np.argmax(np.abs(confident))]) if confident.size else 0.0
        ),
        confident_windows=int(confident.size),
        correlation_windows=[float(v) for v in scores],
        correlation_min=float(scores.min()) if scores.size else 0.0,
        correlation_median=float(np.median(scores)) if scores.size else 0.0,
        correlation_mean=float(scores.mean()) if scores.size else 0.0,
        fine_correlation_min=float(fine_scores.min()) if fine_scores.size else 0.0,
        fine_correlation_median=float(np.median(fine_scores)) if fine_scores.size else 0.0,
    )

    spectral = measure_spectral(aligned_ref, aligned_test, sr_in)
    return QualityReport(
        original=info_original,
        processed=info_processed,
        timing=timing,
        spectral=spectral,
        transient=measure_transients(aligned_ref, aligned_test, sr_in),
        dynamic=measure_dynamics(aligned_ref, aligned_test, sr_in),
        stereo=measure_stereo(original, processed),
        artifacts=measure_artifacts(aligned_ref, aligned_test, sr_in),
    )


def compare_files(original: str | Path, processed: str | Path) -> QualityReport:
    """Compare two files at the original's own sample rate."""
    reference, sr_in = load_audio(original, target_sr=None)
    candidate, sr_out = load_audio(processed, target_sr=None)
    return compare(reference, candidate, sr_in, sr_out)


__all__ = [
    "ArtifactMetrics",
    "DynamicMetrics",
    "OffsetEstimate",
    "QualityReport",
    "SignalInfo",
    "SpectralMetrics",
    "StereoMetrics",
    "TimingMetrics",
    "compare",
    "compare_files",
    "describe",
    "estimate_offset",
    "integrated_lufs",
    "loudness_range",
    "measure_artifacts",
    "measure_dynamics",
    "measure_spectral",
    "measure_stereo",
    "measure_transients",
    "track_local_offsets",
]
