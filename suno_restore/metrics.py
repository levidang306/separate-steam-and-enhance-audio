"""Measurements reported alongside each restoration step.

These describe what a step did; they never decide whether it runs.
"""

from __future__ import annotations

import librosa
import numpy as np

from .audio_io import to_mono

N_FFT = 8192
HOP = 2048


def _spectrum_db(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    y = to_mono(audio)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    # A high percentile rather than the mean: silence between notes drags the
    # mean down and smears a cutoff that is actually sharp.
    spec = np.percentile(stft, 95, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    return freqs, 20 * np.log10(spec + 1e-12)


def spectral_cliff_hz(
    audio: np.ndarray,
    sr: int,
    min_drop_db: float = 25.0,
    window_hz: float = 1500.0,
    floor_hz: float = 2000.0,
) -> float | None:
    """Highest frequency where the spectrum falls `min_drop_db` within `window_hz`.

    A codec or generator cutoff appears as a cliff. Measuring level relative to
    the spectral peak instead would just report tonal balance -- it reads a bass
    stem as "cut at 400Hz" purely because bass has no treble to begin with.

    `floor_hz` is where the search starts, and it is a real limit rather than a
    formality: a cutoff *below* it cannot be found at all, because by the time
    the search begins the signal is already at the noise floor and there is no
    further drop left to detect. A file low-passed at 1500Hz reads as having no
    cliff whatsoever under the 2000Hz default, despite falling 73dB between
    1400Hz and 1800Hz. Callers that can tolerate the false positives this guards
    against -- a bass stem's natural rolloff -- should lower it and lean on the
    corroborating evidence instead (energy below the cliff, and whether a model's
    output is actually derived from the source).
    """
    freqs, spec_db = _spectrum_db(audio, sr)
    bin_hz = freqs[1] - freqs[0]
    window = max(1, int(window_hz / bin_hz))
    cliff = None
    for i in range(len(spec_db) - window):
        if freqs[i] < floor_hz:
            continue
        if spec_db[i] - spec_db[i + window] >= min_drop_db:
            cliff = float(freqs[i])
    return cliff


def energy_above_hz(audio: np.ndarray, sr: int, cutoff_hz: float = 13000.0) -> float:
    """Share of spectral energy above `cutoff_hz`, as a percentage."""
    freqs, spec_db = _spectrum_db(audio, sr)
    spec = 10 ** (spec_db / 20)
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float(spec[freqs >= cutoff_hz].sum() / total * 100)


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def residual_level_db(original: np.ndarray, cleaned: np.ndarray) -> float:
    """Level of what a step removed, relative to the input.

    Near 0 dB means the step rewrote the signal; -50 dB means it barely touched
    it. This is the honest read on whether a denoiser found anything to remove.
    """
    n = min(len(original), len(cleaned))
    residual = to_mono(original[:n]) - to_mono(cleaned[:n])
    return rms_db(residual) - rms_db(to_mono(original[:n]))


def tempo_stability(
    beat_times: np.ndarray, target_interval_s: float | None = None
) -> dict[str, float]:
    """Summarise how steady a beat sequence is, in BPM.

    Gaps are divided by the number of beats they span before being converted to
    BPM. Beat trackers skip beats wherever a stem falls quiet, so reading raw
    consecutive-beat gaps as tempo reports a 28-second silence as 2 BPM and
    swamps any real drift.
    """
    if len(beat_times) < 4:
        return {"median_bpm": float("nan"), "std_bpm": float("nan"),
                "min_bpm": float("nan"), "max_bpm": float("nan"),
                "n_beats": float(len(beat_times))}

    gaps = np.diff(beat_times)
    interval = target_interval_s or float(np.median(gaps))
    spanned = np.maximum(1, np.round(gaps / interval))
    per_beat = gaps / spanned
    # Beats detected implausibly close together are spurious, not fast tempo;
    # left in, a 0.16s gap reports as 375 BPM and dominates the spread.
    plausible = (per_beat > interval * 0.5) & (per_beat < interval * 2.0)
    if plausible.sum() >= 3:
        per_beat = per_beat[plausible]
    bpm = 60.0 / per_beat
    return {
        "median_bpm": float(np.median(bpm)),
        "std_bpm": float(np.std(bpm)),
        "min_bpm": float(bpm.min()),
        "max_bpm": float(bpm.max()),
        "n_beats": float(len(beat_times)),
    }
