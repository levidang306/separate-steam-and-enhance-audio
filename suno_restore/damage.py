"""Locate the damage, so restoration can be aimed instead of sprayed.

A gate answers "should this step run on this stem". That is still a whole-stem
answer, and most damage is not whole-stem. Hiss is loudest in the gaps between
notes and inaudible under a loud chord; a codec cutoff is a property of one
region of the spectrum and none of the rest. Applying a step evenly across a
stem therefore spends most of its effect on material that was never damaged --
which is where the cost lands, because processing healthy audio can only take
something away.

So each step's wet contribution is multiplied by a mask:

    wet = model_output * confidence * damage_mask

The mask is 1 where the evidence says the signal is damaged and 0 where it says
the signal is fine, at 20ms resolution, feathered by +/-100ms so the treatment
fades in and out rather than switching. Without the feather the mask itself
becomes the artifact: a hard on/off gate at 20ms is audible as pumping, and it
would be a self-inflicted version of exactly the problem this file exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import to_mono
from .config import settings as default_settings

N_FFT = 2048
EPS = 1e-12

# Where a cell's signal-to-noise sits between "this is noise" and "this is
# music". At 0dB above the estimated floor a cell is noise and fully in scope;
# by 12dB above it the noise is masked and there is nothing worth removing.
SNR_FULL_DAMAGE_DB = 0.0
SNR_NO_DAMAGE_DB = 12.0

# Smoothing across frequency, in bins. Narrow-band on/off decisions ring; a few
# bins of blur costs nothing and removes that.
FREQUENCY_SMOOTHING_BINS = 5


@dataclass
class DamageMask:
    """A per-frame, per-band statement of where restoration is warranted."""

    mask: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    frame_ms: float
    feather_ms: float
    kind: str

    @property
    def coverage(self) -> float:
        """Fraction of the time-frequency plane the mask lets through."""
        return float(np.mean(self.mask)) if self.mask.size else 0.0

    def to_dict(self) -> dict:
        """Summary only. The array itself is megabytes and belongs in memory."""
        return {
            "kind": self.kind,
            "coverage": self.coverage,
            "frame_ms": self.frame_ms,
            "feather_ms": self.feather_ms,
            "bands": int(self.mask.shape[0]) if self.mask.ndim == 2 else 0,
            "frames": int(self.mask.shape[1]) if self.mask.ndim == 2 else 0,
            "max": float(self.mask.max()) if self.mask.size else 0.0,
            "min": float(self.mask.min()) if self.mask.size else 0.0,
        }


def _smooth(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Hann-weighted moving average along one axis."""
    if width < 3:
        return values
    window = np.hanning(width)
    window /= window.sum()
    return np.apply_along_axis(
        lambda row: np.convolve(row, window, mode="same"), axis, values
    )


def feather_mask(mask: np.ndarray, frame_ms: float, feather_ms: float) -> np.ndarray:
    """Soften a mask in time so treatment fades in and out.

    Feathering is applied *after* dilation. Softening alone would pull the mask
    down at the edges of a damaged region, so the first and last few frames of
    real damage would go under-treated; widening first and then softening keeps
    the interior at full strength and puts the ramp outside it.
    """
    if mask.size == 0 or feather_ms <= 0 or frame_ms <= 0:
        return mask

    width = int(round(feather_ms / frame_ms))
    if width < 1:
        return mask

    from scipy.ndimage import maximum_filter1d

    dilated = maximum_filter1d(mask, size=2 * width + 1, axis=-1, mode="nearest")
    smoothed = _smooth(dilated, 2 * width + 1, axis=-1)
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def _spectrogram(audio: np.ndarray, sr: int, frame_ms: float) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray]:
    import librosa

    hop = max(1, int(round(frame_ms / 1000.0 * sr)))
    mono = to_mono(audio).astype(np.float32)
    spec = np.abs(librosa.stft(mono, n_fft=N_FFT, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = np.arange(spec.shape[1]) * hop / sr
    return spec, freqs, times


def noise_damage(audio: np.ndarray, sr: int, *, settings=None) -> DamageMask:
    """Where noise is a significant part of what is present.

    The noise floor is estimated per frequency bin as a low percentile across
    time -- what is present at that frequency even at its quietest. A cell close
    to that floor is mostly noise and is in scope; a cell far above it is music
    with the noise masked underneath, where removing the noise achieves nothing
    audible and risks removing music instead.

    This is the same shape of decision a downward expander makes, and it is
    worth being explicit about why that is acceptable here: the expander applied
    unlimited gain reduction as its entire output, whereas this only decides
    *where* a wet contribution that is already partial, frequency-weighted and
    capped at `RESTORE_MAX_ATTENUATION_DB` gets to act.
    """
    settings = settings or default_settings
    frame_ms = settings.damage_frame_ms
    spec, freqs, times = _spectrogram(audio, sr, frame_ms)

    if spec.shape[1] < 4:
        return DamageMask(np.zeros_like(spec), freqs, times, frame_ms,
                          settings.damage_feather_ms, "noise")

    profile = np.percentile(spec, 10, axis=1, keepdims=True)
    snr_db = 20 * np.log10((spec + EPS) / (profile + EPS))

    span = max(SNR_NO_DAMAGE_DB - SNR_FULL_DAMAGE_DB, EPS)
    mask = np.clip((SNR_NO_DAMAGE_DB - snr_db) / span, 0.0, 1.0)
    mask = _smooth(mask, FREQUENCY_SMOOTHING_BINS, axis=0)
    mask = feather_mask(mask, frame_ms, settings.damage_feather_ms)

    return DamageMask(mask.astype(np.float32), freqs, times, frame_ms,
                      settings.damage_feather_ms, "noise")


def bandwidth_damage(
    audio: np.ndarray, sr: int, cliff_hz: float, *, settings=None, taper_hz: float = 1000.0
) -> DamageMask:
    """The band above a codec cutoff, wherever there is material to extend from.

    Damage here is a property of frequency rather than time -- the cutoff is
    there for the whole file. But not of every moment: a frame with nothing
    below the cliff has nothing for a model to extrapolate from, and inventing a
    high band over silence is how a stem acquires content that does not belong
    to it.
    """
    settings = settings or default_settings
    frame_ms = settings.damage_frame_ms
    spec, freqs, times = _spectrogram(audio, sr, frame_ms)

    above = np.clip((freqs - (cliff_hz - taper_hz)) / max(taper_hz, EPS), 0.0, 1.0)
    above = np.sin(above * np.pi / 2) ** 2

    shoulder = (freqs >= cliff_hz * 0.5) & (freqs < cliff_hz)
    if shoulder.any():
        feed = np.sqrt(np.mean(spec[shoulder] ** 2, axis=0))
        reference = np.percentile(feed, 90) if feed.size else 0.0
        # Full strength wherever the octave below the cliff is within 20dB of
        # its own loudest; nothing where that octave is empty.
        present = np.clip(
            (20 * np.log10((feed + EPS) / (reference + EPS)) + 20.0) / 20.0, 0.0, 1.0
        )
    else:
        present = np.ones(spec.shape[1])

    mask = above[:, None] * present[None, :]
    mask = feather_mask(mask, frame_ms, settings.damage_feather_ms)

    return DamageMask(mask.astype(np.float32), freqs, times, frame_ms,
                      settings.damage_feather_ms, "bandwidth")


def combine(*masks: DamageMask) -> np.ndarray:
    """Element-wise product of several masks, on the finest grid supplied."""
    arrays = [m.mask for m in masks if m.mask.size]
    if not arrays:
        return np.ones((1, 1), dtype=np.float32)
    bands = max(a.shape[0] for a in arrays)
    frames = max(a.shape[1] for a in arrays)

    out = np.ones((bands, frames), dtype=np.float32)
    for array in arrays:
        rows = np.clip(
            np.round(np.linspace(0, array.shape[0] - 1, bands)).astype(int),
            0, array.shape[0] - 1,
        )
        columns = np.clip(
            np.round(np.linspace(0, array.shape[1] - 1, frames)).astype(int),
            0, array.shape[1] - 1,
        )
        out *= array[np.ix_(rows, columns)]
    return out


__all__ = ["DamageMask", "bandwidth_damage", "combine", "feather_mask", "noise_damage"]
