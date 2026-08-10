"""Combine a restoration model's output with the signal it was given.

No stage writes its result straight to the output. Every stage produces a
candidate, and this module decides how much of that candidate survives -- by
frequency, by frame, and scaled by how confident the gate was that the stage
should run at all.

That indirection is the whole point. The old chain replaced the signal with the
model's output at full strength, which meant every judgement the model made was
final: what it removed was gone, and what it invented was kept. Three specific
harms followed, and each has a corresponding control here.

  * The denoiser behaved as a downward expander, pulling the quietest frames
    down 15.75dB and taking room tone, reverb tails and note decay with them.
    `RESTORE_MAX_ATTENUATION_DB` caps how far any frame can be pulled down, no
    matter what the model returned.

  * 3.4dB went missing between 5 and 8kHz -- pick attack and string
    articulation, the difference between an articulate guitar and a veiled one.
    The frequency weights let the denoiser act where hiss lives and hold it back
    where music does.

  * Bandwidth extension rewrote material below the cutoff as well as above it.
    Here it can only add above the cliff; everything below is the dry signal,
    untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import channel_count
from .config import settings as default_settings

N_FFT = 2048
HOP = 512
EPS = 1e-12


@dataclass
class BlendResult:
    """What the blend actually did, as opposed to what it was asked to do."""

    audio: np.ndarray
    requested_wet: float
    effective_wet: float
    max_attenuation_db: float
    limited_frames: int
    total_frames: int

    def to_dict(self) -> dict:
        return {
            "requested_wet": self.requested_wet,
            "effective_wet": self.effective_wet,
            "max_attenuation_db": self.max_attenuation_db,
            "limited_frames": self.limited_frames,
            "total_frames": self.total_frames,
        }


def _align(dry: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to a common length and channel layout.

    A model may return a sample or two more or less than it was given. Anything
    larger than that is a fault the caller should have caught, not something to
    paper over here.
    """
    if channel_count(dry) != channel_count(wet):
        raise ValueError(
            f"cannot blend {channel_count(dry)} channels with {channel_count(wet)}"
        )
    n = min(dry.shape[0], wet.shape[0])
    return dry[:n], wet[:n]


def band_weights(sr: int, n_fft: int, settings=None) -> np.ndarray:
    """Per-bin weight for how freely denoising may act at each frequency.

    The four configured values are treated as the weight at the geometric
    centre of each band and interpolated in log frequency between them. A
    piecewise-constant profile with hard edges would be a brick-wall filter
    applied to the difference signal, and would ring; a smooth curve does the
    same job without adding an artifact of its own.
    """
    settings = settings or default_settings
    import librosa

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    centres = np.array([1000.0, np.sqrt(2000 * 5000), np.sqrt(5000 * 10000), 14142.0])
    values = np.array([
        settings.denoise_weight_below_2k,
        settings.denoise_weight_2k_5k,
        settings.denoise_weight_5k_10k,
        settings.denoise_weight_above_10k,
    ])
    # Flat outside the outermost centres rather than extrapolated: extrapolating
    # a log-linear ramp past 14kHz would push the weight above 1.0.
    safe = np.maximum(freqs, 1.0)
    return np.interp(np.log(safe), np.log(centres), values).astype(np.float32)


def _stft(mono: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    import librosa

    return librosa.stft(mono.astype(np.float32), n_fft=n_fft, hop_length=hop)


def _istft(spec: np.ndarray, length: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    import librosa

    return librosa.istft(spec, n_fft=n_fft, hop_length=hop, length=length)


def _channels(audio: np.ndarray) -> list[np.ndarray]:
    return [audio] if audio.ndim == 1 else [audio[:, i] for i in range(audio.shape[1])]


def _restack(parts: list[np.ndarray], like: np.ndarray) -> np.ndarray:
    if like.ndim == 1:
        return parts[0].astype(np.float32)
    return np.stack(parts, axis=1).astype(np.float32)


def limit_attenuation(
    dry_spec: np.ndarray, blended_spec: np.ndarray, max_db: float
) -> tuple[np.ndarray, float, int]:
    """Stop any frame being pulled more than `max_db` below the dry signal.

    This is the hard backstop, and it is deliberately per-frame rather than
    per-bin. The damage it exists to prevent is a whole quiet frame being
    expanded downward -- a decaying note, a reverb tail, the room between
    phrases -- and that shows up as a broadband level drop, not a spectral one.
    Per-bin limiting would additionally stop the denoiser removing a narrow
    noise component it is supposed to remove.
    """
    if max_db <= 0 or blended_spec.size == 0:
        return blended_spec, 0.0, 0

    dry_level = np.sqrt(np.mean(np.abs(dry_spec) ** 2, axis=0))
    out_level = np.sqrt(np.mean(np.abs(blended_spec) ** 2, axis=0))
    floor = dry_level * (10 ** (-max_db / 20))

    too_quiet = (out_level < floor) & (out_level > EPS)
    if not too_quiet.any():
        deepest = float(
            np.max(20 * np.log10((dry_level + EPS) / (out_level + EPS)))
        ) if out_level.size else 0.0
        return blended_spec, max(0.0, deepest), 0

    scale = np.ones_like(out_level)
    scale[too_quiet] = floor[too_quiet] / out_level[too_quiet]
    deepest = float(np.max(20 * np.log10((dry_level + EPS) / (out_level + EPS))))
    return blended_spec * scale[None, :], max(0.0, deepest), int(np.count_nonzero(too_quiet))


def blend_spectral(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    amount: float,
    *,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    max_attenuation_db: float | None = None,
    settings=None,
) -> BlendResult:
    """Blend `wet` into `dry` per bin and per frame.

    `amount` is the wet fraction before weighting; `weights` scales it by
    frequency; `mask` scales it by time and frequency (the damage mask, so that
    healthy regions are left alone). The blend is complex, not magnitude-only,
    because the two signals are sample-aligned and a complex blend is then just
    a time-varying filter -- magnitude-only blending would recombine the dry
    phase with a foreign magnitude and smear transients.
    """
    settings = settings or default_settings
    dry, wet = _align(dry, wet)
    limit = settings.max_attenuation_db if max_attenuation_db is None else max_attenuation_db
    amount = float(np.clip(amount, 0.0, 1.0))

    if amount <= 0:
        return BlendResult(dry.copy(), amount, 0.0, 0.0, 0, 0)

    length = dry.shape[0]
    out_parts, effective, deepest, limited, frames = [], [], 0.0, 0, 0

    for dry_channel, wet_channel in zip(_channels(dry), _channels(wet)):
        dry_spec = _stft(dry_channel)
        wet_spec = _stft(wet_channel)
        bins = min(dry_spec.shape[1], wet_spec.shape[1])
        dry_spec, wet_spec = dry_spec[:, :bins], wet_spec[:, :bins]

        gain = np.full(dry_spec.shape, amount, dtype=np.float32)
        if weights is not None:
            gain *= weights[: dry_spec.shape[0], None]
        if mask is not None:
            gain *= _fit_mask(mask, dry_spec.shape)

        blended = dry_spec + gain * (wet_spec - dry_spec)
        blended, channel_deepest, channel_limited = limit_attenuation(dry_spec, blended, limit)

        out_parts.append(_istft(blended, length))
        effective.append(float(np.mean(gain)))
        deepest = max(deepest, channel_deepest)
        limited += channel_limited
        frames += bins

    return BlendResult(
        audio=_restack(out_parts, dry),
        requested_wet=amount,
        effective_wet=float(np.mean(effective)) if effective else 0.0,
        max_attenuation_db=deepest,
        limited_frames=limited,
        total_frames=frames,
    )


def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a damage mask onto the STFT grid.

    The mask is generated at its own resolution -- 20ms frames -- so it does not
    have to agree with whatever FFT size a stage happens to use.
    """
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask[None, :]
    bins, frames = shape
    rows = np.clip(
        np.round(np.linspace(0, mask.shape[0] - 1, bins)).astype(int), 0, mask.shape[0] - 1
    )
    columns = np.clip(
        np.round(np.linspace(0, mask.shape[1] - 1, frames)).astype(int), 0, mask.shape[1] - 1
    )
    return mask[np.ix_(rows, columns)]


def blend_denoise(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    *,
    confidence: float = 1.0,
    mask: np.ndarray | None = None,
    settings=None,
) -> BlendResult:
    """Apply a denoise result, weighted by frequency, damage and confidence.

    The wet amount is the configured maximum scaled by the gate's confidence, so
    a stem with a marginal amount of hiss gets a marginal amount of processing.
    That is what "apply at reduced strength" means in practice.
    """
    settings = settings or default_settings
    return blend_spectral(
        dry,
        wet,
        sr,
        amount=settings.denoise_wet * float(np.clip(confidence, 0.0, 1.0)),
        weights=band_weights(sr, N_FFT, settings),
        mask=mask,
        settings=settings,
    )


def high_band_weights(sr: int, n_fft: int, cliff_hz: float, taper_hz: float = 1000.0) -> np.ndarray:
    """1 above `cliff_hz`, 0 below, with a smooth transition.

    Bandwidth extension has no business altering material below the cutoff --
    that part was never missing. Restricting it to the band above the cliff
    means the step can add, but cannot rewrite.
    """
    import librosa

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    taper = max(taper_hz, 1.0)
    ramp = np.clip((freqs - (cliff_hz - taper)) / taper, 0.0, 1.0)
    return (np.sin(ramp * np.pi / 2) ** 2).astype(np.float32)


def blend_bandwidth(
    dry: np.ndarray,
    wet: np.ndarray,
    sr: int,
    *,
    cliff_hz: float,
    confidence: float = 1.0,
    mask: np.ndarray | None = None,
    settings=None,
) -> BlendResult:
    """Apply a bandwidth-extension result above the cliff only."""
    settings = settings or default_settings
    return blend_spectral(
        dry,
        wet,
        sr,
        amount=settings.bandwidth_wet * float(np.clip(confidence, 0.0, 1.0)),
        weights=high_band_weights(sr, N_FFT, cliff_hz),
        mask=mask,
        settings=settings,
    )


__all__ = [
    "BlendResult",
    "band_weights",
    "blend_bandwidth",
    "blend_denoise",
    "blend_spectral",
    "high_band_weights",
    "limit_attenuation",
]
