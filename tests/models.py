"""Stand-in models that behave the way the real ones were measured to behave.

The three real models -- Beat This!, Mel-Roformer-Denoise and Apollo -- are
gigabytes of checkpoint and live only in the V3 image. None of them is needed
to test the thing under test here, which is the architecture around them: what
decides whether a step runs, how much of its output survives, where it is
allowed to act, and whether the finished result is kept.

So each is replaced by a function that reproduces its *measured misbehaviour*
from `docs/audio-restoration-plan.md`. That is the honest choice for an
ablation: a stand-in that behaved well would prove only that a well-behaved
model causes no harm, which was never in question. These are deliberately as
bad as the originals were measured to be, so that a step surviving the chain
unharmed is evidence about the chain.

  denoise   -> a downward expander reaching -15.75dB on the quietest frames,
               plus 3.4dB out of 5-10kHz.
  bandwidth -> an incoherent high band ~65dB down, swinging on the chunk stride.
  tempo     -> a phase-vocoder warp accumulating 375ms across a region.
"""

from __future__ import annotations

import numpy as np

from tests import corpus


def measured_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """The denoiser as measured: a downward expander that also dulls presence.

    Knee is placed against the material's own frame-level distribution rather
    than at a fixed -75dBFS, so the behaviour is reproduced on any fixture
    instead of only on one whose floor happens to sit there.
    """
    knee = float(np.percentile(corpus.frame_level_db(audio, sr), 85))
    expanded = corpus.damage_expander(audio, sr, knee_db=knee, max_attenuation_db=15.75)
    return corpus.damage_presence(expanded, sr, low_hz=5000, high_hz=10000, gain_db=-3.43)


def gentle_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """A denoiser that behaves: removes the stationary bed, keeps everything else.

    The control case. If the chain damages material even with this, the problem
    is the chain.
    """
    import librosa

    channels = [audio] if audio.ndim == 1 else [audio[:, i] for i in range(audio.shape[1])]
    out = []
    for channel in channels:
        spec = librosa.stft(channel.astype(np.float32), n_fft=2048, hop_length=512)
        magnitude = np.abs(spec)
        floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        # Spectral subtraction of the estimated floor, with a gain floor so no
        # bin is ever driven to silence.
        gain = np.maximum((magnitude - floor) / (magnitude + 1e-12), 0.25)
        out.append(
            librosa.istft(spec * gain, n_fft=2048, hop_length=512, length=len(channel))
        )
    result = out[0] if len(out) == 1 else np.stack(out, axis=1)
    return result.astype(np.float32)


def measured_bandwidth(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apollo as measured: high band with ~0.012 coherence, pulsing per chunk."""
    return corpus.damage_synthetic_hf(audio, sr, level_db=-65.0, period_s=9.0)


def honest_bandwidth(audio: np.ndarray, sr: int) -> np.ndarray:
    """A bandwidth extender that behaves: harmonics derived from the source.

    The high band is generated from the material below the cutoff, so it is
    coherent with it -- which is what a working extender produces and what the
    artifact metrics should reward.
    """
    from scipy.signal import butter, sosfilt

    sos = butter(6, 6000 / (sr / 2), btype="high", output="sos")
    band = sosfilt(sos, audio, axis=0)
    # A pointwise nonlinearity generates harmonics of what is already there.
    harmonics = np.sign(band) * band**2
    peak = float(np.abs(harmonics).max())
    if peak > 0:
        harmonics = harmonics / peak * float(np.abs(audio).max()) * 0.05
    return (audio + harmonics).astype(np.float32)


def measured_tempo(audio: np.ndarray, sr: int) -> np.ndarray:
    """The warp as measured: 375ms accumulated across a region, tail inherits it."""
    duration = audio.shape[0] / sr
    return corpus.damage_warp(
        audio, sr,
        start_s=min(15.0, duration * 0.2),
        end_s=min(90.0, duration * 0.7),
        drift_ms=-375.0,
    )


def identity(audio: np.ndarray, sr: int) -> np.ndarray:
    """A model that does nothing, for isolating the chain's own overhead."""
    return audio.copy()


MEASURED = {
    "tempo_model": measured_tempo,
    "denoise_model": measured_denoise,
    "bandwidth_model": measured_bandwidth,
}

WELL_BEHAVED = {
    "tempo_model": identity,
    "denoise_model": gentle_denoise,
    "bandwidth_model": honest_bandwidth,
}
