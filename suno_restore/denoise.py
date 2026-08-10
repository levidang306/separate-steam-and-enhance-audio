"""Step 2 -- denoise.

Removes hiss, crackle and "AI shimmer" using Mel-Roformer-Denoise, a
music-source-separation model that emits a cleaned "dry" track and the noise it
pulled out. Speech enhancers such as DeepFilterNet are a poor fit here despite
scoring well on noise suppression: DeepFilterNet describes itself as a "Speech
Enhancement Framework", and on bass, drums or piano it suppresses musical
content it cannot recognise as speech.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import (
    channel_count,
    from_mid_side,
    is_effectively_silent,
    load_audio,
    save_audio,
    to_mid_side,
)
from .config import settings as default_settings
from .metrics import residual_level_db, rms_db

MODEL_FILENAME = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"

# Set on a separator once its output has been seen to drop channels, so the
# discovery is made once per process instead of once per stem.
_MONO_ONLY_FLAG = "_suno_restore_mono_only"


@dataclass
class DenoiseResult:
    audio: np.ndarray
    sr: int
    residual_db: float
    skipped: bool = False
    reason: str = ""
    stereo_strategy: str = "joint"
    input_channels: int = 0
    output_channels: int = 0


def load_separator(output_dir: str | Path, model_filename: str = MODEL_FILENAME):
    """Build an audio-separator instance with the denoise model loaded.

    Callers should cache this: the checkpoint is ~900MB and takes seconds to
    load even from disk.
    """
    from audio_separator.separator import Separator

    separator = Separator(output_dir=str(output_dir))
    separator.load_model(model_filename=model_filename)
    return separator


class DenoiseFailed(RuntimeError):
    pass


def _run_model(
    audio: np.ndarray, sr: int, separator, work_dir: str | Path | None, want_channels: int
) -> np.ndarray:
    """One pass of the separator, decoded back at `want_channels`.

    `want_channels` is not a request to the model -- it is how the *result* is
    decoded. The model decides what it writes; forcing the decode means a mono
    file can never quietly redefine a stereo stem as mono. Whether the model
    actually kept the channels is reported separately, by comparing what it
    wrote against what went in.
    """
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        source = save_audio(tmp_path / "input.wav", audio, sr)

        # audio-separator keeps the output directory in two places and its own
        # internal redirect sets both; setting only the outer one leaves the
        # model writing somewhere else, and the files never appear here.
        previous_dir = separator.output_dir
        model = getattr(separator, "model_instance", None)
        previous_model_dir = getattr(model, "output_dir", None) if model else None
        separator.output_dir = str(tmp_path)
        if model is not None:
            model.output_dir = str(tmp_path)
        try:
            outputs = separator.separate(str(source))
        finally:
            separator.output_dir = previous_dir
            if model is not None and previous_model_dir is not None:
                model.output_dir = previous_model_dir

        # separate() may return bare filenames or full paths depending on version.
        produced = [Path(name) if Path(name).is_absolute() else tmp_path / name for name in outputs]
        dry = next((p for p in produced if "dry" in p.name.lower()), None)
        if dry is None:
            raise DenoiseFailed(f"model produced no 'dry' output: {outputs}")

        from .audio_io import probe

        _, written_channels = probe(dry)
        cleaned, _ = load_audio(dry, target_sr=sr, channels=want_channels)

    if written_channels < want_channels:
        setattr(separator, _MONO_ONLY_FLAG, True)
    return cleaned


def denoise_stem(
    audio: np.ndarray,
    sr: int,
    separator,
    work_dir: str | Path | None = None,
    settings=None,
) -> DenoiseResult:
    """Denoise one stem, preserving its channel count and sample rate.

    The stereo image is the thing most easily lost here. The denoise model may
    write a mono file, and the previous version adopted whatever channel count
    the file happened to have -- which turned every stereo stem into a mono one
    and discarded the side channel entirely. Now the input's channel count is
    the contract, and a mono-only model is handled by running it twice, on mid
    and side, instead of by giving up the image.
    """
    settings = settings or default_settings
    if is_effectively_silent(audio):
        return DenoiseResult(audio, sr, residual_db=-np.inf, skipped=True,
                             reason="stem is effectively silent",
                             stereo_strategy="skipped",
                             input_channels=channel_count(audio),
                             output_channels=channel_count(audio))

    channels = channel_count(audio)
    mode = settings.stereo_mode
    known_mono_only = bool(getattr(separator, _MONO_ONLY_FLAG, False))
    use_mid_side = channels == 2 and (mode == "mid_side" or (mode == "auto" and known_mono_only))

    try:
        if use_mid_side:
            cleaned, strategy = _denoise_mid_side(audio, sr, separator, work_dir)
        else:
            cleaned = _run_model(audio, sr, separator, work_dir, want_channels=channels)
            strategy = "joint"
            # "auto" only learns a model is mono-only by watching it write one.
            # Having just learned it, redo this stem properly rather than ship
            # a stem whose side channel was thrown away.
            if mode == "auto" and channels == 2 and getattr(separator, _MONO_ONLY_FLAG, False):
                cleaned, strategy = _denoise_mid_side(audio, sr, separator, work_dir)
    except DenoiseFailed as error:
        return DenoiseResult(audio, sr, residual_db=-np.inf, skipped=True, reason=str(error),
                             stereo_strategy="failed", input_channels=channels,
                             output_channels=channels)

    if settings.enforce_channel_count and channel_count(cleaned) != channels:
        raise DenoiseFailed(
            f"denoise changed the channel count from {channels} to {channel_count(cleaned)}"
        )

    return DenoiseResult(
        audio=cleaned,
        sr=sr,
        residual_db=residual_level_db(audio, cleaned),
        stereo_strategy=strategy,
        input_channels=channels,
        output_channels=channel_count(cleaned),
    )


def _denoise_mid_side(
    audio: np.ndarray, sr: int, separator, work_dir: str | Path | None
) -> tuple[np.ndarray, str]:
    """Denoise mid and side independently, then rebuild the stereo pair."""
    mid, side = to_mid_side(audio)
    clean_mid = _run_model(mid, sr, separator, work_dir, want_channels=1)
    clean_side = _run_model(side, sr, separator, work_dir, want_channels=1)
    return from_mid_side(clean_mid, clean_side), "mid_side"


def describe_residual(residual_db: float) -> str:
    """Plain reading of how much the denoiser actually changed."""
    if residual_db == -np.inf:
        return "nothing removed"
    if residual_db < -40:
        return "removed content is inaudible (>40dB down) -- this stem was already clean"
    if residual_db < -20:
        return "removed a quiet noise layer"
    return "removed a substantial amount -- check the result for lost musical content"


__all__ = [
    "DenoiseFailed",
    "DenoiseResult",
    "denoise_stem",
    "describe_residual",
    "load_separator",
    "rms_db",
]
