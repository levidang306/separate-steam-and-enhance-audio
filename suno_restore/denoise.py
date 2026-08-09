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

from .audio_io import is_effectively_silent, load_audio, save_audio
from .metrics import residual_level_db, rms_db

MODEL_FILENAME = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"


@dataclass
class DenoiseResult:
    audio: np.ndarray
    sr: int
    residual_db: float
    skipped: bool = False
    reason: str = ""


def load_separator(output_dir: str | Path, model_filename: str = MODEL_FILENAME):
    """Build an audio-separator instance with the denoise model loaded.

    Callers should cache this: the checkpoint is ~900MB and takes seconds to
    load even from disk.
    """
    from audio_separator.separator import Separator

    separator = Separator(output_dir=str(output_dir))
    separator.load_model(model_filename=model_filename)
    return separator


def denoise_stem(
    audio: np.ndarray, sr: int, separator, work_dir: str | Path | None = None
) -> DenoiseResult:
    """Denoise one stem, returning the cleaned signal at the same sample rate."""
    if is_effectively_silent(audio):
        return DenoiseResult(audio, sr, residual_db=-np.inf, skipped=True,
                             reason="stem is effectively silent")

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
            return DenoiseResult(audio, sr, residual_db=-np.inf, skipped=True,
                                 reason=f"model produced no 'dry' output: {outputs}")

        cleaned, cleaned_sr = load_audio(dry, target_sr=sr, mono=audio.ndim == 1)

    return DenoiseResult(
        audio=cleaned,
        sr=cleaned_sr,
        residual_db=residual_level_db(audio, cleaned),
    )


def describe_residual(residual_db: float) -> str:
    """Plain reading of how much the denoiser actually changed."""
    if residual_db == -np.inf:
        return "nothing removed"
    if residual_db < -40:
        return "removed content is inaudible (>40dB down) -- this stem was already clean"
    if residual_db < -20:
        return "removed a quiet noise layer"
    return "removed a substantial amount -- check the result for lost musical content"


__all__ = ["DenoiseResult", "denoise_stem", "describe_residual", "load_separator", "rms_db"]
