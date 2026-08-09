"""Step 3 -- bandwidth extension.

Regenerates the high frequencies lost above the export cutoff using Apollo,
which is trained on MP3 codec artifacts and evaluated on separated-stem data
(MUSDB18-HQ, MoisesDB). AudioSR was the obvious alternative and is not used: it
allocates ~7.2GB and does not fit the 8GB card this runs on, and its own README
warns that training on synthetic low-pass data leaves it weak on exactly the
MP3-shaped cutoffs these stems have.

Apollo ships no pip package, so its `look2hear` package is imported from a
clone. Its bundled `inference.py` is deliberately not used:

  * That script calls `from_pretrain("JusperLee/Apollo")`, but `from_pretrain`
    runs `torch.load()` on the string. It is a local checkpoint path, not a
    repository id, so the script fails outright with FileNotFoundError. The
    checkpoint is fetched from the Hub here and the real path passed in.
  * It hardcodes `.cuda()` in two places, with no CPU fallback.
  * Its loader does not resample. It feeds the file's own rate straight into a
    model built for 44.1kHz and writes the result at 44.1kHz, so 48kHz input
    comes back shifted in pitch and speed by 48000/44100 -- about 8.8%.

Working at the library level avoids all three: audio is resampled to 44.1kHz
going in and back to 48kHz coming out, and the device is chosen at runtime.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio_io import TARGET_SR, is_effectively_silent, resample
from .metrics import energy_above_hz, spectral_cliff_hz

APOLLO_SR = 44100
APOLLO_REPO_ID = "JusperLee/Apollo"
APOLLO_CHECKPOINT = "pytorch_model.bin"

# The architecture Apollo's own inference script instantiates.
APOLLO_MODEL_ARGS = {"sr": APOLLO_SR, "win": 20, "feature_dim": 256, "layer": 6}

# Apollo is a sequence model; a whole stem at once exceeds VRAM, so it runs in
# chunks with an overlap that is crossfaded away.
CHUNK_S = 10.0
OVERLAP_S = 0.5


class ApolloUnavailable(RuntimeError):
    pass


@dataclass
class BandwidthResult:
    audio: np.ndarray
    sr: int
    cliff_before_hz: float | None
    cliff_after_hz: float | None
    energy_before_pct: float
    energy_after_pct: float
    skipped: bool = False
    reason: str = ""


def check_apollo(repo_dir: str | Path) -> Path:
    """Validate an Apollo checkout and return its root."""
    root = Path(repo_dir).resolve()
    if not (root / "look2hear" / "models").is_dir():
        raise ApolloUnavailable(
            f"Apollo checkout not found at {root}. Clone it with:\n"
            "  git clone https://github.com/JusperLee/Apollo vendor/Apollo"
        )
    return root


@lru_cache(maxsize=1)
def load_model(repo_dir: str, device: str = "cuda"):
    """Import Apollo from its checkout and load the Hub checkpoint onto `device`."""
    root = check_apollo(repo_dir)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from huggingface_hub import hf_hub_download

    try:
        import look2hear.models
    except ImportError as error:  # pragma: no cover - depends on the checkout
        raise ApolloUnavailable(f"Could not import look2hear from {root}: {error}") from error

    checkpoint = hf_hub_download(repo_id=APOLLO_REPO_ID, filename=APOLLO_CHECKPOINT)
    model = look2hear.models.BaseModel.from_pretrain(checkpoint, **APOLLO_MODEL_ARGS)
    return model.to(device).eval(), torch


def _process_chunks(model, torch, audio: np.ndarray, device: str) -> np.ndarray:
    """Run Apollo over overlapping chunks and crossfade the seams."""
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    signal = audio if audio.ndim == 2 else audio[:, None]

    chunk = int(CHUNK_S * APOLLO_SR)
    overlap = int(OVERLAP_S * APOLLO_SR)
    step = chunk - overlap

    output = np.zeros((signal.shape[0], channels), dtype=np.float32)
    weights = np.zeros((signal.shape[0], 1), dtype=np.float32)
    window = np.ones((chunk, 1), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, None]
    window[:overlap] = ramp
    window[-overlap:] = ramp[::-1]

    for start in range(0, signal.shape[0], step):
        piece = signal[start:start + chunk]
        if piece.shape[0] < overlap:
            break
        tensor = torch.from_numpy(piece.T.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            processed = model(tensor)
        result = processed.squeeze(0).cpu().numpy().T[: piece.shape[0]]

        taper = window[: piece.shape[0]]
        output[start:start + result.shape[0]] += result * taper
        weights[start:start + result.shape[0]] += taper

    output /= np.maximum(weights, 1e-8)
    return output[:, 0] if audio.ndim == 1 else output


def extend_stem(
    audio: np.ndarray,
    sr: int,
    repo_dir: str | Path,
    work_dir: str | Path | None = None,
    output_sr: int = TARGET_SR,
    device: str | None = None,
) -> BandwidthResult:
    """Run Apollo on one stem and return it at `output_sr`."""
    cliff_before = spectral_cliff_hz(audio, sr)
    energy_before = energy_above_hz(audio, sr)

    if is_effectively_silent(audio):
        return BandwidthResult(
            audio, sr, cliff_before, cliff_before, energy_before, energy_before,
            skipped=True, reason="stem is effectively silent",
        )

    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    model, torch = load_model(str(Path(repo_dir).resolve()), device)

    # Apollo is built for 44.1kHz and does no resampling of its own.
    feed = resample(audio, sr, APOLLO_SR) if sr != APOLLO_SR else audio
    extended = _process_chunks(model, torch, feed.astype(np.float32), device)

    if APOLLO_SR != output_sr:
        extended = resample(extended, APOLLO_SR, output_sr)

    return BandwidthResult(
        audio=extended,
        sr=output_sr,
        cliff_before_hz=cliff_before,
        cliff_after_hz=spectral_cliff_hz(extended, output_sr),
        energy_before_pct=energy_before,
        energy_after_pct=energy_above_hz(extended, output_sr),
    )
