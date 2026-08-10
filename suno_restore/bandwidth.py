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
from typing import Callable

import numpy as np

from .audio_io import channel_count, is_effectively_silent, resample
from .config import resolve_output_sample_rate
from .config import settings as default_settings
from .metrics import energy_above_hz, spectral_cliff_hz

APOLLO_SR = 44100
APOLLO_REPO_ID = "JusperLee/Apollo"
APOLLO_CHECKPOINT = "pytorch_model.bin"

# The architecture Apollo's own inference script instantiates.
APOLLO_MODEL_ARGS = {"sr": APOLLO_SR, "win": 20, "feature_dim": 256, "layer": 6}

# Apollo is a sequence model; a whole stem at once exceeds VRAM, so it runs in
# chunks with an overlap that is crossfaded away. The sizes live in Settings
# (`RESTORE_BANDWIDTH_CHUNK_S` / `_OVERLAP_S`); these are the fallbacks.
CHUNK_S = 10.0
OVERLAP_S = 2.0


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


def chunk_window(
    length: int, overlap: int, fade_in: bool = True, fade_out: bool = True
) -> np.ndarray:
    """Raised-sine taper for one chunk, as a column vector.

    The ramp is sin^2, which is complementary -- sin^2 + cos^2 = 1 -- so
    overlapping tapers form an exact partition of unity. That matters: the low
    band is essentially identical in two neighbouring chunks, and any taper
    whose weights did *not* sum to one would boost or cut it at every seam.

    What sin^2 buys over a linear ramp is smoothness. A linear ramp has a slope
    discontinuity at each end of the fade; sin^2 meets 0 and 1 with zero slope,
    so the seam introduces no kink for the spectrum to splatter around.

    Neither shape fixes the other half of the problem. Where two chunks
    *disagree* -- which is what independently synthesised high bands do -- a
    weighted average of uncorrelated content loses power in the middle of the
    fade, at worst 3dB, and that is what made the generated high band swing once
    per chunk.

    A longer fade was expected to spread that out. Measured, it only helps when
    the chunks largely agree, which is the realistic case for a model deriving
    its high band from a shared low band; where they disagree completely, a
    longer fade is *worse*, because more of the timeline then sits inside an
    average of unrelated content. Both effects are under ~1dB. The default is
    2.0s because the agreeing case is the one that occurs when bandwidth
    extension is correctly gated on -- and when it is not, the real answer is
    for the gate to decline the step, not for the crossfade to hide it.
    See `tests/test_signal_integrity.py` for both measurements.

    `fade_in` / `fade_out` are switched off at the very start and very end of a
    signal. There is no neighbouring chunk to fade against there, so tapering
    would just attenuate the first and last seconds -- and, since the overlap-add
    normalises by the accumulated weight, a final chunk that tapered to zero
    would end up dividing by nothing at all.
    """
    window = np.ones((length, 1), dtype=np.float32)
    fade = min(overlap, length // 2)
    if fade <= 0:
        return window
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)) ** 2
    if fade_in:
        window[:fade, 0] = ramp
    if fade_out:
        window[length - fade:, 0] = ramp[::-1]
    return window


def process_chunked(
    audio: np.ndarray,
    process: Callable[[np.ndarray], np.ndarray],
    chunk: int,
    overlap: int,
) -> np.ndarray:
    """Apply `process` over overlapping chunks and crossfade the seams.

    Split out from the Apollo call so the overlap-add can be tested against a
    stand-in model. The seam behaviour is a property of this function, not of
    the network, and it is the part that produced a measurable artifact.
    """
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    signal = audio if audio.ndim == 2 else audio[:, None]
    length = signal.shape[0]
    overlap = max(0, min(overlap, chunk // 2))
    step = max(1, chunk - overlap)

    # Spans are decided before anything is processed, so the fade flags can be
    # set from which chunk actually neighbours which. Deciding them inside the
    # loop meant asking "is this the last one?" from a position that could not
    # yet see whether a further chunk would be taken, and a taper that fades out
    # with nothing to fade into is a hole in the coverage.
    spans = []
    for start in range(0, length, step):
        stop = min(start + chunk, length)
        if stop <= start:
            break
        spans.append((start, stop))
        if stop >= length:
            break
    # A final span shorter than the overlap is entirely covered by its
    # predecessor already; processing it would only add a redundant seam.
    while len(spans) > 1 and spans[-1][1] - spans[-1][0] <= overlap:
        spans.pop()
    if spans:
        spans[-1] = (spans[-1][0], length)

    output = np.zeros((length, channels), dtype=np.float32)
    weights = np.zeros((length, 1), dtype=np.float32)

    for index, (start, stop) in enumerate(spans):
        piece = signal[start:stop]
        result = np.asarray(process(piece))[: stop - start]
        if result.ndim == 1:
            result = result[:, None]
        if result.shape[0] < piece.shape[0]:
            # A model that returns short would otherwise leave the tail of its
            # span carried by nothing, and the weight normalisation below would
            # then divide near-silence by near-zero. Carry the source there
            # instead: unextended is a far better failure than amplified noise.
            result = np.concatenate([result, piece[result.shape[0]:]], axis=0)

        taper = chunk_window(
            result.shape[0],
            overlap,
            fade_in=index > 0,
            fade_out=index < len(spans) - 1,
        )
        output[start:start + result.shape[0]] += result * taper
        weights[start:start + result.shape[0]] += taper

    # The tapers are complementary, so the weights are 1 everywhere a chunk
    # reached. Anywhere they are not, the source is the honest answer.
    covered = weights[:, 0] > 1e-6
    output[covered] /= weights[covered]
    output[~covered] = signal[~covered]
    return output[:, 0] if audio.ndim == 1 else output


def _process_chunks(model, torch, audio: np.ndarray, device: str, settings=None) -> np.ndarray:
    """Run Apollo over overlapping chunks and crossfade the seams."""
    settings = settings or default_settings

    def run(piece: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(piece.T.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            processed = model(tensor)
        return processed.squeeze(0).cpu().numpy().T

    return process_chunked(
        audio,
        run,
        chunk=int(settings.bandwidth_chunk_s * APOLLO_SR),
        overlap=int(settings.bandwidth_overlap_s * APOLLO_SR),
    )


def extend_stem(
    audio: np.ndarray,
    sr: int,
    repo_dir: str | Path,
    work_dir: str | Path | None = None,
    output_sr: int | None = None,
    device: str | None = None,
    settings=None,
) -> BandwidthResult:
    """Run Apollo on one stem and return it at the resolved output rate.

    `output_sr=None` means follow the source. The previous default pinned every
    result to 48kHz, so a 44.1kHz source came back resampled twice -- once into
    Apollo's rate and once out of it -- for no reason anyone had asked for.
    """
    settings = settings or default_settings
    cliff_before = spectral_cliff_hz(audio, sr)
    energy_before = energy_above_hz(audio, sr)
    target_sr = resolve_output_sample_rate(sr, output_sr)

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
    extended = _process_chunks(model, torch, feed.astype(np.float32), device, settings)

    if APOLLO_SR != target_sr:
        extended = resample(extended, APOLLO_SR, target_sr)

    if channel_count(extended) != channel_count(audio):
        raise RuntimeError(
            f"bandwidth extension changed the channel count from "
            f"{channel_count(audio)} to {channel_count(extended)}"
        )

    return BandwidthResult(
        audio=extended,
        sr=target_sr,
        cliff_before_hz=cliff_before,
        cliff_after_hz=spectral_cliff_hz(extended, target_sr),
        energy_before_pct=energy_before,
        energy_after_pct=energy_above_hz(extended, target_sr),
    )
