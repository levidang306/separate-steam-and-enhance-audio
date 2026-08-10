"""Audio loading and saving for Suno stem exports.

Everything decodes through ffmpeg rather than soundfile/librosa. Suno's stem
exports are very low bitrate (32-36 kbps) MP3s without dependable VBR headers,
and libsndfile 1.2.2 silently truncates most of them -- on the reference stem
set it read 87s of a 294s bass stem and 68s of a 294s backing-vocal stem, with
`sf.info` reporting the same wrong length, so nothing downstream can notice.
ffmpeg decodes all of them in full.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soxr

TARGET_SR = 48000

# winget installs ffmpeg here but only amends PATH for new shells, so a running
# process started before the install cannot see it.
_WINGET_FFMPEG = Path(
    os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0-full_build\bin"
    )
)


class FFmpegNotFound(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _binaries() -> tuple[str, str]:
    """Locate ffmpeg and ffprobe, falling back to the winget install location."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not (ffmpeg and ffprobe) and _WINGET_FFMPEG.is_dir():
        candidate_ffmpeg = _WINGET_FFMPEG / "ffmpeg.exe"
        candidate_ffprobe = _WINGET_FFMPEG / "ffprobe.exe"
        if candidate_ffmpeg.exists() and candidate_ffprobe.exists():
            ffmpeg, ffprobe = str(candidate_ffmpeg), str(candidate_ffprobe)
    if not (ffmpeg and ffprobe):
        raise FFmpegNotFound(
            "ffmpeg/ffprobe not found. Install with: winget install Gyan.FFmpeg"
        )
    return ffmpeg, ffprobe


def probe(path: str | Path) -> tuple[int, int]:
    """Return (sample_rate, channels) read from the audio stream.

    Stream fields are reliable on these files; container-level duration is not,
    so it is deliberately not read here.
    """
    _, ffprobe = _binaries()
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def load_audio(
    path: str | Path,
    target_sr: int | None = TARGET_SR,
    mono: bool = False,
    channels: int | None = None,
) -> tuple[np.ndarray, int]:
    """Decode audio to float32 with shape (samples,) or (samples, channels).

    `channels` forces the layout regardless of what the file holds. Callers use
    it when the file's own channel count must not be allowed to decide the
    result -- reading a model's output, for instance, where a mono file would
    otherwise silently turn a stereo stem into a mono one.
    """
    ffmpeg, _ = _binaries()
    source_sr, file_channels = probe(path)
    sr = target_sr or source_sr
    out_channels = channels if channels else (1 if mono else file_channels)

    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(out_channels), "-ar", str(sr), "-",
        ],
        capture_output=True, check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if out_channels > 1:
        audio = audio.reshape(-1, out_channels)
    return np.ascontiguousarray(audio), sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write float32 audio as 24-bit PCM WAV."""
    import soundfile as sf  # writing WAV is unaffected by the MP3 decoding issue

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_24")
    return path


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample, preserving the (samples, channels) layout."""
    if sr_from == sr_to:
        return audio
    return soxr.resample(audio, in_rate=sr_from, out_rate=sr_to).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def channel_count(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def to_mid_side(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split stereo into mid and side, both mono.

    This is the escape hatch for a model that only emits mono. Feeding it the
    downmix and calling the result stereo throws away the side channel
    outright; feeding it mid and side separately keeps the stereo image, at the
    cost of a second pass.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"mid/side needs 2 channels, got shape {audio.shape}")
    left, right = audio[:, 0], audio[:, 1]
    return (left + right) / 2.0, (left - right) / 2.0


def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Inverse of `to_mid_side`, exact up to float rounding."""
    n = min(len(mid), len(side))
    mid, side = mid[:n], side[:n]
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Force a channel layout, duplicating or downmixing as needed.

    A last resort. It keeps the *count* correct but cannot recreate side
    information that a stage already discarded, so callers should prefer
    mid/side processing and use this only to satisfy a hard contract.
    """
    current = channel_count(audio)
    if current == channels:
        return audio
    if channels == 1:
        return to_mono(audio)
    mono = to_mono(audio)
    return np.repeat(mono[:, None], channels, axis=1).astype(np.float32)


def is_effectively_silent(audio: np.ndarray, peak_threshold_db: float = -45.0) -> bool:
    """True when a stem carries no usable signal.

    Suno exports can contain a stem that is essentially digital silence (the
    reference set has one at -52 dB peak). Restoration steps have nothing to act
    on there, and loudness-relative operations are undefined, so callers pass
    such stems through untouched.
    """
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    return peak_db < peak_threshold_db
