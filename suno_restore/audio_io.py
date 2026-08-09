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
    path: str | Path, target_sr: int | None = TARGET_SR, mono: bool = False
) -> tuple[np.ndarray, int]:
    """Decode audio to float32 with shape (samples,) or (samples, channels)."""
    ffmpeg, _ = _binaries()
    source_sr, channels = probe(path)
    sr = target_sr or source_sr
    out_channels = 1 if mono else channels

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
