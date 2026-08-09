"""Restoration workflow for stems exported by Suno.

Three defects, one step each:
  1. tempo drift        -> tempo.correct_tempo   (Beat This!)
  2. hiss / crackle     -> denoise.denoise_stem  (Mel-Roformer-Denoise)
  3. bandwidth loss     -> bandwidth.extend_stem (Apollo)
"""

from .audio_io import TARGET_SR, load_audio, save_audio
from .pipeline import RunReport, StemReport, discover_stems, restore

__all__ = [
    "TARGET_SR",
    "RunReport",
    "StemReport",
    "discover_stems",
    "load_audio",
    "restore",
    "save_audio",
]
