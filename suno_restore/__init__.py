"""Restoration workflow for stems exported by Suno.

Three defects, one step each, run in this order:

  1. tempo drift        -> tempo.correct_tempo   (Beat This!)
  2. hiss / crackle     -> denoise.denoise_stem  (Mel-Roformer-Denoise)
  3. bandwidth loss     -> bandwidth.extend_stem (Apollo)

Each step is individually switchable and, when switched on, runs at full
strength. `quality.py` measures what a run actually did -- timing, bands,
transients, dynamics, stereo, artifacts -- and is the way to check a result
rather than assume one.
"""

from .audio_io import TARGET_SR, load_audio, save_audio
from .config import Settings, settings
from .pipeline import RunReport, StemReport, discover_stems, restore
from .quality import QualityReport, compare, compare_files

__all__ = [
    "TARGET_SR",
    "QualityReport",
    "RunReport",
    "Settings",
    "StemReport",
    "compare",
    "compare_files",
    "discover_stems",
    "load_audio",
    "restore",
    "save_audio",
    "settings",
]
