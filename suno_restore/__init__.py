"""Restoration workflow for stems exported by Suno.

Three defects, one step each:
  1. tempo drift        -> tempo.correct_tempo   (Beat This!)
  2. hiss / crackle     -> denoise.denoise_stem  (Mel-Roformer-Denoise)
  3. bandwidth loss     -> bandwidth.extend_stem (Apollo)

None of them runs unless it can show it should. Each step is gated on evidence
(`gate`), aimed at the damage (`damage`), mixed rather than substituted
(`blend`), and checked against its own input afterwards (`verify`) -- with the
dry signal kept throughout as the fallback. `chain` is the assembly; `quality`
is the measurement everything else is argued from.
"""

from .audio_io import TARGET_SR, load_audio, save_audio
from .chain import RestoreOutcome, restore_stem
from .config import Settings, settings
from .gate import StepDecision
from .pipeline import RunReport, StemReport, discover_stems, restore
from .quality import QualityReport, compare, compare_files
from .verify import Verdict

__all__ = [
    "TARGET_SR",
    "QualityReport",
    "RestoreOutcome",
    "RunReport",
    "Settings",
    "StemReport",
    "StepDecision",
    "Verdict",
    "compare",
    "compare_files",
    "discover_stems",
    "load_audio",
    "restore",
    "restore_stem",
    "save_audio",
    "settings",
]
