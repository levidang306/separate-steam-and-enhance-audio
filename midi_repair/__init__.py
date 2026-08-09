"""Stage A of the merged restoration pipeline: localized MIDI-based repair.

Ported from the rewrite-v2 Streamlit prototype (streamlit_app.py) with the
Streamlit calls stripped out. Ported, not rewritten -- the model choices and
thresholds are unchanged; see docs/design/2026-08-10-merged-pipeline.md for
why this stage still exists next to suno_restore's whole-track pipeline.
"""

from .repair import EnhanceReport, enhance, unload_models

__all__ = ["EnhanceReport", "enhance", "unload_models"]
