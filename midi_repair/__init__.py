"""Version 2 of the merged restoration pipeline: localized MIDI-based repair.

The model choices and detection thresholds come from the rewrite-v2
Streamlit prototype. The renewed merge adds reliable ffmpeg decoding,
native-rate/channel output, and a risk-based rollback; see
docs/design/2026-08-10-merged-pipeline.md.
"""

from .repair import EnhanceReport, enhance, preserve_source_layout, unload_models

__all__ = ["EnhanceReport", "enhance", "preserve_source_layout", "unload_models"]
