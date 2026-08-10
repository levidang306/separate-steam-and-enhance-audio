"""Restoration settings, read once from the environment.

Every field maps to an environment variable named `RESTORE_<FIELD_NAME>`. Note
that nothing loads a `.env` file: these are read straight from the process
environment, so a `.env` sitting in the working directory has no effect unless
something else exports it first.

The pipeline runs each enabled step unconditionally and at full strength. What
is left here is the small amount of configuration that describes the *signal* --
what sample rate to write, how to keep a stereo image through a mono-only model
-- plus the chunk sizes the bandwidth model runs at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    """Restoration configuration. Construct through `Settings.from_env()`."""

    # --- Signal integrity ------------------------------------------------
    # 0 means "follow the source". Forcing a fixed rate resamples material that
    # never needed it, and a resampler is not free -- it is a filter.
    output_sample_rate: int = 0

    # How to keep stereo through a model that may only emit mono. "auto" runs
    # the model normally and only pays for a mid/side pass if the output came
    # back with fewer channels than went in; "mid_side" always does; "joint"
    # trusts the model. "auto" is the default because it is correct in both
    # cases and costs nothing when the model behaves.
    stereo_mode: str = "auto"

    # Refuse to write a stem whose channel count does not match its input. This
    # is an assertion against a specific bug, not a policy decision: the denoise
    # model can write a mono file, and adopting its channel count turned every
    # stereo stem into a mono one while reporting success.
    enforce_channel_count: bool = True

    # --- Bandwidth chunking ----------------------------------------------
    # Apollo is a sequence model and cannot take a whole stem at once, so it
    # runs in overlapping chunks that are crossfaded together. 2s of overlap
    # rather than 0.5s: the short fade left a level swing in the synthesised
    # high band that tracked the chunk stride exactly.
    bandwidth_chunk_s: float = 10.0
    bandwidth_overlap_s: float = 2.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Read every field from `RESTORE_<FIELD_NAME>`, falling back to its default.

        Derived from the dataclass rather than written out field by field. The
        hand-written version listed each default twice -- once on the field and
        once as the fallback here -- and the two drifted apart the first time a
        default changed, so `Settings()` and `Settings.from_env()` disagreed
        about the same setting with nothing to catch it.
        """
        values = {}
        for field_info in fields(cls):
            name = f"RESTORE_{field_info.name.upper()}"
            default = field_info.default
            if isinstance(default, bool):
                values[field_info.name] = _env_bool(name, default)
            elif isinstance(default, int):
                values[field_info.name] = _env_int(name, default)
            elif isinstance(default, float):
                values[field_info.name] = _env_float(name, default)
            else:
                values[field_info.name] = _env_str(name, default)
        return cls(**values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Module-level singleton, matching how the rest of the engine holds
# configuration. Tests build their own `Settings(...)` and pass it in rather
# than mutating this.
settings = Settings.from_env()


def resolve_output_sample_rate(source_sr: int, configured: int | None = None) -> int:
    """Sample rate a stage should write at.

    `RESTORE_OUTPUT_SAMPLE_RATE=0` means follow the source, which is the
    default: resampling that nobody asked for is a filter nobody asked for.
    """
    value = settings.output_sample_rate if configured is None else configured
    return source_sr if not value else int(value)


__all__ = ["Settings", "resolve_output_sample_rate", "settings"]
