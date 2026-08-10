"""Restoration settings, read once from the environment.

Every default here is chosen to be the *inert* option. A fresh install with no
environment file must behave conservatively: tempo correction off, wet/dry
blends partial, attenuation capped, verification on. The reason is the failure
this configuration exists to prevent -- an enhancement chain that runs
unconditionally will happily make good audio worse, and it does so silently,
because nothing in the old design compared the result against its input.

The rule the defaults encode: **if a step cannot prove it should run, it does
not run.**
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

    # Refuse to write a stem whose channel count does not match its input.
    # A collapse is a silent, unrecoverable loss, so it is worth failing over.
    enforce_channel_count: bool = True

    # --- Tempo gate ------------------------------------------------------
    # Off. On the reference material tempo correction was the single most
    # destructive stage: it warped a solo guitar by 375ms using a beat grid that
    # covered 41% of the track. It only helps material with a real, steady,
    # percussive pulse, and that has to be demonstrated before it runs.
    tempo_enabled: bool = False

    # A grid that covers less than this cannot describe the track's timing; the
    # uncovered regions get copied and inherit whatever drift the covered ones
    # accumulated.
    tempo_min_grid_coverage: float = 0.80

    # Beat-to-beat tempo has to be stable before a "correct" tempo exists to
    # warp towards. Expressed as a percentage of the median BPM.
    tempo_max_bpm_std_pct: float = 2.0

    # Onset strength expected from percussion. Melodic material scores far
    # below this, which is the point: a guitar is not a click track.
    tempo_min_percussive_score: float = 0.30

    tempo_min_confidence: float = 0.60

    # Below this deviation, stretching costs more smearing than the drift is
    # worth. Raised from 0.005: at that level the step spent phase-vocoder
    # damage on rounding error.
    tempo_min_correction: float = 0.02

    # Anything outside this is a beat-tracking failure, not tempo drift. The
    # old (0.80, 1.25) range let a mis-detected beat stretch a segment by a
    # quarter of its length.
    tempo_max_rate_low: float = 0.98
    tempo_max_rate_high: float = 1.02

    # Total timing movement the step is allowed to introduce, end to end.
    tempo_max_cumulative_drift_ms: float = 50.0

    # Separated stems have had their transients reshaped by the separator, so
    # beat tracking on them is less trustworthy than on a full mix.
    tempo_allow_on_separated: bool = False

    # --- Denoise ---------------------------------------------------------
    denoise_enabled: bool = True

    # Partial by default. At 1.0 the model's output replaces the signal, and
    # everything it removes -- including room tone, reverb tails and finger
    # noise -- is gone with no way back.
    denoise_wet: float = 0.5

    # Denoise is allowed to act freely above 10kHz, where hiss actually lives,
    # and barely at all in the fundamental range, where it only removes music.
    #
    # The 5-10kHz weight is 0.15 rather than the 0.40 the plan starts from, and
    # the change is the benchmark's answer to the plan's own instruction to test
    # these values rather than assume them. Swept from 0.0 to 0.40
    # (`scripts/sweep.py`), noise removal is flat at 1.30-1.32dB across the whole
    # range while presence loss grows from -0.71dB to -1.40dB in the 8-10kHz
    # band. The weight buys 0.02dB of hiss for 0.69dB of the band that carries
    # pick attack and string articulation. 0.15 is the highest value that keeps
    # the reference guitar inside the +/-1dB target with the worst-case model.
    denoise_weight_below_2k: float = 0.0
    denoise_weight_2k_5k: float = 0.15
    denoise_weight_5k_10k: float = 0.15
    denoise_weight_above_10k: float = 1.00

    # Hard per-frame floor on how much any restoration stage may pull the
    # source down. Protects decay, room tone, reverb and transient tails from
    # the downward-expander behaviour measured on the reference file.
    max_attenuation_db: float = 6.0

    # Only bother denoising when there is measurable noise to remove. The
    # absolute floor catches material that is simply clean; the hiss figure --
    # the stationary high-band bed, relative to program level -- is the one that
    # actually discriminates, because that is the band the denoiser is weighted
    # to act in, and because a broadband floor number is dominated by ringing
    # low-frequency tails that are music, not noise.
    denoise_min_noise_floor_db: float = -75.0
    denoise_min_hiss_db: float = -70.0

    # Below this the wet contribution would be too small to hear anyway, so the
    # step is not worth the model pass or the risk of running it at all.
    denoise_min_confidence: float = 0.05

    # --- Bandwidth -------------------------------------------------------
    bandwidth_enabled: bool = True
    bandwidth_wet: float = 0.7

    # A cliff outside this range is not a codec cutoff worth restoring. Below
    # the low bound it is tonal balance (a bass stem has no treble to begin
    # with); above the high bound there is nothing meaningful left to add.
    bandwidth_min_cliff_hz: float = 8000.0
    bandwidth_max_cliff_hz: float = 20000.0

    # The signal must actually reach the cliff. The reference guitar rolled off
    # at 2.8kHz and had no 8-20kHz cliff at all, so synthesising a high band for
    # it invented content with 0.012 coherence to the source.
    bandwidth_min_energy_below_cliff_pct: float = 1.0

    # How related the model's high band has to be to the material it was given,
    # checked on the raw model output before any blending. This cannot be left
    # to the final verification: the damage mask follows the music, so blending
    # imposes a music-shaped envelope on whatever the model returned and an
    # invented band ends up looking derived. Measured on the model's own output
    # the two are far apart -- around 0.9 for a derived band against roughly 0
    # for an invented one.
    bandwidth_min_relatedness: float = 0.30

    # Apollo chunking. 2 seconds of equal-power overlap replaces 0.5 seconds of
    # linear ramp: the short linear crossfade left a level swing that tracked
    # the chunk stride.
    bandwidth_chunk_s: float = 10.0
    bandwidth_overlap_s: float = 2.0

    # --- Damage mask -----------------------------------------------------
    damage_frame_ms: float = 20.0
    damage_feather_ms: float = 100.0

    # --- Verification ----------------------------------------------------
    # On. This is the backstop that makes every other stage reversible.
    verify_enabled: bool = True

    verify_max_duration_delta_ms: float = 5.0
    verify_max_offset_drift_ms: float = 5.0
    verify_min_correlation: float = 0.90
    verify_max_band_delta_db: float = 1.5
    verify_band_low_hz: float = 100.0
    verify_band_high_hz: float = 10000.0
    verify_max_attack_slope_loss_db: float = 0.5
    verify_max_expansion_spread_db: float = 6.0

    # A quality score this close to zero is measurement noise, not improvement.
    # Without the dead zone a pure level change reads as an enhancement, and
    # `is_enhanced` has to mean something for the fallback report to be useful.
    verify_min_quality_gain: float = 0.02

    # Roll back when the score is this far negative, even if no preservation
    # gate broke. Some damage is not damage to the *input* -- a synthesised high
    # band with no relation to the source adds something rather than taking
    # anything away, so it passes every preservation check while making the
    # result worse. The plan requires a measurable regression to restore the dry
    # signal automatically, and this is the form such a regression takes.
    verify_min_quality_score: float = -0.25

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

    @property
    def tempo_rate_bounds(self) -> tuple[float, float]:
        return self.tempo_max_rate_low, self.tempo_max_rate_high

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
