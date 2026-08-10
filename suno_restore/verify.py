"""Check a processed stem against its dry input, and roll back if it is worse.

This is the backstop that makes every other stage reversible. Gates decide
whether a step should run and blending decides how much of it survives, but
both are predictions. This module is the only place that looks at the finished
result and asks whether the prediction held.

Two kinds of measurement, deliberately kept apart:

  * **Preservation** -- did processing damage anything? Timing, transients,
    band balance, dynamics, stereo. These are the acceptance gates, and failing
    one returns the dry signal.

  * **Restoration** -- did processing actually help? Noise removed, bandwidth
    recovered, artifacts avoided.

Both are needed. Optimising for preservation alone rewards a pipeline that does
nothing, which passes every safety check and improves nothing; optimising for
restoration alone is how the chain got into this state in the first place. A
result is only reported as enhanced when it is both safe and better.

And when it falls back, it says so. Reporting an enhancement that silently
returned the input is worse than not enhancing, because it is not true.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .audio_io import channel_count
from .config import settings as default_settings
from .quality import QualityReport, compare

EPS = 1e-12

# How much a band may move before it counts as fully spent against the budget,
# for scoring purposes. Gate limits come from Settings; these are the scales
# that turn a violation into a comparable 0-1 number.
SCORE_NOISE_SPAN_DB = 12.0
SCORE_BANDWIDTH_SPAN_DB = 20.0
SCORE_MODULATION_SPAN_DB = 3.0


@dataclass
class Check:
    """One acceptance gate, with the number that decided it."""

    name: str
    passed: bool
    value: float
    limit: float
    critical: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    passed: bool
    is_enhanced: bool
    checks: list[Check] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    preservation_penalty: float = 0.0
    restoration_gain: float = 0.0
    artifact_penalty: float = 0.0
    quality_score: float = 0.0
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload

    def summary(self) -> str:
        lines = ["PASS" if self.passed else f"FAIL -> dry ({self.fallback_reason})"]
        for check in self.checks:
            mark = "ok  " if check.passed else "FAIL"
            lines.append(f"  {mark} {check.name:22s} {check.detail}")
        lines.append(
            f"  score {self.quality_score:+.3f} = restoration {self.restoration_gain:.3f}"
            f" - artifacts {self.artifact_penalty:.3f}"
            f" - preservation {self.preservation_penalty:.3f}"
        )
        return "\n".join(lines)


def _band_violation(report: QualityReport, low_hz: float, high_hz: float) -> tuple[float, str]:
    """Largest band deviation inside the protected range."""
    worst, worst_name = 0.0, ""
    for name, delta in report.spectral.band_delta_db.items():
        edges = name.split("_")
        band_low = float(edges[0].replace("k", "000"))
        if not low_hz <= band_low < high_hz:
            continue
        if abs(delta) > abs(worst):
            worst, worst_name = delta, name
    return worst, worst_name


def build_checks(report: QualityReport, settings=None) -> list[Check]:
    """The acceptance gates, evaluated against one A/B report."""
    settings = settings or default_settings
    band_worst, band_name = _band_violation(
        report, settings.verify_band_low_hz, settings.verify_band_high_hz
    )
    added_clipping = max(
        0, report.processed.clipped_samples - report.original.clipped_samples
    )

    return [
        Check(
            name="duration",
            passed=abs(report.timing.duration_delta_ms) <= settings.verify_max_duration_delta_ms,
            value=abs(report.timing.duration_delta_ms),
            limit=settings.verify_max_duration_delta_ms,
            critical=True,
            detail=f"{report.timing.duration_delta_ms:+.1f} ms "
                   f"(limit {settings.verify_max_duration_delta_ms:.0f})",
        ),
        Check(
            name="timing_drift",
            passed=report.timing.offset_drift_ms <= settings.verify_max_offset_drift_ms,
            value=report.timing.offset_drift_ms,
            limit=settings.verify_max_offset_drift_ms,
            critical=True,
            detail=f"{report.timing.offset_drift_ms:.1f} ms "
                   f"(limit {settings.verify_max_offset_drift_ms:.0f})",
        ),
        Check(
            name="correlation",
            passed=report.timing.correlation_min >= settings.verify_min_correlation,
            value=report.timing.correlation_min,
            limit=settings.verify_min_correlation,
            critical=True,
            detail=f"min 2s window {report.timing.correlation_min:.3f} "
                   f"(floor {settings.verify_min_correlation:.2f})",
        ),
        Check(
            name="band_balance",
            passed=abs(band_worst) <= settings.verify_max_band_delta_db,
            value=abs(band_worst),
            limit=settings.verify_max_band_delta_db,
            critical=True,
            detail=f"worst {band_name or 'n/a'} {band_worst:+.2f} dB "
                   f"(limit +/-{settings.verify_max_band_delta_db:.1f})",
        ),
        Check(
            name="attack_slope",
            passed=(
                report.transient.attack_slope_delta_db
                >= -settings.verify_max_attack_slope_loss_db
            ),
            value=report.transient.attack_slope_delta_db,
            limit=-settings.verify_max_attack_slope_loss_db,
            critical=True,
            detail=f"{report.transient.attack_slope_delta_db:+.2f} dB "
                   f"(floor -{settings.verify_max_attack_slope_loss_db:.1f})",
        ),
        Check(
            name="dynamic_expansion",
            passed=(
                report.dynamic.expansion_spread_db <= settings.verify_max_expansion_spread_db
            ),
            value=report.dynamic.expansion_spread_db,
            limit=settings.verify_max_expansion_spread_db,
            critical=True,
            detail=f"gain spans {report.dynamic.expansion_spread_db:.2f} dB "
                   f"(limit {settings.verify_max_expansion_spread_db:.1f})",
        ),
        Check(
            name="channels",
            passed=report.stereo.channels_preserved,
            value=float(report.stereo.channels_processed),
            limit=float(report.stereo.channels_original),
            critical=True,
            detail=f"{report.stereo.channels_original} -> {report.stereo.channels_processed}",
        ),
        Check(
            name="stereo_image",
            passed=(
                not np.isfinite(report.stereo.side_to_mid_delta_db)
                or report.stereo.side_to_mid_delta_db > -6.0
            ),
            value=report.stereo.side_to_mid_delta_db,
            limit=-6.0,
            critical=True,
            detail=f"side/mid {report.stereo.side_to_mid_delta_db:+.1f} dB",
        ),
        Check(
            name="clipping",
            passed=added_clipping == 0,
            value=float(added_clipping),
            limit=0.0,
            critical=True,
            detail=f"{added_clipping} new clipped samples",
        ),
    ]


def score(report: QualityReport, checks: list[Check], settings=None) -> tuple[float, float, float]:
    """Preservation penalty, restoration gain and artifact penalty, each 0-1.

    Every term is expressed as a fraction of its own budget so they can be
    compared and summed. A preservation penalty of 1.0 means processing landed
    exactly on its limits; above 1.0 it has broken them.

    The weighting is deliberately flat -- each preserved property counts the
    same -- because there is no evidence yet for ranking them, and inventing a
    ranking would give the score a precision it has not earned.
    """
    settings = settings or default_settings

    penalties = []
    for check in checks:
        if check.name in {"channels", "clipping"}:
            penalties.append(0.0 if check.passed else 1.0)
        elif check.name == "correlation":
            shortfall = max(0.0, check.limit - check.value)
            penalties.append(shortfall / max(1.0 - check.limit, EPS))
        elif check.name == "attack_slope":
            penalties.append(max(0.0, -check.value) / max(abs(check.limit), EPS))
        elif check.name == "stereo_image":
            penalties.append(0.0 if check.passed else 1.0)
        else:
            penalties.append(abs(check.value) / max(abs(check.limit), EPS))
    preservation = float(np.mean(penalties)) if penalties else 0.0

    # Restoration is read off the high band, which is where both of the things
    # this pipeline can genuinely fix show up: hiss removed makes it quieter,
    # bandwidth recovered makes it louder. `band_delta_db` is already normalised
    # against program gain, so a stem that was simply turned down does not read
    # as one that had noise removed. The two are mutually exclusive, so the
    # score is the better of them rather than their average.
    high_delta = report.spectral.band_delta_db.get("10k_16k") or 0.0
    noise_gain = float(np.clip(-high_delta / SCORE_NOISE_SPAN_DB, 0.0, 1.0))

    # Two ways for added content to be related to the source, and either counts.
    # Coherence is the direct test but only works when the source has content up
    # there to compare against; on a genuinely band-limited source -- the case
    # bandwidth extension is *for* -- there is nothing to be coherent with, and
    # every extension scores zero. The envelope test still discriminates there:
    # a derived high band follows the music, an invented one does not.
    relatedness = max(
        report.artifacts.hf_coherence if np.isfinite(report.artifacts.hf_coherence) else 0.0,
        report.artifacts.hf_envelope_correlation
        if np.isfinite(report.artifacts.hf_envelope_correlation)
        else 0.0,
        0.0,
    )
    added_high_band = high_delta > 1.0
    bandwidth_gain = (
        float(np.clip(high_delta / SCORE_BANDWIDTH_SPAN_DB, 0.0, 1.0)) * relatedness
        if added_high_band
        else 0.0
    )
    restoration = max(noise_gain, bandwidth_gain)

    # Artifacts only apply when high-frequency content went *up*. Taking a
    # steady noise bed out raises the relative variability of whatever is left
    # and lowers the coherence of a band that is now mostly silence -- so scored
    # unconditionally, successful hiss removal would read as an artifact.
    #
    # Relatedness appears here as well as in the credit above, and that
    # double-counting is deliberate: content invented rather than restored is
    # the specific failure this design exists to prevent, so it both earns no
    # credit and attracts a penalty.
    artifact = 0.0
    if added_high_band:
        artifact = float(
            np.mean([
                np.clip(1.0 - relatedness, 0.0, 1.0),
                np.clip(
                    report.artifacts.hf_modulation_increase_db / SCORE_MODULATION_SPAN_DB,
                    0.0, 1.0,
                ),
            ])
        )

    return preservation, restoration, artifact


def verify(
    dry: np.ndarray,
    processed: np.ndarray,
    sr_dry: int,
    sr_processed: int | None = None,
    *,
    settings=None,
) -> Verdict:
    """Measure a processed stem against its dry input and reach a verdict."""
    settings = settings or default_settings
    sr_processed = sr_processed or sr_dry

    if not settings.verify_enabled:
        return Verdict(passed=True, is_enhanced=True, fallback_reason="",
                       metrics={"skipped": "verification disabled"})

    report = compare(dry, processed, sr_dry, sr_processed)
    checks = build_checks(report, settings)
    failed = [check.name for check in checks if not check.passed]
    critical_failed = [check.name for check in checks if not check.passed and check.critical]

    preservation, restoration, artifact = score(report, checks, settings)
    quality_score = restoration - artifact - preservation

    passed = not critical_failed
    reason = ""
    if critical_failed:
        detail = "; ".join(
            check.detail for check in checks if check.name in critical_failed
        )
        reason = f"failed {', '.join(critical_failed)}: {detail}"
    elif quality_score < settings.verify_min_quality_score:
        # Nothing was damaged, but the result is still worse. This is what an
        # invented high band looks like: it adds rather than removes, so every
        # preservation check passes while the artifact penalty says the added
        # content bears no relation to the source.
        passed = False
        reason = (
            f"quality score {quality_score:.3f} below "
            f"{settings.verify_min_quality_score}: artifacts {artifact:.2f} outweigh "
            f"restoration {restoration:.2f}"
        )

    return Verdict(
        passed=passed,
        # Safe is not the same as better. A result that changed nothing passes
        # every gate and has not enhanced anything.
        is_enhanced=passed and quality_score > settings.verify_min_quality_gain,
        checks=checks,
        failed_checks=failed,
        fallback_reason=reason,
        preservation_penalty=preservation,
        restoration_gain=restoration,
        artifact_penalty=artifact,
        quality_score=quality_score,
        metrics=report.to_dict(),
    )


def accept_or_rollback(
    dry: np.ndarray,
    processed: np.ndarray,
    sr_dry: int,
    sr_processed: int | None = None,
    *,
    settings=None,
) -> tuple[np.ndarray, int, Verdict]:
    """Return the processed stem, or the dry one if processing made it worse.

    The dry signal is the fallback, always available because no stage was ever
    allowed to overwrite it. Returns (audio, sample_rate, verdict) so a caller
    that falls back also gets the dry rate back, rather than keeping a rate that
    belongs to a result it discarded.
    """
    settings = settings or default_settings
    sr_processed = sr_processed or sr_dry
    verdict = verify(dry, processed, sr_dry, sr_processed, settings=settings)

    if not verdict.passed:
        return dry, sr_dry, verdict
    if channel_count(processed) != channel_count(dry):  # belt and braces
        verdict.passed = False
        verdict.is_enhanced = False
        verdict.fallback_reason = "channel count changed"
        return dry, sr_dry, verdict
    return processed, sr_processed, verdict


__all__ = ["Check", "Verdict", "accept_or_rollback", "build_checks", "score", "verify"]
