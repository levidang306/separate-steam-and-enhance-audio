"""The restoration chain: analyse, decide, restore selectively, verify, or roll back.

    INPUT -> ANALYSIS -> DAMAGE DETECTION -> STEP DECISION
          -> SELECTIVE RESTORATION -> WET/DRY BLEND
          -> QUALITY VERIFICATION -> ACCEPT or DRY -> OUTPUT

The dry signal is held from the first line to the last and is never written
over. Every step produces a candidate that is blended into a working copy, and
if the finished result fails verification the dry signal is what ships. That is
the entire difference from the previous design, in which each step replaced the
signal with its own output and nothing afterwards was in a position to disagree.

The three models are injected rather than imported. They are large, they are
V3-image-only, and none of them is needed to exercise the architecture -- which
is the part that decides whether a step runs, how much of it survives, and
whether the result is kept. That separation is what makes the ablation matrix
in `tests/test_ablation.py` possible at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import blend, damage, gate, verify
from .audio_io import channel_count
from .config import settings as default_settings

Model = Callable[[np.ndarray, int], np.ndarray]

STEPS = ("tempo", "denoise", "bandwidth")


@dataclass
class StepOutcome:
    """What happened to one step, and why."""

    name: str
    decision: gate.StepDecision
    ran: bool
    skip_reason: str = ""
    damage_coverage: float = 0.0
    blend_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ran": self.ran,
            "skip_reason": self.skip_reason,
            "damage_coverage": self.damage_coverage,
            "blend": self.blend_summary,
            **self.decision.to_dict(),
        }


@dataclass
class RestoreOutcome:
    audio: np.ndarray
    sr: int
    is_enhanced: bool
    fell_back: bool
    steps: dict[str, StepOutcome] = field(default_factory=dict)
    verdict: verify.Verdict | None = None

    @property
    def engine_config(self) -> dict:
        """Everything a later reader needs to explain this run."""
        return {
            "is_enhanced": self.is_enhanced,
            "fell_back_to_dry": self.fell_back,
            "steps": {name: outcome.to_dict() for name, outcome in self.steps.items()},
            "verification": self.verdict.to_dict() if self.verdict else {},
        }

    def summary(self) -> str:
        lines = []
        for name in STEPS:
            outcome = self.steps.get(name)
            if outcome is None:
                continue
            state = "RAN " if outcome.ran else "SKIP"
            reason = outcome.skip_reason or outcome.decision.reason
            lines.append(f"  {state} {name:10s} {reason}")
        lines.append(
            f"  {'DRY ' if self.fell_back else 'KEPT'} "
            f"is_enhanced={self.is_enhanced}"
            + (f"  ({self.verdict.fallback_reason})" if self.fell_back else "")
        )
        return "\n".join(lines)


def restore_stem(
    audio: np.ndarray,
    sr: int,
    *,
    tempo_model: Model | None = None,
    denoise_model: Model | None = None,
    bandwidth_model: Model | None = None,
    stem_name: str = "",
    available_stems: list[str] | None = None,
    is_separated: bool = False,
    enable: tuple[str, ...] = STEPS,
    settings=None,
) -> RestoreOutcome:
    """Run the chain over one stem.

    `enable` names the steps this run is allowed to consider, which is how the
    ablation matrix is built: a step outside it is not disabled by configuration
    or by evidence, it simply was not offered. That distinction is recorded, so
    an ablation result cannot be mistaken for a gate decision.
    """
    settings = settings or default_settings
    dry = np.array(audio, copy=True)
    working = dry
    working_sr = sr
    outcomes: dict[str, StepOutcome] = {}

    decisions = gate.decide_all(
        dry, sr, stem_name=stem_name, available_stems=available_stems,
        is_separated=is_separated, settings=settings,
    )

    for name in STEPS:
        decision = decisions[name]
        model = {"tempo": tempo_model, "denoise": denoise_model,
                 "bandwidth": bandwidth_model}[name]

        if name not in enable:
            outcomes[name] = StepOutcome(name, decision, ran=False,
                                         skip_reason="not enabled for this run")
            continue
        if not decision.run:
            outcomes[name] = StepOutcome(name, decision, ran=False,
                                         skip_reason=decision.reason)
            continue
        if model is None:
            outcomes[name] = StepOutcome(name, decision, ran=False,
                                         skip_reason="no model supplied")
            continue

        working, working_sr, outcomes[name] = _apply(
            name, decision, model, working, working_sr, settings
        )

    final, final_sr, verdict = verify.accept_or_rollback(
        dry, working, sr, working_sr, settings=settings
    )
    fell_back = not verdict.passed

    return RestoreOutcome(
        audio=final,
        sr=final_sr,
        is_enhanced=verdict.is_enhanced,
        fell_back=fell_back,
        steps=outcomes,
        verdict=verdict,
    )


def _apply(
    name: str,
    decision: gate.StepDecision,
    model: Model,
    working: np.ndarray,
    sr: int,
    settings,
) -> tuple[np.ndarray, int, StepOutcome]:
    """Run one model and fold its output into the working copy."""
    wet = model(working, sr)

    if name == "tempo":
        # Tempo is the one step with nothing to blend: it moves the time axis,
        # and a wet/dry mix of two different timelines is comb filtering, not a
        # gentler correction. It is all or nothing, and the final verification
        # is what makes that safe.
        outcome = StepOutcome(name, decision, ran=True, damage_coverage=1.0,
                              blend_summary={"mode": "replace", "reason": "time axis"})
        return wet, sr, outcome

    if channel_count(wet) != channel_count(working):
        return working, sr, StepOutcome(
            name, decision, ran=False,
            skip_reason=f"model returned {channel_count(wet)} channels, expected "
                        f"{channel_count(working)}",
        )

    if name == "denoise":
        mask = damage.noise_damage(working, sr, settings=settings)
        result = blend.blend_denoise(
            working, wet, sr, confidence=decision.confidence, mask=mask.mask, settings=settings
        )
    else:
        cliff = float(decision.measurements.get("cliff_hz") or settings.bandwidth_min_cliff_hz)
        relatedness = _relatedness(working, wet, sr, cliff)
        if relatedness < settings.bandwidth_min_relatedness:
            return working, sr, StepOutcome(
                name, decision, ran=False,
                skip_reason=(
                    f"the model's output above {cliff:.0f} Hz is unrelated to the material it "
                    f"was given (relatedness {relatedness:.2f} < "
                    f"{settings.bandwidth_min_relatedness}); that is invention, not restoration"
                ),
            )
        mask = damage.bandwidth_damage(working, sr, cliff, settings=settings)
        result = blend.blend_bandwidth(
            working, wet, sr, cliff_hz=cliff, confidence=decision.confidence,
            mask=mask.mask, settings=settings,
        )

    outcome = StepOutcome(
        name, decision, ran=True,
        damage_coverage=mask.coverage,
        blend_summary={**result.to_dict(), "mask": mask.to_dict()},
    )
    return result.audio, sr, outcome


def _relatedness(source: np.ndarray, candidate: np.ndarray, sr: int, cliff_hz: float) -> float:
    """How much of the model's high band came from the material it was given.

    Two tests, and either passing is enough. Coherence is direct but needs the
    source to have content above the cliff to compare against -- which, for a
    band-limited source, is exactly what it does not have. The envelope test
    works there: a derived high band rises and falls with the music, an invented
    one keeps its own contour.
    """
    from .quality import measure_artifacts

    artifacts = measure_artifacts(source, candidate, sr, hf_low=cliff_hz)
    values = [artifacts.hf_coherence, artifacts.hf_envelope_correlation]
    return max([value for value in values if np.isfinite(value)] + [0.0])


__all__ = ["Model", "RestoreOutcome", "STEPS", "StepOutcome", "restore_stem"]
