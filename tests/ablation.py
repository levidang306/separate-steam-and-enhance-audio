"""The per-step ablation matrix.

Eight runs per case -- every subset of {tempo, denoise, bandwidth} -- each
compared against the dry input. The point is not to produce a leaderboard but
to make one question answerable: when a result gets worse, which step did it?

A step being absent from a run is recorded separately from a step being skipped
by its gate. Those are different facts: the first says the ablation did not
offer it, the second says it was offered and declined on the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from suno_restore import chain, quality
from tests import corpus, models

SR = corpus.SR

COMBINATIONS: list[tuple[str, ...]] = [
    subset
    for size in range(4)
    for subset in combinations(chain.STEPS, size)
]


@dataclass
class AblationRow:
    case: str
    steps: tuple[str, ...]
    ran: tuple[str, ...]
    fell_back: bool
    is_enhanced: bool
    quality_score: float
    timing_score: float
    spectral_score: float
    transient_score: float
    artifact_score: float
    worst_band_db: float
    drift_ms: float
    correlation: float
    channels_preserved: bool
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "+".join(self.steps) if self.steps else "baseline"


def _goodness(value: float, limit: float) -> float:
    """1.0 when a measurement is at zero, 0.0 when it reaches its limit."""
    if limit <= 0:
        return 1.0
    return float(np.clip(1.0 - abs(value) / limit, 0.0, 1.0))


def score_run(dry: np.ndarray, outcome: chain.RestoreOutcome, case: str,
              steps: tuple[str, ...]) -> AblationRow:
    report = quality.compare(dry, outcome.audio, SR, outcome.sr)
    verdict = outcome.verdict

    # The same protected range the verification gate uses. Including 10k-16k
    # here would score a bandwidth extender's intended output as damage, and the
    # ablation and the gate would then disagree about the same run.
    worst_band = 0.0
    for name, delta in report.spectral.band_delta_db.items():
        low = float(name.split("_")[0].replace("k", "000"))
        if 100 <= low < 10000 and abs(delta) > abs(worst_band):
            worst_band = delta

    return AblationRow(
        case=case,
        steps=steps,
        ran=tuple(name for name, step in outcome.steps.items() if step.ran),
        fell_back=outcome.fell_back,
        is_enhanced=outcome.is_enhanced,
        quality_score=verdict.quality_score if verdict else 0.0,
        timing_score=_goodness(report.timing.offset_drift_ms, 5.0),
        spectral_score=_goodness(worst_band, 1.5),
        transient_score=_goodness(
            min(0.0, report.transient.attack_slope_delta_db), 0.5
        ),
        artifact_score=1.0 - (verdict.artifact_penalty if verdict else 0.0),
        worst_band_db=worst_band,
        drift_ms=report.timing.offset_drift_ms,
        correlation=report.timing.correlation_min,
        channels_preserved=report.stereo.channels_preserved,
        notes=[
            step.skip_reason for step in outcome.steps.values()
            if step.skip_reason and step.skip_reason != "not enabled for this run"
        ],
    )


def run_case(case: str, audio: np.ndarray, model_set: dict, settings=None,
             combinations_to_run: list[tuple[str, ...]] | None = None) -> list[AblationRow]:
    rows = []
    for steps in combinations_to_run or COMBINATIONS:
        outcome = chain.restore_stem(
            audio, SR, **model_set, stem_name=case, enable=steps, settings=settings
        )
        rows.append(score_run(audio, outcome, case, steps))
    return rows


def default_cases(seconds: float = 12.0) -> dict[str, np.ndarray]:
    """One representative of each class the plan asks the corpus to cover."""
    mix = corpus.full_mix(seconds)
    guitar = corpus.solo_guitar(seconds)
    return {
        "clean_full_band": corpus.clean_full_band(seconds),
        "solo_guitar": guitar,
        "solo_piano": corpus.solo_piano(seconds),
        "vocal": corpus.vocal(seconds),
        "full_mix": mix,
        "noisy_guitar": corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0),
        "bandwidth_limited": corpus.band_limit(mix, SR, cutoff_hz=8000.0),
        "codec_degraded": corpus.codec_degrade(mix, SR),
        "drifting": corpus.damage_warp(mix, SR, start_s=2.0, end_s=9.0, drift_ms=-120.0),
    }


def run_matrix(cases: dict[str, np.ndarray] | None = None, model_set: dict | None = None,
               settings=None) -> list[AblationRow]:
    cases = cases if cases is not None else default_cases()
    model_set = model_set if model_set is not None else models.MEASURED
    rows: list[AblationRow] = []
    for case, audio in cases.items():
        rows.extend(run_case(case, audio, model_set, settings))
    return rows


def format_table(rows: list[AblationRow]) -> str:
    header = (
        f"{'case':<20}{'steps enabled':<26}{'actually ran':<20}"
        f"{'qual':>7}{'time':>6}{'spec':>6}{'tran':>6}{'artf':>6}"
        f"{'band dB':>9}{'drift':>8}{'corr':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    current = None
    for row in rows:
        if row.case != current:
            if current is not None:
                lines.append("")
            current = row.case
        verdict = "DRY" if row.fell_back else ("enhanced" if row.is_enhanced else "kept")
        lines.append(
            f"{row.case:<20}{row.label:<26}{'+'.join(row.ran) or '-':<20}"
            f"{row.quality_score:>7.3f}{row.timing_score:>6.2f}{row.spectral_score:>6.2f}"
            f"{row.transient_score:>6.2f}{row.artifact_score:>6.2f}"
            f"{row.worst_band_db:>9.2f}{row.drift_ms:>8.2f}{row.correlation:>7.3f}  {verdict}"
        )
    return "\n".join(lines)
