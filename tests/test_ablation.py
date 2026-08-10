"""Phase 7: which step helps, which hurts, and which combinations interact.

The matrix is run once per model set at module scope -- eight combinations
across several classes is not cheap, and every test here asks a different
question of the same runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import chain
from tests import ablation, corpus, models

SR = corpus.SR


def _cases() -> dict[str, np.ndarray]:
    """A subset of the corpus: one clean, one solo, one noisy, one band-limited."""
    mix = corpus.full_mix(10.0)
    guitar = corpus.solo_guitar(10.0)
    return {
        "clean_full_band": corpus.clean_full_band(10.0),
        "solo_guitar": guitar,
        "noisy_guitar": corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0),
        "bandwidth_limited": corpus.band_limit(mix, SR, cutoff_hz=8000.0),
    }


@pytest.fixture(scope="module")
def measured() -> list[ablation.AblationRow]:
    """The chain given models that misbehave exactly as the real ones did."""
    return ablation.run_matrix(_cases(), models.MEASURED)


@pytest.fixture(scope="module")
def well_behaved() -> list[ablation.AblationRow]:
    return ablation.run_matrix(_cases(), models.WELL_BEHAVED)


def _row(rows: list[ablation.AblationRow], case: str,
         steps: tuple[str, ...]) -> ablation.AblationRow:
    for row in rows:
        if row.case == case and row.steps == steps:
            return row
    raise AssertionError(f"no row for {case} / {steps}")


# --- Matrix shape --------------------------------------------------------


def test_the_matrix_covers_every_combination(measured):
    assert len(ablation.COMBINATIONS) == 8
    assert set(ablation.COMBINATIONS) == {
        (), ("tempo",), ("denoise",), ("bandwidth",),
        ("tempo", "denoise"), ("tempo", "bandwidth"), ("denoise", "bandwidth"),
        ("tempo", "denoise", "bandwidth"),
    }
    for case in _cases():
        assert len([row for row in measured if row.case == case]) == 8


def test_the_baseline_changes_nothing(measured):
    """With no step enabled the chain must be a pass-through."""
    for case in _cases():
        row = _row(measured, case, ())
        assert row.ran == ()
        assert row.timing_score == 1.0
        assert row.spectral_score == 1.0
        assert row.correlation > 0.999
        assert not row.fell_back


def test_a_disabled_step_is_distinguishable_from_a_gated_one(measured):
    """The ablation not offering a step is a different fact from the gate
    declining it, and the record has to keep them apart."""
    outcome = chain.restore_stem(
        corpus.solo_guitar(10.0), SR, **models.MEASURED, enable=("denoise", "bandwidth")
    )
    assert outcome.steps["tempo"].skip_reason == "not enabled for this run"
    assert outcome.steps["denoise"].skip_reason == ""
    # Bandwidth was offered and declined on the evidence, which reads
    # differently from never having been offered.
    assert "cliff" in outcome.steps["bandwidth"].skip_reason


# --- Which step hurts ----------------------------------------------------


def test_no_combination_damages_clean_material(measured):
    """The property that matters most, checked across all eight runs."""
    for steps in ablation.COMBINATIONS:
        row = _row(measured, "clean_full_band", steps)
        assert row.correlation > 0.98, f"{row.label} broke correlation"
        assert abs(row.worst_band_db) <= 0.5, f"{row.label} moved a band {row.worst_band_db:+.2f}"
        assert row.drift_ms <= 5.0, f"{row.label} drifted {row.drift_ms:.1f} ms"
        assert row.channels_preserved


def test_tempo_never_runs_on_any_of_these_cases(measured):
    """It is off by default, and none of this material would justify it anyway."""
    for row in measured:
        assert "tempo" not in row.ran


def test_bandwidth_only_runs_where_there_is_a_cutoff(measured):
    for row in measured:
        if "bandwidth" in row.ran:
            assert row.case == "bandwidth_limited"


def test_denoise_only_runs_where_there_is_noise(measured):
    for row in measured:
        if "denoise" in row.ran:
            assert row.case in {"solo_guitar", "noisy_guitar"}


def test_an_inventing_model_is_refused_after_it_runs(measured):
    """The question the matrix exists to answer: which step was responsible?

    The bandwidth gate lets this material through -- there is a real cutoff --
    and the model then returns a high band unrelated to what it was given. That
    is caught after the model runs and before its output is applied, so the step
    reverts on its own rather than costing the whole stem a rollback.
    """
    offered = [row for row in measured
               if row.case == "bandwidth_limited" and "bandwidth" in row.steps]
    assert offered

    for row in offered:
        assert "bandwidth" not in row.ran, "an invented high band must not be applied"
    for row in measured:
        if row.ran == ("denoise",):
            assert not row.fell_back

    outcome = chain.restore_stem(
        corpus.band_limit(corpus.full_mix(10.0), SR, cutoff_hz=8000.0), SR,
        **models.MEASURED, enable=("bandwidth",),
    )
    step = outcome.steps["bandwidth"]
    assert step.decision.run, "the gate should accept the material"
    assert not step.ran, "and the model's output should still be refused"
    assert "unrelated" in step.skip_reason


# --- Which step helps ----------------------------------------------------


def test_a_good_bandwidth_model_is_accepted_where_a_bad_one_is_rejected(
    measured, well_behaved
):
    """The central claim of the design, stated as a single comparison.

    Same input, same gate decision to run, same blend. The only difference is
    whether the model's output is derived from the source or invented, and the
    verification layer is what tells them apart.
    """
    bad = _row(measured, "bandwidth_limited", ("bandwidth",))
    good = _row(well_behaved, "bandwidth_limited", ("bandwidth",))

    assert "bandwidth" not in bad.ran, "an invented band is refused"
    assert "bandwidth" in good.ran, "a derived one is applied"
    assert good.is_enhanced and not bad.is_enhanced
    assert good.quality_score > bad.quality_score
    # Neither damages the input; the difference is entirely about whether the
    # added content means anything.
    assert bad.correlation > 0.98 and good.correlation > 0.98


def test_a_good_denoiser_helps_a_noisy_stem(well_behaved):
    row = _row(well_behaved, "noisy_guitar", ("denoise",))

    assert "denoise" in row.ran
    assert not row.fell_back
    assert row.timing_score == 1.0
    assert row.channels_preserved


def test_steps_do_not_interact_badly(measured, well_behaved):
    """Adding a step must not make another step's result worse.

    Each stage sees the working copy the previous stage produced, so a
    regression that only appears in combination would be a real coupling bug.
    """
    for rows in (measured, well_behaved):
        for case in _cases():
            alone = {
                step: _row(rows, case, (step,)) for step in ("denoise", "bandwidth")
            }
            both = _row(rows, case, ("denoise", "bandwidth"))
            worst_alone = min(row.quality_score for row in alone.values())
            assert both.quality_score >= worst_alone - 0.05, (
                f"{case}: combining scored {both.quality_score:.3f} against "
                f"{worst_alone:.3f} for the worse step alone"
            )


def test_the_full_chain_is_never_worse_than_the_baseline_on_clean_input(
    measured, well_behaved
):
    for rows in (measured, well_behaved):
        baseline = _row(rows, "clean_full_band", ())
        full = _row(rows, "clean_full_band", chain.STEPS)
        assert full.correlation >= baseline.correlation - 0.001
        assert abs(full.worst_band_db) <= abs(baseline.worst_band_db) + 0.1


# --- Reporting -----------------------------------------------------------


def test_the_table_renders(measured):
    table = ablation.format_table(measured)
    assert "baseline" in table
    assert "tempo+denoise+bandwidth" in table
    assert "clean_full_band" in table
