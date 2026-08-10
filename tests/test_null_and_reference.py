"""Phase 10 and Phase 9: the two cases the whole design is judged on.

The null test is the more important of the two. A pipeline that improves damaged
audio but quietly degrades good audio is worse than no pipeline, because most
material is already fine and the degradation is invisible until someone listens
carefully to something they cared about.

Both run with the *measured* stand-in models -- the ones that reproduce how the
real models were observed to misbehave. Passing with well-behaved models would
show nothing; the claim being tested is that the architecture holds even when
what it is given is as bad as what was measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import chain, quality
from suno_restore.config import Settings
from tests import corpus, models

SR = corpus.SR

# Phase 10 targets, verbatim from the plan.
NULL_MIN_CORRELATION = 0.98
NULL_MAX_BAND_DEVIATION_DB = 0.5
NULL_MAX_DRIFT_MS = 5.0

# Phase 9 target for the reference guitar.
PRESENCE_TOLERANCE_DB = 1.0


def _run(audio: np.ndarray, **kwargs) -> chain.RestoreOutcome:
    return chain.restore_stem(audio, SR, **models.MEASURED, **kwargs)


# --- Phase 10: the null test --------------------------------------------


@pytest.mark.parametrize("name", ["clean_full_band", "drums", "vocal"])
def test_clean_material_comes_back_essentially_unchanged(name):
    """The plan's headline safety requirement."""
    clean = corpus.CLEAN_CLASSES[name](14.0)
    outcome = _run(clean, stem_name=name)
    report = quality.compare(clean, outcome.audio, SR, SR)

    assert report.timing.correlation_min > NULL_MIN_CORRELATION, outcome.summary()
    worst = max(abs(value) for value in report.spectral.band_delta_db.values())
    assert worst <= NULL_MAX_BAND_DEVIATION_DB, f"{name} moved {worst:.2f} dB"
    assert report.timing.offset_drift_ms <= NULL_MAX_DRIFT_MS
    assert report.stereo.channels_preserved
    assert report.processed.clipped_samples == 0


def test_a_clean_stem_is_not_reported_as_enhanced():
    """Doing nothing is the right answer here, and has to be reported as such."""
    outcome = _run(corpus.clean_full_band(14.0), stem_name="mix")

    assert not outcome.is_enhanced
    assert not outcome.fell_back
    assert all(not step.ran for step in outcome.steps.values())


def test_the_null_test_holds_at_a_different_sample_rate():
    from suno_restore.audio_io import resample

    clean = resample(corpus.clean_full_band(14.0), SR, 44100)
    outcome = chain.restore_stem(clean, 44100, **models.MEASURED)
    report = quality.compare(clean, outcome.audio, 44100, outcome.sr)

    assert outcome.sr == 44100, "the source rate must be followed"
    assert report.timing.correlation_min > NULL_MIN_CORRELATION


def test_the_null_test_holds_for_mono():
    clean = corpus.clean_full_band(14.0).mean(axis=1).astype(np.float32)
    outcome = _run(clean)
    report = quality.compare(clean, outcome.audio, SR, SR)

    assert report.timing.correlation_min > NULL_MIN_CORRELATION
    assert outcome.audio.ndim == 1


# --- Phase 9: the reference guitar --------------------------------------


@pytest.fixture(scope="module")
def guitar_run() -> tuple[np.ndarray, chain.RestoreOutcome]:
    guitar = corpus.solo_guitar(14.0)
    return guitar, chain.restore_stem(
        guitar, SR, **models.MEASURED, stem_name="guitar", available_stems=["guitar"]
    )


def test_tempo_is_skipped_with_a_recorded_reason(guitar_run):
    _, outcome = guitar_run
    step = outcome.steps["tempo"]

    assert not step.ran
    assert step.decision.reason
    assert outcome.engine_config["steps"]["tempo"]["run"] is False


def test_bandwidth_is_skipped_with_a_recorded_reason(guitar_run):
    _, outcome = guitar_run
    step = outcome.steps["bandwidth"]

    assert not step.ran
    assert "cliff" in step.decision.reason or "rolls off" in step.decision.reason
    assert outcome.engine_config["steps"]["bandwidth"]["run"] is False


def test_denoise_runs_but_gently(guitar_run):
    _, outcome = guitar_run
    step = outcome.steps["denoise"]

    assert step.ran
    assert step.decision.confidence < 0.5
    assert step.blend_summary["effective_wet"] < 0.25


def test_presence_survives(guitar_run):
    """The critical Phase 9 target: 5-10kHz within +/-1dB.

    Measured at -3.43dB and -2.62dB before this work, from a denoise stage
    running at full strength across the whole spectrum.
    """
    guitar, outcome = guitar_run
    report = quality.compare(guitar, outcome.audio, SR, SR)

    for band in ("5k_8k", "8k_10k"):
        delta = report.spectral.band_delta_db[band]
        assert abs(delta) <= PRESENCE_TOLERANCE_DB, f"{band} moved {delta:+.2f} dB"


def test_timing_is_untouched(guitar_run):
    """Measured at -420ms duration and -375ms accumulated drift before."""
    guitar, outcome = guitar_run
    report = quality.compare(guitar, outcome.audio, SR, SR)

    assert abs(report.timing.duration_delta_ms) < 5.0
    assert report.timing.offset_drift_ms < 5.0
    assert report.timing.correlation_min > 0.98


def test_stereo_survives(guitar_run):
    """Measured as 2 channels in, 1 out, with the side channel lost."""
    guitar, outcome = guitar_run
    report = quality.compare(guitar, outcome.audio, SR, SR)

    assert report.stereo.channels_preserved
    assert report.stereo.side_to_mid_delta_db > -3.0


def test_decay_and_dynamics_survive(guitar_run):
    """Measured as a 15.75dB downward expansion eating note tails."""
    guitar, outcome = guitar_run
    report = quality.compare(guitar, outcome.audio, SR, SR)

    assert report.dynamic.expansion_spread_db <= 6.0
    assert report.transient.decay_delta_db > -1.0
    assert report.transient.attack_slope_delta_db > -0.5


def test_the_run_explains_itself(guitar_run):
    import json

    _, outcome = guitar_run
    payload = json.loads(json.dumps(outcome.engine_config, default=float))

    assert set(payload["steps"]) == {"tempo", "denoise", "bandwidth"}
    for step in payload["steps"].values():
        assert step["reason"]
        assert "measurements" in step
    assert "verification" in payload
    assert payload["is_enhanced"] is outcome.is_enhanced


# --- Damaged material is still improved ---------------------------------


def test_a_genuinely_noisy_stem_is_cleaned_up():
    """The other half of the bargain: conservative must not mean inert."""
    guitar = corpus.solo_guitar(14.0)
    noisy = corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0)

    outcome = chain.restore_stem(noisy, SR, **models.WELL_BEHAVED, stem_name="guitar")
    assert outcome.steps["denoise"].ran

    report = quality.compare(noisy, outcome.audio, SR, SR)
    assert report.spectral.band_delta_db["10k_16k"] < -0.5, "hiss should come down"
    assert report.timing.correlation_min > 0.9


def test_a_bandwidth_limited_stem_gets_its_high_band_considered():
    limited = corpus.band_limit(corpus.full_mix(14.0), SR, cutoff_hz=8000.0)
    outcome = chain.restore_stem(limited, SR, **models.WELL_BEHAVED, stem_name="mix")

    assert outcome.steps["bandwidth"].ran
    assert outcome.steps["bandwidth"].decision.measurements["cliff_hz"] >= 8000


def test_a_harmful_result_falls_back_to_dry():
    """When a step gets through its gate, runs, and damages the stem anyway.

    Routed through denoise rather than bandwidth: bandwidth has a second check
    on the model's own output, so a badly-behaved bandwidth model is refused
    before it is ever applied. Denoise has no such check -- a denoiser's output
    is legitimately unlike its input -- so it is the step that needs the
    whole-stem backstop, and this is that backstop.
    """
    guitar = corpus.solo_guitar(14.0)

    def wrecking_ball(audio: np.ndarray, sr: int) -> np.ndarray:
        return corpus.damage_warp(audio, sr, start_s=2.0, end_s=10.0, drift_ms=-300.0)

    outcome = chain.restore_stem(
        guitar, SR, denoise_model=wrecking_ball, settings=Settings(),
    )

    assert outcome.steps["denoise"].ran
    assert outcome.fell_back
    assert not outcome.is_enhanced
    assert outcome.verdict.fallback_reason
    assert np.array_equal(outcome.audio, guitar)


def test_an_inventing_model_is_refused_before_it_is_applied():
    """The cheaper path: catch it at the step, not with a whole-stem rollback."""
    limited = corpus.band_limit(corpus.full_mix(14.0), SR, cutoff_hz=8000.0)
    outcome = chain.restore_stem(limited, SR, **models.MEASURED, enable=("bandwidth",))
    step = outcome.steps["bandwidth"]

    assert step.decision.run, "the gate accepts the material -- there is a real cutoff"
    assert not step.ran, "but the model's output is refused"
    assert "unrelated" in step.skip_reason
    assert not outcome.fell_back, "no whole-stem rollback was needed"
    assert np.array_equal(outcome.audio, limited)
