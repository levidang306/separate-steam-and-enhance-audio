"""Phase 2: each step has to justify itself before it runs.

The decisions that matter most are the *skips*. A test suite that only checked
the gate can say yes would pass while the pipeline went on damaging everything
it touched, which is the failure this layer exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import gate
from suno_restore.config import Settings
from tests import corpus

SR = corpus.SR

# Tempo is off by default and must stay that way; these settings force it on so
# the conditions underneath the switch can be tested at all.
TEMPO_ON = Settings(tempo_enabled=True)


@pytest.fixture(scope="module")
def guitar() -> np.ndarray:
    return corpus.solo_guitar(14.0)


@pytest.fixture(scope="module")
def mix() -> np.ndarray:
    return corpus.full_mix(14.0)


# --- The default posture -------------------------------------------------


def test_tempo_is_disabled_by_default(guitar):
    decision = gate.decide_tempo(guitar, SR)
    assert not decision.run
    assert "disabled by default" in decision.reason


def test_default_settings_are_the_cautious_ones():
    settings = Settings()
    assert settings.tempo_enabled is False
    assert settings.tempo_allow_on_separated is False
    assert settings.denoise_wet < 1.0
    assert settings.bandwidth_wet < 1.0
    assert settings.verify_enabled is True
    assert settings.output_sample_rate == 0


def test_every_decision_carries_its_evidence(guitar):
    decisions = gate.decide_all(guitar, SR, stem_name="guitar")

    assert set(decisions) == {"tempo", "denoise", "bandwidth"}
    for name, decision in decisions.items():
        assert isinstance(decision.run, bool), name
        assert 0.0 <= decision.confidence <= 1.0, name
        assert decision.reason, name
        assert isinstance(decision.measurements, dict), name


def test_decisions_serialise_for_engine_config(guitar):
    import json

    config = gate.decisions_to_config(gate.decide_all(guitar, SR))
    payload = json.loads(json.dumps(config, default=float))

    assert payload["tempo"]["run"] is False
    assert "reason" in payload["tempo"]
    assert "measurements" in payload["tempo"]


# --- Tempo reference selection ------------------------------------------


def test_guitar_and_piano_are_no_longer_tempo_references():
    """The specific mechanism that warped the reference guitar.

    The old priority list ended in ("guitar", "piano"), so a solo guitar became
    its own tempo reference and was warped to a grid derived from its own uneven
    picking.
    """
    assert "guitar" not in gate.REFERENCE_PRIORITY
    assert "piano" not in gate.REFERENCE_PRIORITY
    assert gate.rank_reference_stems(["guitar", "piano"]) == []


def test_percussive_stems_are_still_preferred():
    ranked = gate.rank_reference_stems(["vocals", "drums", "bass"])
    assert ranked[0] == "drums"
    assert "bass" in ranked
    assert "vocals" not in ranked


def test_no_percussive_stem_means_skip(guitar):
    decision = gate.decide_tempo(
        guitar, SR, available_stems=["guitar", "vocals"], settings=TEMPO_ON
    )
    assert not decision.run
    assert "percussive reference" in decision.reason


def test_separated_stems_are_refused_by_default(mix):
    decision = gate.decide_tempo(mix, SR, is_separated=True, settings=TEMPO_ON)
    assert not decision.run
    assert "separated" in decision.reason


def test_separated_stems_can_be_allowed_explicitly(mix):
    settings = Settings(tempo_enabled=True, tempo_allow_on_separated=True)
    decision = gate.decide_tempo(mix, SR, is_separated=True, settings=settings)
    assert "separated" not in decision.reason


# --- Tempo: material ----------------------------------------------------


@pytest.mark.parametrize("name", ["solo_guitar", "solo_piano", "vocal"])
def test_melodic_material_is_never_tempo_corrected(name):
    """The Phase 9 target for the reference guitar, and its neighbours."""
    audio = corpus.CLEAN_CLASSES[name](14.0)
    decision = gate.decide_tempo(audio, SR, settings=TEMPO_ON)

    assert not decision.run
    assert "percussive" in decision.reason
    assert decision.measurements["percussive_score"] < 0.30


@pytest.mark.parametrize("name", ["drums", "full_mix"])
def test_percussive_material_clears_the_percussive_test(name):
    audio = corpus.CLEAN_CLASSES[name](14.0)
    decision = gate.decide_tempo(audio, SR, settings=TEMPO_ON)

    assert decision.measurements["percussive_score"] >= 0.30
    assert "percussive" not in decision.reason


def test_steady_material_with_no_drift_is_left_alone(mix):
    """Nothing to fix is a reason to skip, not a reason to process."""
    decision = gate.decide_tempo(mix, SR, settings=TEMPO_ON)
    assert not decision.run


def test_a_sparse_grid_is_refused():
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 12.0, 12.5, 13.0])
    grid = gate.measure_grid(beats, duration_s=14.0)

    assert grid["coverage"] < 0.80


def test_a_full_grid_is_accepted():
    beats = np.arange(0.0, 14.0, 0.5)
    grid = gate.measure_grid(beats, duration_s=14.0)

    assert grid["coverage"] > 0.90
    assert grid["bpm_std_pct"] < 2.0


def test_an_unsteady_grid_reports_a_wide_spread():
    rng = np.random.default_rng(3)
    beats = np.cumsum(rng.uniform(0.35, 0.75, 40))
    grid = gate.measure_grid(beats, duration_s=float(beats[-1]))

    assert grid["bpm_std_pct"] > 2.0


def test_the_plan_parameters_leave_an_empty_correction_window():
    """MIN_CORRECTION=0.02 and MAX_RATE_DEVIATION=(0.98, 1.02) are the same 2%.

    A segment must deviate by at least 2% to be worth correcting, and more than
    2% means it is a mis-detected beat. Nothing can satisfy both. The gate says
    so explicitly rather than reporting a silent zero, because a step that can
    never run for a configuration reason is a very different thing from one that
    examined the audio and declined.
    """
    settings = Settings()
    low, high = settings.tempo_rate_bounds
    assert settings.tempo_min_correction >= max(abs(1 - low), abs(high - 1)) - 1e-9

    grid = gate.measure_grid(np.arange(0.0, 14.0, 0.5), 14.0,
                             min_correction=0.02, max_deviation=0.02)
    assert grid["correctable_window_empty"] is True

    widened = gate.measure_grid(np.arange(0.0, 14.0, 0.5), 14.0,
                                min_correction=0.005, max_deviation=0.02)
    assert widened["correctable_window_empty"] is False


def test_a_correction_larger_than_the_drift_cap_is_refused():
    """Even good evidence does not license moving the timeline very far."""
    settings = Settings(tempo_enabled=True, tempo_min_correction=0.005,
                        tempo_max_rate_low=0.95, tempo_max_rate_high=1.05,
                        tempo_max_cumulative_drift_ms=50.0)
    drifting = corpus.damage_warp(corpus.full_mix(20.0), SR,
                                  start_s=3.0, end_s=15.0, drift_ms=-40.0)
    decision = gate.decide_tempo(drifting, SR, settings=settings)

    assert not decision.run
    assert "limit" in decision.reason or "move the timeline" in decision.reason


# --- Bandwidth ----------------------------------------------------------


def test_guitar_bandwidth_is_skipped(guitar):
    """The Phase 9 target.

    The reference guitar's content stops well below 8kHz and there is no cliff
    to restore, so synthesising a high band would be invention.
    """
    decision = gate.decide_bandwidth(guitar, SR)

    assert not decision.run
    assert decision.measurements["rolloff85_hz"] < 8000


@pytest.mark.parametrize("name", list(corpus.CLEAN_CLASSES))
def test_undamaged_material_never_gets_a_synthetic_high_band(name):
    audio = corpus.CLEAN_CLASSES[name](14.0)
    assert not gate.decide_bandwidth(audio, SR).run


def test_a_real_cutoff_is_accepted(guitar):
    limited = corpus.band_limit(guitar, SR, cutoff_hz=8000.0)
    decision = gate.decide_bandwidth(limited, SR)

    assert decision.run
    assert 8000 <= decision.measurements["cliff_hz"] <= 20000
    assert decision.confidence > 0


def test_codec_damage_is_accepted(mix):
    decision = gate.decide_bandwidth(corpus.codec_degrade(mix, SR), SR)
    assert decision.run


def test_a_low_cutoff_is_tonal_balance_not_a_codec(guitar):
    """A bass stem has no treble to begin with; that is not damage."""
    muffled = corpus.band_limit(guitar, SR, cutoff_hz=3000.0)
    decision = gate.decide_bandwidth(muffled, SR)

    assert not decision.run


def test_bandwidth_can_be_disabled(guitar):
    limited = corpus.band_limit(guitar, SR, cutoff_hz=8000.0)
    decision = gate.decide_bandwidth(limited, SR, settings=Settings(bandwidth_enabled=False))
    assert not decision.run


# --- Denoise ------------------------------------------------------------


def test_clean_material_is_not_denoised():
    """The null test, decided before any audio is touched."""
    decision = gate.decide_denoise(corpus.clean_full_band(14.0), SR)
    assert not decision.run


def test_a_very_quiet_room_is_not_worth_denoising():
    quiet = corpus.solo_guitar(14.0, room_db=-95.0)
    assert not gate.decide_denoise(quiet, SR).run


def test_guitar_is_denoised_gently(guitar):
    """The Phase 9 target: APPLY, at reduced strength.

    Confidence is what makes "reduced" concrete -- downstream it scales the wet
    contribution, so a modest amount of hiss produces a modest amount of
    processing rather than a full-strength pass.
    """
    decision = gate.decide_denoise(guitar, SR)

    assert decision.run
    assert 0.0 < decision.confidence < 0.5


def test_confidence_rises_with_the_amount_of_noise(guitar):
    ladder = [
        gate.decide_denoise(corpus.add_hiss(guitar, level_db=level), SR).confidence
        for level in (-55.0, -45.0, -35.0, -25.0)
    ]
    assert ladder == sorted(ladder), f"confidence should increase with noise: {ladder}"
    assert ladder[-1] > ladder[0] + 0.2


def test_natural_decay_is_not_mistaken_for_noise():
    """A ringing tail is peaky; hiss is flat. The gate has to tell them apart.

    Without this the denoiser would be pointed at note decay, which is exactly
    how the reference guitar lost its tails.
    """
    from suno_restore.gate import measure_noise

    ringing = corpus.solo_piano(14.0, room_db=-100.0)
    hissy = corpus.add_hiss(corpus.solo_piano(14.0, room_db=-100.0), level_db=-35.0)

    assert measure_noise(ringing, SR)["noise_flatness"] < \
        measure_noise(hissy, SR)["noise_flatness"]


def test_denoise_can_be_disabled(guitar):
    decision = gate.decide_denoise(guitar, SR, settings=Settings(denoise_enabled=False))
    assert not decision.run


def test_silence_is_skipped():
    decision = gate.decide_denoise(np.zeros((SR * 2, 2), dtype=np.float32), SR)
    assert not decision.run


# --- The reference case end to end --------------------------------------


def test_reference_guitar_decisions_match_the_plan(guitar):
    """Phase 9: tempo SKIP, bandwidth SKIP, denoise APPLY at reduced strength."""
    decisions = gate.decide_all(guitar, SR, stem_name="guitar", available_stems=["guitar"])

    assert decisions["tempo"].run is False
    assert decisions["bandwidth"].run is False
    assert decisions["denoise"].run is True
    assert decisions["denoise"].confidence < 0.5

    # Each skip has to say why, in terms someone can check.
    assert decisions["tempo"].reason
    assert decisions["bandwidth"].reason
