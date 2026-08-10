"""Phase 4: nothing ships that measures worse than what went in.

Each of the original five defects is applied to a clean signal here, and each
has to be caught and rolled back. The dry signal is always available as the
fallback precisely because no stage was ever allowed to overwrite it.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import verify
from suno_restore.config import Settings
from tests import corpus

SR = corpus.SR


@pytest.fixture(scope="module")
def guitar() -> np.ndarray:
    return corpus.solo_guitar(14.0)


def _knee(audio: np.ndarray, percentile: float = 85.0) -> float:
    return float(np.percentile(corpus.frame_level_db(audio, SR), percentile))


# --- Passing -------------------------------------------------------------


def test_an_unchanged_stem_passes_but_is_not_an_enhancement(guitar):
    """Safe and better are different questions, and both get an answer.

    A pipeline optimised only for 'do not damage anything' scores perfectly by
    doing nothing at all, so the verdict has to distinguish the two.
    """
    verdict = verify.verify(guitar, guitar.copy(), SR, SR)

    assert verdict.passed
    assert not verdict.is_enhanced
    assert verdict.failed_checks == []
    assert verdict.preservation_penalty == pytest.approx(0.0, abs=0.01)
    assert verdict.quality_score == pytest.approx(0.0, abs=0.03)


def test_a_small_level_change_is_not_an_enhancement(guitar):
    """Measurement noise near zero must not be reported as improvement."""
    quieter = (guitar * 10 ** (-1.0 / 20)).astype(np.float32)
    verdict = verify.verify(guitar, quieter, SR, SR)

    assert verdict.passed
    assert not verdict.is_enhanced


def test_genuine_noise_removal_is_an_enhancement(guitar):
    """The other side of the coin: real improvement has to register.

    The hiss here sits clear of the protected 100Hz-10kHz range, which is where
    hiss mostly lives and where the denoiser is weighted to act. Removing it is
    both safe and better.
    """
    noisy = corpus.add_hiss(guitar, level_db=-25.0, sr=SR, high_pass_hz=13000.0)
    verdict = verify.verify(noisy, guitar, SR, SR)

    assert verdict.passed
    assert verdict.restoration_gain > 0.1
    assert verdict.is_enhanced


def test_noise_inside_the_protected_bands_cannot_be_removed_freely(guitar):
    """A real tension in the specification, pinned so it is not rediscovered.

    The acceptance gate allows +/-1.5dB across 100Hz-10kHz. Noise that lives
    inside that range therefore cannot be removed in quantity without breaking
    the gate -- removing it *is* a band change, and nothing in the measurement
    can tell "took out hiss" from "took out music".

    The pipeline's answer is to weight the denoiser away from that range
    (0.0 below 2kHz, 0.15 to 5kHz, 0.40 to 10kHz) and, when a stem still fails,
    to fall back to dry and report it honestly rather than ship the change. That
    is the conservative behaviour the plan asks for, but it does mean heavily
    in-band noise is left alone.
    """
    in_band = corpus.add_hiss(guitar, level_db=-25.0)
    verdict = verify.verify(in_band, guitar, SR, SR)

    assert not verdict.passed
    assert "band_balance" in verdict.failed_checks


# --- Each defect is caught ----------------------------------------------


def test_a_time_warp_is_rejected(guitar):
    warped = corpus.damage_warp(guitar, SR, start_s=3.0, end_s=9.0, drift_ms=-300.0)
    verdict = verify.verify(guitar, warped, SR, SR)

    assert not verdict.passed
    assert "duration" in verdict.failed_checks
    assert "timing_drift" in verdict.failed_checks
    assert "correlation" in verdict.failed_checks
    assert verdict.fallback_reason


def test_downward_expansion_is_rejected(guitar):
    expanded = corpus.damage_expander(guitar, SR, knee_db=_knee(guitar),
                                      max_attenuation_db=15.75)
    verdict = verify.verify(guitar, expanded, SR, SR)

    assert not verdict.passed
    assert "dynamic_expansion" in verdict.failed_checks


def test_presence_loss_is_rejected(guitar):
    """3.4dB out of 5-10kHz is over the +/-1.5dB budget."""
    dulled = corpus.damage_presence(guitar, SR, 5000, 10000, gain_db=-3.43)
    verdict = verify.verify(guitar, dulled, SR, SR)

    assert not verdict.passed
    assert "band_balance" in verdict.failed_checks


def test_a_channel_drop_is_rejected(guitar):
    mono = guitar.mean(axis=1).astype(np.float32)
    verdict = verify.verify(guitar, mono, SR, SR)

    assert not verdict.passed
    assert "channels" in verdict.failed_checks


def test_a_stereo_fold_is_rejected_even_with_two_channels(guitar):
    """The channel count is unchanged, so only the image test can catch this."""
    collapsed = corpus.damage_collapse_stereo(guitar)
    verdict = verify.verify(guitar, collapsed, SR, SR)

    assert not verdict.passed
    assert "stereo_image" in verdict.failed_checks
    assert "channels" not in verdict.failed_checks


def test_new_clipping_is_rejected(guitar):
    clipped = np.clip(guitar * 6.0, -1.0, 1.0).astype(np.float32)
    verdict = verify.verify(guitar, clipped, SR, SR)

    assert not verdict.passed
    assert "clipping" in verdict.failed_checks


def test_transient_softening_is_rejected(guitar):
    """Smearing the attacks costs articulation even when nothing else moves."""
    from scipy.signal import butter, sosfilt

    sos = butter(2, 6000 / (SR / 2), btype="low", output="sos")
    smeared = sosfilt(sos, guitar, axis=0).astype(np.float32)
    verdict = verify.verify(guitar, smeared, SR, SR)

    assert not verdict.passed


def test_invented_high_band_is_safe_but_not_an_enhancement(guitar):
    """The subtler case, and the one the plan cares about most.

    Adding an unrelated high band damages nothing measurable -- it passes every
    preservation gate -- so a damage-only verdict would wave it through. The
    artifact penalty is what stops it counting as an improvement.
    """
    limited = corpus.band_limit(guitar, SR, cutoff_hz=9000.0)
    faked = corpus.damage_synthetic_hf(limited, SR, level_db=-40.0)
    verdict = verify.verify(limited, faked, SR, SR)

    assert verdict.passed, "it does not damage anything"
    assert verdict.artifact_penalty > 0.2
    assert not verdict.is_enhanced, "but it is not an improvement either"


# --- Rollback ------------------------------------------------------------


def test_a_failed_stem_is_replaced_by_its_dry_input(guitar):
    warped = corpus.damage_warp(guitar, SR, start_s=3.0, end_s=9.0, drift_ms=-300.0)
    audio, sr, verdict = verify.accept_or_rollback(guitar, warped, SR, SR)

    assert np.array_equal(audio, guitar)
    assert sr == SR
    assert not verdict.passed
    assert not verdict.is_enhanced
    assert verdict.fallback_reason


def test_a_passing_stem_is_kept(guitar):
    cleaned = (guitar * 10 ** (-0.5 / 20)).astype(np.float32)
    audio, sr, verdict = verify.accept_or_rollback(guitar, cleaned, SR, SR)

    assert np.array_equal(audio, cleaned)
    assert verdict.passed


def test_rollback_restores_the_dry_sample_rate(guitar):
    """A discarded result must not leave its sample rate behind."""
    from suno_restore.audio_io import resample

    warped = corpus.damage_warp(resample(guitar, SR, 44100), 44100, 2.0, 8.0, -300.0)
    audio, sr, verdict = verify.accept_or_rollback(guitar, warped, SR, 44100)

    assert not verdict.passed
    assert sr == SR
    assert audio.shape == guitar.shape


def test_a_failed_verdict_never_reports_an_enhancement(guitar):
    """The one thing the report must never do is lie about what shipped."""
    for damaged in (
        corpus.damage_warp(guitar, SR, 3.0, 9.0, -300.0),
        corpus.damage_expander(guitar, SR, knee_db=_knee(guitar), max_attenuation_db=15.0),
        corpus.damage_presence(guitar, SR, 5000, 10000, -4.0),
        guitar.mean(axis=1).astype(np.float32),
    ):
        verdict = verify.verify(guitar, damaged, SR, SR)
        assert not (verdict.passed and not verdict.is_enhanced and verdict.fallback_reason == "")
        if not verdict.passed:
            assert not verdict.is_enhanced
            assert verdict.fallback_reason


# --- Reporting -----------------------------------------------------------


def test_the_verdict_records_every_check(guitar):
    verdict = verify.verify(guitar, guitar.copy(), SR, SR)
    names = {check.name for check in verdict.checks}

    assert names == {
        "duration", "timing_drift", "correlation", "band_balance", "attack_slope",
        "dynamic_expansion", "channels", "stereo_image", "clipping",
    }
    for check in verdict.checks:
        assert check.detail


def test_the_verdict_serialises_for_engine_config(guitar):
    import json

    verdict = verify.verify(guitar, corpus.damage_collapse_stereo(guitar), SR, SR)
    payload = json.loads(json.dumps(verdict.to_dict(), default=float))

    assert payload["passed"] is False
    assert payload["fallback_reason"]
    assert isinstance(payload["checks"], list)
    assert "metrics" in payload
    assert isinstance(verdict.summary(), str)


def test_verification_can_be_disabled(guitar):
    """Off is a supported configuration, and says so rather than pretending."""
    warped = corpus.damage_warp(guitar, SR, 3.0, 9.0, -300.0)
    settings = Settings(verify_enabled=False)
    audio, _, verdict = verify.accept_or_rollback(guitar, warped, SR, SR, settings=settings)

    assert verdict.passed
    assert np.array_equal(audio, warped)
    assert "disabled" in str(verdict.metrics)


def test_limits_come_from_settings(guitar):
    """A tighter budget has to reject something the default budget accepts."""
    dulled = corpus.damage_presence(guitar, SR, 5000, 10000, gain_db=-1.0)

    assert verify.verify(guitar, dulled, SR, SR).passed
    strict = Settings(verify_max_band_delta_db=0.5)
    assert not verify.verify(guitar, dulled, SR, SR, settings=strict).passed
