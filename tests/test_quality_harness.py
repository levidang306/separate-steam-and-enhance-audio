"""Phase 0 acceptance: the harness must recover damage it did not know about.

Every expectation here is *computed* from the parameters handed to the damage
function, never pasted from a previous run. A harness that only agrees with
itself would pass a regression suite while measuring the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import quality
from tests import corpus

SR = corpus.SR


@pytest.fixture(scope="module")
def guitar() -> np.ndarray:
    return corpus.solo_guitar(seconds=14.0, sr=SR)


# --- Identity ------------------------------------------------------------


def test_identity_reports_no_change(guitar):
    report = quality.compare(guitar, guitar, SR, SR)

    assert report.timing.global_offset_ms == pytest.approx(0.0, abs=0.05)
    assert report.timing.offset_drift_ms == pytest.approx(0.0, abs=0.05)
    assert report.timing.correlation_min > 0.999
    assert report.timing.fine_correlation_min > 0.999
    assert report.spectral.program_gain_db == pytest.approx(0.0, abs=0.01)
    for delta in report.spectral.band_delta_db.values():
        assert abs(delta) < 0.05
    assert report.dynamic.expansion_spread_db < 0.05
    assert report.stereo.channels_preserved
    assert report.transient.attack_slope_delta_db == pytest.approx(0.0, abs=0.01)


def test_pure_gain_is_reported_as_gain_not_as_spectral_change(guitar):
    """A level change must land in program_gain_db and nowhere else.

    This is the property that makes every band number readable: without it,
    turning a stem down 2dB would look like losing 2dB of presence.
    """
    quieter = (guitar * 10 ** (-2.0 / 20)).astype(np.float32)
    report = quality.compare(guitar, quieter, SR, SR)

    assert report.spectral.program_gain_db == pytest.approx(-2.0, abs=0.05)
    for name, delta in report.spectral.band_delta_db.items():
        assert abs(delta) < 0.1, f"band {name} moved {delta:+.2f} dB under a pure gain change"


# --- Offset detection ----------------------------------------------------


def _delay(audio: np.ndarray, samples: int) -> np.ndarray:
    """Delay by `samples` (positive) or advance it (negative)."""
    if samples >= 0:
        return np.concatenate([np.zeros((samples, audio.shape[1]), dtype=np.float32), audio])
    return audio[-samples:]


def test_positive_offset_means_the_processed_signal_is_late(guitar):
    """Pin the sign convention: everything downstream assumes it."""
    late = _delay(guitar, int(0.100 * SR))
    assert quality.compare(guitar, late, SR, SR).timing.global_offset_ms > 0

    early = _delay(guitar, -int(0.100 * SR))
    assert quality.compare(guitar, early, SR, SR).timing.global_offset_ms < 0


@pytest.mark.parametrize("shift_ms", [-80.0, -12.0, 7.5, 45.0])
def test_fine_offset_is_recovered_to_the_sample(guitar, shift_ms):
    moved = _delay(guitar, int(round(shift_ms / 1000 * SR)))
    report = quality.compare(guitar, moved, SR, SR)

    assert report.timing.global_offset_ms == pytest.approx(shift_ms, abs=0.5)


@pytest.mark.parametrize("shift_s", [1.5, -3.0, 9.0])
def test_coarse_stage_finds_offsets_far_outside_the_fine_window(guitar, shift_s):
    """The reason the search is two-stage.

    A single +/-120ms search cannot see these at all; it would lock onto a
    spurious local peak and report a confident wrong answer.
    """
    moved = _delay(guitar, int(round(shift_s * SR)))
    report = quality.compare(guitar, moved, SR, SR)

    assert report.timing.global_offset_ms == pytest.approx(shift_s * 1000, abs=2.0)
    assert abs(report.timing.coarse_offset_ms) > 100.0


def test_single_stage_search_would_have_failed(guitar):
    """Pin the failure mode the plan warns about, so nobody 'simplifies' it back."""
    shift = int(round(4.0 * SR))
    moved = np.concatenate([np.zeros((shift, 2), dtype=np.float32), guitar])

    crippled = quality.estimate_offset(guitar, moved, SR, coarse_s=0.150, fine_s=0.150)
    honest = quality.estimate_offset(guitar, moved, SR)

    assert abs(crippled.total_samples - shift) > SR  # off by more than a second
    assert abs(honest.total_samples - shift) < SR * 0.01


# --- Defect #1: tempo warp ----------------------------------------------


def test_warp_is_measured_as_drift_not_as_a_constant_shift(guitar):
    """Reproduces defect #1 on known ground truth.

    The signature that matters is not the shortening -- it is that the offset
    *accumulates* over a region and then stays put, while short-window
    correlation collapses inside that region and recovers outside it.
    """
    drift_ms = -300.0
    warped = corpus.damage_warp(guitar, SR, start_s=3.0, end_s=9.0, drift_ms=drift_ms)
    report = quality.compare(guitar, warped, SR, SR)

    assert report.timing.duration_delta_ms == pytest.approx(drift_ms, abs=5.0)
    assert report.timing.offset_drift_ms == pytest.approx(abs(drift_ms), rel=0.25)

    times = np.array(report.timing.local_offset_times_s)
    offsets = np.array(report.timing.local_offset_ms)
    before = offsets[times < 2.0]
    after = offsets[times > 9.5]
    assert np.abs(before).max() < 30.0, "drift appeared before the warped region"
    assert np.median(after) == pytest.approx(drift_ms, rel=0.3)

    # The tail stays at a constant offset once the warp ends.
    assert after.max() - after.min() < 40.0


def test_warp_damages_short_window_correlation_even_when_realigned(guitar):
    """Time-stretching is not a shift; re-aligning does not undo it."""
    warped = corpus.damage_warp(guitar, SR, start_s=3.0, end_s=9.0, drift_ms=-300.0)
    report = quality.compare(guitar, warped, SR, SR)

    assert report.timing.fine_correlation_min < 0.9
    assert report.timing.correlation_min < report.timing.correlation_median


def test_clean_signal_shows_no_drift(guitar):
    report = quality.compare(guitar, guitar.copy(), SR, SR)
    assert report.timing.offset_drift_ms < 1.0


# --- Defect #2: downward expansion --------------------------------------


def _knee_inside(audio: np.ndarray, percentile: float = 35.0) -> float:
    """A knee that actually engages on this material."""
    return float(np.percentile(corpus.frame_level_db(audio, SR), percentile))


def test_expander_shows_as_level_dependent_gain(guitar):
    """Reproduces defect #2: quiet frames pulled down, loud frames barely touched."""
    attenuation = 15.75
    expanded = corpus.damage_expander(guitar, SR, knee_db=_knee_inside(guitar),
                                      max_attenuation_db=attenuation)
    report = quality.compare(guitar, expanded, SR, SR)

    assert report.dynamic.quiet_gain_db < -6.0
    assert report.dynamic.loud_gain_db > -3.0
    assert report.dynamic.quiet_gain_db < report.dynamic.loud_gain_db - 4.0
    assert report.dynamic.expansion_spread_db > 4.0

    # The curve must be ordered: louder input, less attenuation.
    keys = sorted(report.dynamic.gain_curve_db, key=lambda k: float(k.split("_")[0]))
    values = [report.dynamic.gain_curve_db[k] for k in keys]
    assert values[0] < values[-1]


def test_expander_suppresses_note_decay(guitar):
    """The audible consequence: tails die early.

    The knee sits high enough that a note's 400ms tail falls below it while its
    attack does not -- which is exactly the condition under which an expander
    eats decay, and reproduces the ~-1.8dB the plan reports at 400ms.
    """
    expanded = corpus.damage_expander(guitar, SR, knee_db=_knee_inside(guitar, 85.0),
                                      max_attenuation_db=15.0)
    report = quality.compare(guitar, expanded, SR, SR)

    assert report.transient.decay_delta_db < -0.5
    # The attack itself survives, so this is decay loss and not an overall dip.
    assert report.transient.attack_slope_delta_db > -0.5


def test_clean_copy_shows_flat_gain_curve(guitar):
    report = quality.compare(guitar, guitar.copy(), SR, SR)
    assert report.dynamic.expansion_spread_db < 0.05


# --- Defect #3: presence loss -------------------------------------------


@pytest.mark.parametrize("gain_db", [-3.43, -2.0, -6.0])
def test_presence_cut_is_recovered_in_the_right_bands(guitar, gain_db):
    """Reproduces defect #3, and confirms the cut lands only where it was applied."""
    damaged = corpus.damage_presence(guitar, SR, low_hz=5000, high_hz=10000, gain_db=gain_db)
    report = quality.compare(guitar, damaged, SR, SR)

    assert report.spectral.band_delta_db["5k_8k"] == pytest.approx(gain_db, abs=0.6)
    assert report.spectral.band_delta_db["8k_10k"] == pytest.approx(gain_db, abs=0.6)
    for name in ("250_500", "500_1k", "1k_2k", "2k_5k"):
        assert abs(report.spectral.band_delta_db[name]) < 0.5, f"{name} should be untouched"


def test_band_delta_is_immune_to_sparse_loud_events(guitar):
    """Why the metric is a gated median rather than an energy sum.

    One very loud clean burst must not be able to outvote a cut that runs for
    the whole track -- an energy-sum metric would report almost nothing here.
    """
    damaged = corpus.damage_presence(guitar, SR, 5000, 10000, gain_db=-4.0)
    burst = np.zeros_like(guitar)
    burst[: SR // 2] = guitar[: SR // 2] * 8.0
    original = (guitar + burst).astype(np.float32)
    processed = (damaged + burst).astype(np.float32)

    report = quality.compare(original, processed, SR, SR)
    assert report.spectral.band_delta_db["5k_8k"] < -2.0


# --- Defect #4: synthetic high band -------------------------------------


def test_invented_high_band_shows_as_low_coherence_and_periodic_level():
    """Reproduces defect #4: HF unrelated to the source, pulsing on the chunk stride.

    A longer excerpt than the other tests use: a 9-second period needs several
    cycles before its period can be pinned down at all.
    """
    long_guitar = corpus.solo_guitar(seconds=45.0, sr=SR)
    limited = corpus.band_limit(long_guitar, SR, cutoff_hz=9000.0)
    faked = corpus.damage_synthetic_hf(limited, SR, level_db=-55.0, period_s=9.0)

    report = quality.compare(limited, faked, SR, SR)
    assert report.artifacts.hf_coherence < 0.35
    assert report.artifacts.hf_modulation_period_s == pytest.approx(9.0, rel=0.25)
    assert report.artifacts.hf_modulation_depth_db > 0.4


def test_real_high_band_is_coherent(guitar):
    """The control: genuine shared content must read as coherent."""
    report = quality.compare(guitar, guitar.copy(), SR, SR)
    assert report.artifacts.hf_coherence > 0.9


# --- Defect #5: stereo collapse -----------------------------------------


def test_stereo_collapse_is_detected_even_with_two_channels(guitar):
    """A fold to mono that keeps the channel count still has to be caught."""
    collapsed = corpus.damage_collapse_stereo(guitar)
    report = quality.compare(guitar, collapsed, SR, SR)

    assert report.stereo.channels_preserved  # two channels out -- count alone is not enough
    assert report.stereo.side_to_mid_db_processed < -60.0
    assert report.stereo.side_to_mid_delta_db < -20.0
    assert report.stereo.lr_correlation_processed > 0.999


def test_channel_count_drop_is_detected(guitar):
    mono = guitar.mean(axis=1).astype(np.float32)
    report = quality.compare(guitar, mono, SR, SR)

    assert not report.stereo.channels_preserved
    assert report.stereo.channels_original == 2
    assert report.stereo.channels_processed == 1


# --- Loudness ------------------------------------------------------------


def test_lufs_tracks_a_known_gain_change(guitar):
    quieter = (guitar * 10 ** (-6.0 / 20)).astype(np.float32)
    before = quality.integrated_lufs(guitar, SR)
    after = quality.integrated_lufs(quieter, SR)

    assert after - before == pytest.approx(-6.0, abs=0.1)


def test_lufs_is_in_a_plausible_range(guitar):
    value = quality.integrated_lufs(guitar, SR)
    assert -40.0 < value < 0.0


def test_lra_is_invariant_to_gain(guitar):
    """Loudness range describes spread, so a fader move must not change it."""
    quieter = (guitar * 10 ** (-6.0 / 20)).astype(np.float32)
    assert quality.loudness_range(quieter, SR) == pytest.approx(
        quality.loudness_range(guitar, SR), abs=0.15
    )


def test_lra_grows_when_dynamics_are_expanded(guitar):
    expanded = corpus.damage_expander(guitar, SR, knee_db=-60.0, max_attenuation_db=12.0)
    assert quality.loudness_range(expanded, SR) > quality.loudness_range(guitar, SR)


# --- Clipping and rate ---------------------------------------------------


def test_clipping_is_counted(guitar):
    clipped = np.clip(guitar * 8.0, -1.0, 1.0).astype(np.float32)
    report = quality.compare(guitar, clipped, SR, SR)

    assert report.original.clipped_samples == 0
    assert report.processed.clipped_samples > 0


def test_sample_rate_change_does_not_leak_into_spectral_metrics(guitar):
    """Comparing 44.1k output against 48k input must not read as a spectral change."""
    from suno_restore.audio_io import resample

    resampled = resample(guitar, SR, 44100)
    report = quality.compare(guitar, resampled, SR, 44100)

    assert report.processed.sample_rate == 44100
    assert report.original.sample_rate == SR
    for name in ("250_500", "500_1k", "1k_2k", "2k_5k", "5k_8k"):
        assert abs(report.spectral.band_delta_db[name]) < 0.5


# --- Report shape --------------------------------------------------------


def test_report_serialises(guitar):
    import json

    report = quality.compare(guitar, corpus.add_hiss(guitar), SR, SR)
    payload = json.dumps(report.to_dict(), default=float)
    assert "band_delta_db" in payload
    assert isinstance(report.summary(), str)
    assert "duration" in report.summary()
