"""Phase 5: restoration is aimed at the damage, not sprayed across the stem."""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import damage, quality
from suno_restore.config import Settings
from tests import corpus

SR = corpus.SR


@pytest.fixture(scope="module")
def guitar() -> np.ndarray:
    return corpus.solo_guitar(12.0)


# --- Resolution and shape ------------------------------------------------


def test_the_mask_is_at_the_configured_resolution(guitar):
    mask = damage.noise_damage(guitar, SR)

    assert mask.frame_ms == 20.0
    assert mask.feather_ms == 100.0
    step = float(np.median(np.diff(mask.times)))
    assert step == pytest.approx(0.020, abs=0.001)


def test_the_mask_stays_within_zero_and_one(guitar):
    for mask in (
        damage.noise_damage(guitar, SR),
        damage.bandwidth_damage(corpus.band_limit(guitar, SR, 9000.0), SR, 9000.0),
    ):
        assert mask.mask.min() >= 0.0
        assert mask.mask.max() <= 1.0
        assert np.isfinite(mask.mask).all()


def test_the_mask_summarises_without_carrying_the_array(guitar):
    payload = damage.noise_damage(guitar, SR).to_dict()

    assert set(payload) >= {"kind", "coverage", "frame_ms", "feather_ms", "bands", "frames"}
    assert not any(isinstance(value, np.ndarray) for value in payload.values())


# --- Noise damage --------------------------------------------------------


def test_noisy_material_gets_more_coverage_than_clean(guitar):
    """The mask has to actually distinguish; otherwise it is a constant."""
    clean = damage.noise_damage(corpus.solo_guitar(12.0, room_db=-110.0), SR).coverage
    noisy = damage.noise_damage(corpus.add_hiss(guitar, level_db=-25.0), SR).coverage

    assert noisy > clean


def test_loud_music_is_left_alone(guitar):
    """Under a loud chord the noise is masked; removing it achieves nothing."""
    mask = damage.noise_damage(corpus.add_hiss(guitar, level_db=-30.0), SR)

    import librosa

    hop = int(0.020 * SR)
    mono = guitar.mean(axis=1).astype(np.float32)
    rms = librosa.feature.rms(y=mono, frame_length=damage.N_FFT, hop_length=hop)[0]
    n = min(len(rms), mask.mask.shape[1])
    rms, per_frame = rms[:n], mask.mask[:, :n].mean(axis=0)

    loud = rms >= np.percentile(rms, 85)
    quiet = rms <= np.percentile(rms, 15)
    assert per_frame[loud].mean() < per_frame[quiet].mean()


def test_the_mask_responds_to_the_amount_of_noise(guitar):
    coverage = [
        damage.noise_damage(corpus.add_hiss(guitar, level_db=level), SR).coverage
        for level in (-60.0, -45.0, -30.0)
    ]
    assert coverage == sorted(coverage), f"not monotonic: {coverage}"


# --- Bandwidth damage ----------------------------------------------------


def test_bandwidth_damage_sits_above_the_cliff(guitar):
    """Below the cliff nothing is missing; above it, everything is.

    The average above the cliff is taken over frames that are actually playing.
    The mask deliberately backs off during rests -- there is nothing to
    extrapolate from there -- so a whole-file average understates it.
    """
    limited = corpus.band_limit(guitar, SR, cutoff_hz=9000.0)
    mask = damage.bandwidth_damage(limited, SR, cliff_hz=9000.0)

    high = mask.freqs > 10500
    per_frame = mask.mask[high].mean(axis=0)
    playing = per_frame >= np.percentile(per_frame, 60)

    assert mask.mask[mask.freqs < 7500].mean() < 0.05
    assert per_frame[playing].mean() > 0.5


def test_bandwidth_damage_needs_material_to_extend_from(guitar):
    """Nothing below the cliff means nothing to extrapolate, so no treatment."""
    limited = corpus.band_limit(guitar, SR, cutoff_hz=9000.0)
    with_silence = limited.copy()
    with_silence[: 3 * SR] = 0.0

    mask = damage.bandwidth_damage(with_silence, SR, cliff_hz=9000.0)
    high = mask.freqs > 10500
    silent_frames = mask.times < 2.0
    playing_frames = (mask.times > 4.0) & (mask.times < 10.0)

    assert mask.mask[np.ix_(high, silent_frames)].mean() < 0.2
    assert mask.mask[np.ix_(high, playing_frames)].mean() > 0.4


# --- Feathering ----------------------------------------------------------


def test_feathering_widens_and_softens_a_hard_edge():
    """A 20ms on/off gate is audible as pumping; the ramp is the fix."""
    mask = np.zeros((4, 200), dtype=np.float32)
    mask[:, 100:110] = 1.0

    feathered = damage.feather_mask(mask, frame_ms=20.0, feather_ms=100.0)
    row = feathered[0]

    assert row[104] > 0.9, "the interior of the damaged region stays fully treated"
    assert 0.0 < row[95] < 0.9, "and the edges ramp instead of switching"
    assert 0.0 < row[115] < 0.9
    assert row[:80].max() < 0.05, "far from the damage nothing is treated"


def test_feathering_is_a_no_op_when_disabled():
    mask = np.zeros((2, 50), dtype=np.float32)
    mask[:, 20:25] = 1.0
    assert np.array_equal(damage.feather_mask(mask, 20.0, 0.0), mask)


def test_feather_width_follows_the_setting():
    mask = np.zeros((1, 400), dtype=np.float32)
    mask[:, 200:210] = 1.0

    narrow = damage.feather_mask(mask, 20.0, 40.0)[0]
    wide = damage.feather_mask(mask, 20.0, 200.0)[0]

    assert np.count_nonzero(wide > 0.01) > np.count_nonzero(narrow > 0.01)


# --- Combination and use -------------------------------------------------


def test_combining_masks_takes_the_intersection():
    a = damage.DamageMask(np.ones((4, 10), np.float32), np.zeros(4), np.zeros(10), 20, 100, "a")
    b = damage.DamageMask(np.zeros((4, 10), np.float32), np.zeros(4), np.zeros(10), 20, 100, "b")

    assert damage.combine(a, a).max() == pytest.approx(1.0)
    assert damage.combine(a, b).max() == pytest.approx(0.0)


def test_combining_masks_of_different_shapes():
    a = damage.DamageMask(np.ones((4, 10), np.float32), np.zeros(4), np.zeros(10), 20, 100, "a")
    b = damage.DamageMask(np.ones((8, 25), np.float32), np.zeros(8), np.zeros(25), 20, 100, "b")

    assert damage.combine(a, b).shape == (8, 25)


def test_the_mask_confines_processing_to_the_damaged_region(guitar):
    """The property the whole file exists for.

    Half the stem is given hiss and the other half left clean. With the mask in
    play, the clean half must come out essentially untouched even though the
    step ran over the whole stem.
    """
    from suno_restore import blend

    half = guitar.shape[0] // 2
    noisy = guitar.copy()
    noisy[half:] = corpus.add_hiss(guitar[half:], level_db=-25.0, sr=SR)

    mask = damage.noise_damage(noisy, SR)
    # A stand-in denoiser that is far too aggressive everywhere.
    over_processed = (noisy * 10 ** (-9.0 / 20)).astype(np.float32)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)

    with_mask = blend.blend_spectral(noisy, over_processed, SR, amount=1.0, weights=flat,
                                     mask=mask.mask, max_attenuation_db=0.0).audio
    without_mask = blend.blend_spectral(noisy, over_processed, SR, amount=1.0, weights=flat,
                                        max_attenuation_db=0.0).audio

    clean_masked = quality.compare(noisy[:half], with_mask[:half], SR, SR)
    clean_unmasked = quality.compare(noisy[:half], without_mask[:half], SR, SR)

    assert abs(clean_masked.spectral.program_gain_db) < abs(
        clean_unmasked.spectral.program_gain_db
    )
    assert abs(clean_unmasked.spectral.program_gain_db) > 8.0


def test_settings_drive_the_resolution(guitar):
    settings = Settings(damage_frame_ms=10.0, damage_feather_ms=50.0)
    mask = damage.noise_damage(guitar, SR, settings=settings)

    assert mask.frame_ms == 10.0
    step = float(np.median(np.diff(mask.times)))
    assert step == pytest.approx(0.010, abs=0.001)


def test_short_input_does_not_crash():
    tiny = np.zeros((1000, 2), dtype=np.float32)
    assert damage.noise_damage(tiny, SR).mask.size >= 0
