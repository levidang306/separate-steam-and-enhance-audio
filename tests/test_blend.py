"""Phase 3: a stage proposes, the blend disposes.

The properties worth testing here are the limits, not the arithmetic. Anyone
can verify that 0.5 of A plus 0.5 of B is a mixture; what matters is that a
model returning something destructive cannot destroy anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from suno_restore import blend, quality
from suno_restore.config import Settings
from tests import corpus

SR = corpus.SR


@pytest.fixture(scope="module")
def guitar() -> np.ndarray:
    return corpus.solo_guitar(12.0)


# --- Wet/dry ------------------------------------------------------------


def test_zero_wet_returns_the_dry_signal_untouched(guitar):
    wet = corpus.damage_expander(guitar, SR, knee_db=-20.0, max_attenuation_db=20.0)
    result = blend.blend_spectral(guitar, wet, SR, amount=0.0)

    assert np.array_equal(result.audio, guitar)
    assert result.effective_wet == 0.0


def test_the_dry_signal_is_never_modified_in_place(guitar):
    original = guitar.copy()
    wet = corpus.add_hiss(guitar, -20.0)
    blend.blend_spectral(guitar, wet, SR, amount=1.0)

    assert np.array_equal(guitar, original), "blending mutated its input"


def test_more_wet_means_more_of_the_wet_signal(guitar):
    """The blend has to be monotonic, or 'reduced strength' means nothing."""
    wet = corpus.damage_presence(guitar, SR, 5000, 10000, gain_db=-8.0)
    flat = np.ones(N_BINS := blend.N_FFT // 2 + 1, dtype=np.float32)

    deltas = []
    for amount in (0.2, 0.5, 0.8):
        out = blend.blend_spectral(guitar, wet, SR, amount=amount, weights=flat,
                                   max_attenuation_db=0.0).audio
        report = quality.compare(guitar, out, SR, SR)
        deltas.append(report.spectral.band_delta_db["5k_8k"])

    assert deltas[0] > deltas[1] > deltas[2], f"not monotonic: {deltas}"
    assert N_BINS > 0


def test_full_wet_with_a_flat_weight_reproduces_the_wet_signal(guitar):
    wet = corpus.damage_presence(guitar, SR, 5000, 10000, gain_db=-6.0)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)
    out = blend.blend_spectral(guitar, wet, SR, amount=1.0, weights=flat,
                               max_attenuation_db=0.0).audio

    n = min(len(out), len(wet))
    report = quality.compare(wet[:n], out[:n], SR, SR)
    assert report.timing.correlation_min > 0.98


# --- Frequency weighting ------------------------------------------------


def test_band_weights_match_the_configured_values():
    weights = blend.band_weights(SR, blend.N_FFT)
    import librosa

    freqs = librosa.fft_frequencies(sr=SR, n_fft=blend.N_FFT)

    def at(hz: float) -> float:
        return float(weights[int(np.argmin(np.abs(freqs - hz)))])

    settings = Settings()
    assert at(1000) == pytest.approx(settings.denoise_weight_below_2k, abs=0.02)
    assert at(3162) == pytest.approx(settings.denoise_weight_2k_5k, abs=0.02)
    assert at(7071) == pytest.approx(settings.denoise_weight_5k_10k, abs=0.02)
    assert at(15000) == pytest.approx(settings.denoise_weight_above_10k, abs=0.02)


def test_band_weights_are_smooth():
    """A hard-edged weight profile is a brick-wall filter on the difference
    signal, and rings. The curve has to have no step in it."""
    weights = blend.band_weights(SR, blend.N_FFT)
    assert np.abs(np.diff(weights)).max() < 0.05


def test_band_weights_stay_within_range():
    weights = blend.band_weights(SR, blend.N_FFT)
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0


def test_low_frequencies_are_protected_from_the_denoiser(guitar):
    """Reproduces defect #3's mirror image: the fundamentals must survive.

    The stand-in model attenuates everything by 12dB. With the frequency
    weights in place, the bands carrying the music barely move while the high
    band -- where hiss lives -- takes the intended cut.
    """
    wet = (guitar * 10 ** (-12.0 / 20)).astype(np.float32)
    out = blend.blend_denoise(guitar, wet, SR, confidence=1.0).audio

    report = quality.compare(guitar, out, SR, SR)
    for name in ("100_250", "250_500", "500_1k", "1k_2k"):
        assert abs(report.spectral.band_delta_db[name]) < 0.5, f"{name} should be protected"


def test_the_presence_band_is_held_back(guitar):
    """5-10kHz carries pick attack and articulation.

    The weight there is small, so a model asking for a 12dB cut across the board
    gets a fraction of it in this band -- well inside the +/-1dB the plan sets
    as the target for the reference guitar once confidence scaling applies.
    """
    wet = (guitar * 10 ** (-12.0 / 20)).astype(np.float32)
    out = blend.blend_denoise(guitar, wet, SR, confidence=1.0).audio

    report = quality.compare(guitar, out, SR, SR)
    presence = report.spectral.band_delta_db["5k_8k"]
    assert -4.0 < presence < 0.5, f"presence moved {presence:+.2f} dB"


def test_confidence_scales_the_result(guitar):
    wet = (guitar * 10 ** (-12.0 / 20)).astype(np.float32)

    strong = blend.blend_denoise(guitar, wet, SR, confidence=1.0)
    weak = blend.blend_denoise(guitar, wet, SR, confidence=0.25)

    assert weak.effective_wet < strong.effective_wet
    assert weak.effective_wet == pytest.approx(strong.effective_wet * 0.25, rel=0.01)


# --- Maximum attenuation ------------------------------------------------


def test_the_attenuation_limit_caps_a_destructive_model(guitar):
    """Reproduces defect #2 and shows the cap holding.

    The stand-in model is the measured expander: 15.75dB off the quietest
    frames. Without a limit that reaches the output; with one it cannot exceed
    RESTORE_MAX_ATTENUATION_DB no matter what the model returned.
    """
    knee = float(np.percentile(corpus.frame_level_db(guitar, SR), 85))
    wet = corpus.damage_expander(guitar, SR, knee_db=knee, max_attenuation_db=15.75)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)

    unlimited = blend.blend_spectral(guitar, wet, SR, amount=1.0, weights=flat,
                                     max_attenuation_db=0.0).audio
    limited = blend.blend_spectral(guitar, wet, SR, amount=1.0, weights=flat,
                                   max_attenuation_db=6.0).audio

    loose = quality.compare(guitar, unlimited, SR, SR).dynamic
    tight = quality.compare(guitar, limited, SR, SR).dynamic

    assert loose.expansion_spread_db > 8.0, "the stand-in model should be destructive"
    assert tight.expansion_spread_db < loose.expansion_spread_db
    assert tight.quiet_gain_db > -7.0, f"quiet frames fell {tight.quiet_gain_db:.2f} dB"


def test_the_limit_does_not_touch_a_gentle_result(guitar):
    """A model that stays inside the limit must pass through unchanged."""
    wet = (guitar * 10 ** (-2.0 / 20)).astype(np.float32)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)

    result = blend.blend_spectral(guitar, wet, SR, amount=1.0, weights=flat,
                                  max_attenuation_db=6.0)
    assert result.limited_frames == 0

    report = quality.compare(guitar, result.audio, SR, SR)
    assert report.spectral.program_gain_db == pytest.approx(-2.0, abs=0.15)


def test_note_decay_survives_the_limiter(guitar):
    """The audible consequence of the cap: tails are still there.

    The knee sits high enough that a note's 400ms tail is itself well below it.
    That matters: with a lower knee the tail is already inside the 6dB budget,
    the limiter never binds there, and the test would pass or fail for reasons
    unrelated to decay.
    """
    knee = float(np.percentile(corpus.frame_level_db(guitar, SR), 97))
    wet = corpus.damage_expander(guitar, SR, knee_db=knee, max_attenuation_db=15.0)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)

    unlimited = quality.compare(
        guitar,
        blend.blend_spectral(guitar, wet, SR, 1.0, weights=flat, max_attenuation_db=0.0).audio,
        SR, SR,
    )
    limited = quality.compare(
        guitar,
        blend.blend_spectral(guitar, wet, SR, 1.0, weights=flat, max_attenuation_db=6.0).audio,
        SR, SR,
    )

    assert limited.transient.decay_delta_db > unlimited.transient.decay_delta_db
    # And the level-dependent gain is capped at the configured limit exactly.
    assert limited.dynamic.quiet_gain_db == pytest.approx(-6.0, abs=0.5)
    assert unlimited.dynamic.quiet_gain_db < -12.0


def test_limit_reports_what_it_did(guitar):
    knee = float(np.percentile(corpus.frame_level_db(guitar, SR), 85))
    wet = corpus.damage_expander(guitar, SR, knee_db=knee, max_attenuation_db=15.0)
    flat = np.ones(blend.N_FFT // 2 + 1, dtype=np.float32)

    result = blend.blend_spectral(guitar, wet, SR, 1.0, weights=flat, max_attenuation_db=6.0)

    assert result.limited_frames > 0
    assert result.total_frames > result.limited_frames
    assert "limited_frames" in result.to_dict()


# --- Bandwidth: add above, never rewrite below --------------------------


def test_bandwidth_leaves_the_low_band_alone(guitar):
    """The step exists to add what is missing, not to revise what is there."""
    limited = corpus.band_limit(guitar, SR, cutoff_hz=9000.0)
    # A stand-in model that adds a high band *and* meddles below the cutoff.
    meddling = corpus.damage_presence(
        corpus.damage_synthetic_hf(limited, SR, level_db=-40.0), SR,
        low_hz=500, high_hz=2000, gain_db=-6.0,
    )

    out = blend.blend_bandwidth(limited, meddling, SR, cliff_hz=9000.0, confidence=1.0).audio
    report = quality.compare(limited, out, SR, SR)

    for name in ("500_1k", "1k_2k", "2k_5k"):
        assert abs(report.spectral.band_delta_db[name]) < 0.5, (
            f"{name} moved {report.spectral.band_delta_db[name]:+.2f} dB below the cliff"
        )


def test_bandwidth_does_add_above_the_cliff(guitar):
    limited = corpus.band_limit(guitar, SR, cutoff_hz=9000.0)
    extended = corpus.damage_synthetic_hf(limited, SR, level_db=-35.0)

    out = blend.blend_bandwidth(limited, extended, SR, cliff_hz=9000.0, confidence=1.0).audio
    report = quality.compare(limited, out, SR, SR)

    assert report.spectral.band_delta_db["10k_16k"] > 1.0


def test_high_band_weights_are_zero_below_the_cliff():
    import librosa

    weights = blend.high_band_weights(SR, blend.N_FFT, cliff_hz=9000.0)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=blend.N_FFT)

    assert weights[freqs < 7500].max() < 0.01
    assert weights[freqs > 10000].min() > 0.99
    assert np.abs(np.diff(weights)).max() < 0.2


# --- Mask plumbing ------------------------------------------------------


def test_a_zero_mask_blocks_all_processing(guitar):
    wet = (guitar * 10 ** (-12.0 / 20)).astype(np.float32)
    mask = np.zeros((64, 100), dtype=np.float32)

    out = blend.blend_denoise(guitar, wet, SR, confidence=1.0, mask=mask).audio
    report = quality.compare(guitar, out, SR, SR)

    assert abs(report.spectral.program_gain_db) < 0.05
    assert report.timing.correlation_min > 0.999


def test_a_mask_of_a_different_shape_is_resized(guitar):
    wet = (guitar * 10 ** (-6.0 / 20)).astype(np.float32)
    for shape in ((16, 20), (1, 5), (513, 900)):
        result = blend.blend_denoise(guitar, wet, SR, mask=np.ones(shape, dtype=np.float32))
        assert result.audio.shape == guitar.shape


# --- Contracts ----------------------------------------------------------


def test_channel_count_is_preserved(guitar):
    wet = corpus.add_hiss(guitar, -30.0)
    assert blend.blend_denoise(guitar, wet, SR).audio.shape[1] == 2

    mono = guitar.mean(axis=1).astype(np.float32)
    mono_wet = corpus.add_hiss(mono, -30.0)
    assert blend.blend_denoise(mono, mono_wet, SR).audio.ndim == 1


def test_mismatched_channels_are_refused(guitar):
    with pytest.raises(ValueError):
        blend.blend_denoise(guitar, guitar.mean(axis=1).astype(np.float32), SR)


def test_settings_drive_the_defaults(guitar):
    wet = (guitar * 10 ** (-12.0 / 20)).astype(np.float32)

    gentle = blend.blend_denoise(guitar, wet, SR, settings=Settings(denoise_wet=0.2))
    strong = blend.blend_denoise(guitar, wet, SR, settings=Settings(denoise_wet=0.9))

    assert gentle.effective_wet < strong.effective_wet


# --- Values the benchmark chose, not values that were assumed ------------


def test_the_presence_weight_is_the_benchmarked_value():
    """0.15, not the 0.40 the plan starts from.

    The plan supplies the weights as starting values and says explicitly to let
    a benchmark decide whether they should be adjusted. `scripts/sweep.py` says
    they should: across 0.0 to 0.40 the amount of noise removed is flat, while
    the presence band steadily loses level. The weight was buying nothing and
    costing something.
    """
    assert Settings().denoise_weight_5k_10k == 0.15
    assert Settings().denoise_weight_below_2k == 0.0
    assert Settings().denoise_weight_above_10k == 1.0


def test_raising_the_presence_weight_costs_presence_and_buys_no_noise_removal(guitar):
    """The measurement behind that default, so it cannot quietly drift back."""
    noisy = corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0)
    wet_harmful = corpus.damage_presence(
        corpus.damage_expander(
            guitar, SR,
            knee_db=float(np.percentile(corpus.frame_level_db(guitar, SR), 85)),
            max_attenuation_db=15.75,
        ),
        SR, low_hz=5000, high_hz=10000, gain_db=-3.43,
    )

    presence, removed = [], []
    for weight in (0.0, 0.15, 0.40):
        settings = Settings(denoise_weight_5k_10k=weight)
        harmed = blend.blend_denoise(
            guitar, wet_harmful, SR, confidence=1.0, settings=settings
        ).audio
        cleaned = blend.blend_denoise(
            noisy, guitar, SR, confidence=1.0, settings=settings
        ).audio
        presence.append(quality.compare(guitar, harmed, SR, SR).spectral.band_delta_db["8k_10k"])
        removed.append(
            -quality.compare(noisy, cleaned, SR, SR).spectral.band_delta_db["10k_16k"]
        )

    # Cost rises with the weight...
    assert presence[0] > presence[1] > presence[2]
    # ...while the benefit does not.
    assert max(removed) - min(removed) < 0.2, f"noise removal should be flat: {removed}"
    # And the chosen value holds the Phase 9 target.
    assert abs(presence[1]) <= 1.0


def test_the_denoise_ratio_default_is_the_highest_safe_candidate():
    """0.5, the top of the plan's candidate set that stays inside the budget.

    Swept against the worst-case model at full confidence, 0.3/0.4/0.5 hold the
    +/-1.5dB band budget and 0.6 upward break it.
    """
    assert Settings().denoise_wet == 0.5
    assert Settings().bandwidth_wet == 0.7
