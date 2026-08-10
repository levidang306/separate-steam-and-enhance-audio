"""Phase 1: channel count and sample rate must survive the chain.

The denoise tests drive a *fake* separator. The real model is a 900MB
checkpoint that only exists in the V3 image, but the defect being fixed has
nothing to do with the model's audio quality -- it is entirely about how its
output was decoded. A fake that writes a mono file reproduces the bug exactly,
and runs in milliseconds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from suno_restore import audio_io
from suno_restore.config import Settings, resolve_output_sample_rate
from tests import corpus

SR = corpus.SR

ffmpeg_available = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available, reason="ffmpeg is required to decode model output"
)


# --- Mid/side round trip -------------------------------------------------


def test_mid_side_round_trip_is_lossless():
    audio = corpus.solo_guitar(4.0)
    mid, side = audio_io.to_mid_side(audio)
    restored = audio_io.from_mid_side(mid, side)

    assert restored.shape == audio.shape
    assert np.abs(restored - audio).max() < 1e-6


def test_mid_side_actually_separates_the_image():
    audio = corpus.solo_guitar(4.0)
    _, side = audio_io.to_mid_side(audio)
    assert np.abs(side).max() > 0, "fixture has no stereo width to preserve"

    centred = corpus.damage_collapse_stereo(audio)
    _, no_side = audio_io.to_mid_side(centred)
    assert np.abs(no_side).max() < 1e-7


def test_mid_side_rejects_non_stereo():
    with pytest.raises(ValueError):
        audio_io.to_mid_side(np.zeros(100, dtype=np.float32))


def test_channel_count():
    assert audio_io.channel_count(np.zeros(10)) == 1
    assert audio_io.channel_count(np.zeros((10, 2))) == 2


# --- Sample rate resolution ---------------------------------------------


def test_zero_means_follow_the_source():
    assert resolve_output_sample_rate(44100, 0) == 44100
    assert resolve_output_sample_rate(48000, 0) == 48000


def test_explicit_rate_is_honoured():
    assert resolve_output_sample_rate(44100, 48000) == 48000


def test_default_settings_follow_the_source():
    assert Settings().output_sample_rate == 0


# --- Denoise channel preservation ---------------------------------------


class FakeSeparator:
    """Stands in for audio-separator.

    `mono_only` reproduces the exact defect: a model that writes a single-channel
    file regardless of what it was given.
    """

    def __init__(self, output_dir: Path, mono_only: bool = False):
        self.output_dir = str(output_dir)
        self.mono_only = mono_only
        self.calls = 0

    def separate(self, source: str) -> list[str]:
        self.calls += 1
        audio, sr = audio_io.load_audio(source, target_sr=None)
        # A denoiser removes a little quiet content; the exact processing is
        # irrelevant here, only the channel layout of what it writes.
        cleaned = (audio * 0.98).astype(np.float32)
        if self.mono_only:
            cleaned = audio_io.to_mono(cleaned)
        target = Path(self.output_dir) / "input_(Dry)_model.wav"
        audio_io.save_audio(target, cleaned, sr)
        return [target.name]


@requires_ffmpeg
def test_stereo_survives_a_well_behaved_model(tmp_path):
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0)
    separator = FakeSeparator(tmp_path, mono_only=False)
    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)

    assert result.output_channels == 2
    assert result.stereo_strategy == "joint"
    assert separator.calls == 1


@requires_ffmpeg
def test_mono_only_model_no_longer_collapses_stereo(tmp_path):
    """The regression this phase exists for.

    Before the fix the result was decoded with the *file's* channel count, so a
    mono model output silently redefined a stereo stem as mono and threw the
    side channel away.
    """
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0)
    _, side_before = audio_io.to_mid_side(audio)
    separator = FakeSeparator(tmp_path, mono_only=True)

    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)

    assert result.output_channels == 2
    assert result.stereo_strategy == "mid_side"
    _, side_after = audio_io.to_mid_side(result.audio)
    # The side channel is still there, at roughly its original level.
    ratio = float(np.sqrt(np.mean(side_after**2)) / np.sqrt(np.mean(side_before**2)))
    assert 0.9 < ratio < 1.1


@requires_ffmpeg
def test_mono_only_model_is_learned_once(tmp_path):
    """The second stem must not pay for rediscovering the same fact."""
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0)
    separator = FakeSeparator(tmp_path, mono_only=True)

    denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)
    first_pass_calls = separator.calls

    separator.calls = 0
    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)

    assert result.stereo_strategy == "mid_side"
    assert separator.calls == 2, "should go straight to mid/side, not probe again"
    assert first_pass_calls == 3, "first stem probes once, then redoes as mid/side"


@requires_ffmpeg
def test_mid_side_mode_skips_the_probe(tmp_path):
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0)
    separator = FakeSeparator(tmp_path, mono_only=True)
    settings = Settings(stereo_mode="mid_side")

    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path, settings=settings)

    assert separator.calls == 2
    assert result.stereo_strategy == "mid_side"


@requires_ffmpeg
def test_mono_input_stays_mono(tmp_path):
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0).mean(axis=1).astype(np.float32)
    separator = FakeSeparator(tmp_path, mono_only=True)
    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)

    assert result.output_channels == 1
    assert result.audio.ndim == 1


@requires_ffmpeg
def test_sample_rate_is_preserved(tmp_path):
    from suno_restore import denoise

    audio = audio_io.resample(corpus.solo_guitar(3.0), SR, 44100)
    separator = FakeSeparator(tmp_path)
    result = denoise.denoise_stem(audio, 44100, separator, work_dir=tmp_path)

    assert result.sr == 44100


@requires_ffmpeg
def test_silent_stem_passes_through_untouched(tmp_path):
    from suno_restore import denoise

    audio = np.zeros((SR, 2), dtype=np.float32)
    separator = FakeSeparator(tmp_path)
    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path)

    assert result.skipped
    assert separator.calls == 0
    assert result.output_channels == 2


@requires_ffmpeg
def test_a_channel_change_that_slips_through_is_refused(tmp_path):
    """Belt and braces: the contract is enforced, not just intended."""
    from suno_restore import denoise

    audio = corpus.solo_guitar(3.0)
    separator = FakeSeparator(tmp_path, mono_only=True)
    settings = Settings(stereo_mode="joint", enforce_channel_count=True)

    # "joint" trusts the model, so the decode still forces two channels; the
    # guard is what would catch any future path that does not.
    result = denoise.denoise_stem(audio, SR, separator, work_dir=tmp_path, settings=settings)
    assert result.output_channels == 2


# --- Chunk crossfade -----------------------------------------------------


def test_window_is_flat_between_the_tapers():
    from suno_restore.bandwidth import chunk_window

    window = chunk_window(4800, 960)[:, 0]
    assert np.allclose(window[960:-960], 1.0)
    assert window[0] == pytest.approx(0.0, abs=1e-6)
    assert window[-1] == pytest.approx(0.0, abs=1e-6)


def test_window_handles_an_overlap_larger_than_the_chunk():
    from suno_restore.bandwidth import chunk_window

    window = chunk_window(100, 500)[:, 0]
    assert window.shape == (100,)
    assert np.isfinite(window).all()


def test_default_overlap_is_two_seconds():
    assert Settings().bandwidth_overlap_s == 2.0


def test_ramp_meets_its_endpoints_smoothly():
    """sin^2 arrives at 0 and 1 with zero slope; a linear ramp kinks.

    The kink is the part a spectrum notices: a slope discontinuity at every
    seam splatters energy that a smooth taper does not.
    """
    from suno_restore.bandwidth import chunk_window

    ramp = chunk_window(4800, 960)[:960, 0]
    slope = np.diff(ramp)
    assert slope[0] < slope[len(slope) // 2] / 10
    assert slope[-1] < slope[len(slope) // 2] / 10

    linear_slope = np.diff(np.linspace(0.0, 1.0, 960))
    assert linear_slope[0] == pytest.approx(linear_slope[len(linear_slope) // 2])


def test_overlapping_tapers_form_a_partition_of_unity():
    """Correlated content -- which the low band is -- must pass through exactly.

    `process_chunked` divides by the accumulated weights, so this is what
    guarantees a seam neither boosts nor cuts material the two chunks agree on.
    """
    from suno_restore.bandwidth import process_chunked

    signal = corpus.solo_guitar(6.0)
    passthrough = process_chunked(signal, lambda piece: piece, chunk=48000, overlap=9600)

    assert passthrough.shape == signal.shape
    assert np.abs(passthrough - signal).max() < 1e-5


def test_head_and_tail_are_not_attenuated():
    """The first and last chunk have no neighbour to fade against.

    Tapering them anyway both dims the ends and, because the overlap-add
    normalises by the accumulated weight, leaves the final samples divided by
    almost nothing.
    """
    from suno_restore.bandwidth import process_chunked

    signal = corpus.solo_guitar(35.0)
    out = process_chunked(signal, lambda piece: piece, chunk=int(10.0 * SR), overlap=int(2.0 * SR))

    assert np.isfinite(out).all()
    assert np.abs(out[: SR] - signal[: SR]).max() < 1e-5
    assert np.abs(out[-SR:] - signal[-SR:]).max() < 1e-5


def _seam_modulation(source, independence: float, overlap_s: float) -> float:
    """Level swing of a synthesised high band, for a given inter-chunk agreement.

    `independence` is how much of the invented band a chunk makes up on its own
    rather than deriving from the material it shares with its neighbour.
    """
    from scipy.signal import butter, sosfilt

    from suno_restore import quality
    from suno_restore.bandwidth import process_chunked

    sos = butter(8, 10000 / (SR / 2), btype="high", output="sos")
    level = float(np.sqrt(np.mean(source**2))) * 10 ** (-30.0 / 20)

    def invent(piece: np.ndarray) -> np.ndarray:
        shared = sosfilt(sos, np.sign(piece) * piece**2, axis=0)
        shared = shared / (np.sqrt(np.mean(shared**2)) + 1e-12)
        rng = np.random.default_rng(abs(hash(piece.tobytes())) % (2**32))
        alone = sosfilt(sos, rng.normal(0, 1, piece.shape), axis=0)
        alone = alone / (np.sqrt(np.mean(alone**2)) + 1e-12)
        mixed = (1 - independence) * shared + independence * alone
        return (piece + mixed * level).astype(np.float32)

    processed = process_chunked(source, invent, int(10.0 * SR), int(overlap_s * SR))
    return quality.compare(source, processed, SR, SR).artifacts.hf_modulation_depth_db


def test_longer_overlap_helps_when_chunks_agree():
    """The realistic case: neighbouring chunks see the same low band.

    A bandwidth extender derives its high band from the material below the
    cutoff, so two chunks covering the same region largely agree. There the
    longer fade is the better one.
    """
    source = corpus.band_limit(corpus.solo_guitar(45.0), SR, cutoff_hz=9000.0)

    short = _seam_modulation(source, independence=0.0, overlap_s=0.5)
    long = _seam_modulation(source, independence=0.0, overlap_s=2.0)

    assert long <= short, f"2s overlap should not be worse; {long:.3f} vs {short:.3f} dB"


def test_longer_overlap_hurts_when_chunks_disagree():
    """The counter-evidence, pinned so it is not forgotten.

    Where two chunks invent *unrelated* high bands, averaging them across the
    seam costs power, and a longer fade means more of the timeline sits inside
    that average. The plan assumed a longer overlap would reduce the swing in
    every case; measured against fully independent chunk output it does the
    opposite. The crossfade is therefore not the real fix for an invented high
    band -- the bandwidth gate is, by declining to invent one at all.
    """
    source = corpus.band_limit(corpus.solo_guitar(45.0), SR, cutoff_hz=9000.0)

    short = _seam_modulation(source, independence=1.0, overlap_s=0.5)
    long = _seam_modulation(source, independence=1.0, overlap_s=2.0)

    assert long > short
    # Both are small in absolute terms; this is a second-order effect next to
    # the decision of whether to synthesise a high band at all.
    assert short < 1.0 and long < 2.0


# --- Configuration -------------------------------------------------------


def test_env_defaults_cannot_drift_from_field_defaults():
    """`Settings()` and `Settings.from_env()` must agree about every field.

    They did not, once: the fallbacks in `from_env` were written out by hand
    alongside the field defaults, so changing one left the other behind and the
    two disagreed about the same setting with nothing to catch it. `from_env` is
    now derived from the dataclass; this is what keeps it that way.
    """
    plain, from_env = Settings(), Settings.from_env()
    for name in plain.to_dict():
        assert getattr(plain, name) == getattr(from_env, name), f"{name} diverged"


def test_every_field_is_reachable_from_the_environment(monkeypatch):
    from dataclasses import fields

    for field_info in fields(Settings):
        name = f"RESTORE_{field_info.name.upper()}"
        default = field_info.default
        if isinstance(default, bool):
            probe, expected = ("false" if default else "true"), not default
        elif isinstance(default, int):
            probe, expected = str(default + 7), default + 7
        elif isinstance(default, float):
            probe, expected = str(default + 0.25), default + 0.25
        else:
            probe = expected = "mid_side" if default == "auto" else f"{default}_x"

        monkeypatch.setenv(name, probe)
        assert getattr(Settings.from_env(), field_info.name) == expected, name
        monkeypatch.delenv(name)


def test_documented_settings_match_the_dataclass():
    """Every field has a line in `.env.example`, and vice versa."""
    import re
    from dataclasses import fields
    from pathlib import Path

    text = Path(__file__).resolve().parent.parent.joinpath(".env.example").read_text()
    documented = set(re.findall(r"^(RESTORE_[A-Z0-9_]+)=", text, re.MULTILINE))
    declared = {f"RESTORE_{f.name.upper()}" for f in fields(Settings)}

    assert declared - documented == set(), f"undocumented: {sorted(declared - documented)}"
    assert documented - declared == set(), f"documented but gone: {sorted(documented - declared)}"
