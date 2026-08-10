"""The directory-level orchestrator still holds its contracts.

`pipeline.restore` keeps one thing the per-stem chain cannot: tempo is decided
once for the whole set and applied identically to every stem. Beat-tracking each
stem separately would give each its own warp and drift them apart, which is the
one thing a stem export must not do.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from suno_restore import audio_io, gate, pipeline
from tests import corpus
from tests.test_signal_integrity import FakeSeparator

SR = corpus.SR

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required to decode audio"
)


def _stems(names: list[str], seconds: float = 6.0) -> dict[str, tuple[np.ndarray, int]]:
    builders = {
        "drums": corpus.drums,
        "bass": corpus.full_mix,
        "guitar": corpus.solo_guitar,
        "piano": corpus.solo_piano,
        "vocals": corpus.vocal,
    }
    return {
        name: (builders.get(name, corpus.solo_guitar)(seconds), SR) for name in names
    }


# --- The shared tempo decision ------------------------------------------


def test_tempo_is_not_requested_means_not_run():
    decision = pipeline._decide_shared_tempo(_stems(["drums"]), requested=False, device="cpu")
    assert not decision.run
    assert "not requested" in decision.reason


def test_a_set_with_no_percussive_stem_is_refused():
    """The exact configuration that warped the reference guitar.

    A solo guitar export has no drum stem, so the old priority list fell through
    to the guitar itself and beat-tracked it against its own uneven picking.
    """
    decision = pipeline._decide_shared_tempo(
        _stems(["guitar", "piano", "vocals"]), requested=True, device="cpu"
    )
    assert not decision.run
    assert "percussive reference" in decision.reason
    assert decision.measurements["available_stems"] == ["guitar", "piano", "vocals"]


def test_a_percussive_set_is_still_refused_on_a_separated_stem():
    """Separation reshapes the transients a beat tracker depends on."""
    decision = pipeline._decide_shared_tempo(
        _stems(["drums", "bass", "vocals"]), requested=True, device="cpu"
    )
    assert not decision.run
    assert "separated" in decision.reason or "disabled" in decision.reason


def test_the_reference_is_chosen_from_percussive_stems_only():
    assert gate.rank_reference_stems(["vocals", "guitar", "drums", "piano"])[0] == "drums"
    assert gate.rank_reference_stems(["guitar", "piano"]) == []


# --- End to end ----------------------------------------------------------


@requires_ffmpeg
def test_a_run_preserves_channels_and_sample_rate(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    for name in ("guitar", "vocals"):
        audio_io.save_audio(source / f"{name}.wav", corpus.solo_guitar(5.0), SR)

    separator = FakeSeparator(tmp_path, mono_only=True)
    report = pipeline.restore(
        source, tmp_path / "out",
        do_tempo=False, do_denoise=True, do_bandwidth=False,
        work_dir=tmp_path,
        separator=separator,
    )

    assert set(report.stems) == {"guitar", "vocals"}
    for entry in report.stems.values():
        assert entry.channels_preserved
        assert entry.input_channels == 2
        assert entry.output_sr == SR
        assert entry.output_path is not None and entry.output_path.exists()


@requires_ffmpeg
def test_every_stem_records_why_each_step_ran_or_did_not(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    audio_io.save_audio(source / "guitar.wav", corpus.solo_guitar(5.0), SR)

    report = pipeline.restore(
        source, tmp_path / "out",
        do_tempo=False, do_denoise=True, do_bandwidth=False,
        work_dir=tmp_path,
        separator=FakeSeparator(tmp_path),
    )

    config = report.stems["guitar"].engine_config
    assert set(config["steps"]) == {"tempo", "denoise", "bandwidth"}
    for step in config["steps"].values():
        assert step["reason"] or step["skip_reason"]
    assert "verification" in config
    assert isinstance(config["is_enhanced"], bool)


@requires_ffmpeg
def test_the_tempo_decision_is_recorded_on_the_run(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    audio_io.save_audio(source / "guitar.wav", corpus.solo_guitar(5.0), SR)

    report = pipeline.restore(
        source, tmp_path / "out",
        do_tempo=True, do_denoise=False, do_bandwidth=False,
        work_dir=tmp_path,
    )

    assert report.tempo_decision is not None
    assert not report.tempo_decision.run
    assert "skipped" in report.tempo_note
