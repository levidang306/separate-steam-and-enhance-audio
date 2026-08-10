"""The directory-level orchestrator still holds its contracts.

Channel count and sample rate have to survive a run. Those are not policy
choices -- they are the two things a restoration pass must never silently
change, and the denoise model has been seen to change both.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from suno_restore import audio_io, pipeline
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


