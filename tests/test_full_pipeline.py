"""Contracts for the renewed Version 2 + Version 3 integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

import full_pipeline
from midi_repair import EnhanceReport, preserve_source_layout
from suno_restore import gate, pipeline
from suno_restore.audio_io import save_audio


def test_v2_repair_delta_preserves_native_stereo_layout():
    source_sr = 48000
    analysis_sr = 22050
    duration = 1.0
    source_t = np.arange(int(source_sr * duration)) / source_sr
    mid = 0.15 * np.sin(2 * np.pi * 220 * source_t)
    side = 0.03 * np.sin(2 * np.pi * 3 * source_t)
    source = np.stack([mid + side, mid - side], axis=1).astype(np.float32)

    analysis = np.zeros(int(analysis_sr * duration), dtype=np.float32)
    repaired = analysis.copy()
    repaired[int(0.3 * analysis_sr) : int(0.6 * analysis_sr)] = 0.02

    result = preserve_source_layout(source, source_sr, analysis, repaired, analysis_sr)

    assert result.shape == source.shape
    # The same repair delta is applied to left and right, so stereo side
    # information remains exactly what the native source carried.
    np.testing.assert_allclose(result[:, 0] - result[:, 1], source[:, 0] - source[:, 1], atol=1e-7)
    assert np.max(np.abs(result - source)) > 0.01


def test_combined_pipeline_saves_and_exposes_each_version(tmp_path, monkeypatch):
    source_sr = 32000
    frames = source_sr // 2
    version_2_audio = np.stack(
        [np.linspace(-0.1, 0.1, frames), np.linspace(0.1, -0.1, frames)],
        axis=1,
    ).astype(np.float32)
    input_path = save_audio(tmp_path / "input.wav", version_2_audio * 0.5, source_sr)
    unloaded = []
    observed = {}

    def fake_enhance(input_path, work_dir, device=None, progress=lambda message: None):
        return EnhanceReport(
            target_stem="instrumental",
            final_audio=version_2_audio,
            final_sr=source_sr,
            is_enhanced=True,
        )

    def fake_restore(input_dir, output_dir, **kwargs):
        staged_paths = list(Path(input_dir).glob("*.wav"))
        assert len(staged_paths) == 1
        staged_audio, staged_sr = sf.read(staged_paths[0], always_2d=True, dtype="float32")
        observed["audio"] = staged_audio
        observed["sr"] = staged_sr

        output_path = save_audio(
            Path(output_dir) / "input_restored.wav",
            staged_audio * 0.9,
            staged_sr,
        )
        entry = pipeline.StemReport(
            name="input",
            output_path=output_path,
            input_channels=2,
            output_channels=2,
            input_sr=staged_sr,
            output_sr=staged_sr,
        )
        return pipeline.RunReport(
            stems={"input": entry},
            tempo_decision=gate.StepDecision(False, 0.0, "test skip", {}),
        )

    monkeypatch.setattr(full_pipeline.midi_repair, "enhance", fake_enhance)
    monkeypatch.setattr(full_pipeline.midi_repair, "unload_models", lambda: unloaded.append(True))
    monkeypatch.setattr(full_pipeline.suno_restore, "restore", fake_restore)

    report = full_pipeline.restore_from_stem(
        input_path,
        tmp_path / "output",
        apollo_repo=None,
        device="cpu",
    )

    assert report.version_2_path == tmp_path / "output" / "v2" / "input_v2.wav"
    assert report.version_3_path == tmp_path / "output" / "v3" / "input_restored.wav"
    assert report.version_2_path.exists()
    assert report.version_3_path.exists()
    assert report.stage_a is not None
    assert report.stage_a.output_path == report.version_2_path
    assert unloaded == [True]
    assert observed["sr"] == source_sr
    assert observed["audio"].shape == version_2_audio.shape
    np.testing.assert_allclose(observed["audio"], version_2_audio, atol=2e-7)


def test_version_2_models_are_released_when_version_2_fails(tmp_path, monkeypatch):
    input_path = save_audio(tmp_path / "input.wav", np.zeros(1000, dtype=np.float32), 1000)
    unloaded = []

    def fail_enhance(*args, **kwargs):
        raise RuntimeError("V2 failed")

    monkeypatch.setattr(full_pipeline.midi_repair, "enhance", fail_enhance)
    monkeypatch.setattr(full_pipeline.midi_repair, "unload_models", lambda: unloaded.append(True))

    try:
        full_pipeline.restore_from_stem(input_path, tmp_path / "output", apollo_repo=None)
    except RuntimeError as error:
        assert str(error) == "V2 failed"
    else:
        raise AssertionError("expected Version 2 failure")

    assert unloaded == [True]
