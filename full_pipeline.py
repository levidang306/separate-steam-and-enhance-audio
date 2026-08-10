"""Combined Version 2 -> Version 3 audio restoration pipeline.

Version 2 (`midi_repair`) detects and repairs one localized damaged span.
Version 3 (`suno_restore`) then runs tempo correction, denoise and bandwidth
extension over the whole file -- each step unconditionally and at full strength
if it is enabled.

Both outputs are kept as durable artifacts rather than one overwriting the
other. That matters more now than it used to: with no verification stage inside
Version 3, the V2 output is the only surviving reference for judging whether V3
improved the file or damaged it. `scripts/ab_quality.py` compares the two.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import midi_repair
import suno_restore
from suno_restore.audio_io import load_audio, save_audio
from suno_restore.pipeline import RunReport


@dataclass
class CombinedReport:
    input_path: Path
    stage_a: midi_repair.EnhanceReport | None
    stage_b: RunReport
    version_2_path: Path | None
    version_3_path: Path


def restore_from_stem(
    input_path: str | Path,
    output_dir: str | Path,
    apollo_repo: str | Path | None,
    device: str | None = None,
    work_dir: str | Path | None = None,
    do_stage_a: bool = True,
    do_tempo: bool = True,
    do_denoise: bool = True,
    do_bandwidth: bool = True,
    progress: Callable[[str], None] = lambda message: None,
    on_tempo: Callable[[object], None] = lambda analysis: None,
    on_stem: Callable[[object], None] = lambda entry: None,
) -> CombinedReport:
    """Run V2 then enhanced V3 on one file and retain both version outputs.

    Outputs are written to `<output_dir>/v2/<name>_v2.wav` and
    `<output_dir>/v3/<name>_restored.wav`. The V2 handoff keeps the original
    sample rate and channel layout; V3 therefore receives the same signal
    representation it is expected to preserve and verify.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    work_dir = Path(work_dir) if work_dir else output_dir / "_work"
    version_2_dir = output_dir / "v2"
    version_3_dir = output_dir / "v3"
    stage_a_dir = work_dir / "v2"
    stage_b_input_dir = work_dir / "v3_input"
    # Scratch space for Version 3. It was passed to `suno_restore.restore`
    # without ever being created, which held for as long as the denoise step
    # kept being skipped -- it is the first thing to open a temporary directory
    # inside it, and `TemporaryDirectory(dir=...)` will not create its parent.
    stage_b_work_dir = work_dir / "v3"

    output_dir.mkdir(parents=True, exist_ok=True)
    version_2_dir.mkdir(parents=True, exist_ok=True)
    version_3_dir.mkdir(parents=True, exist_ok=True)
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    stage_b_work_dir.mkdir(parents=True, exist_ok=True)
    if stage_b_input_dir.exists():
        shutil.rmtree(stage_b_input_dir)
    stage_b_input_dir.mkdir(parents=True, exist_ok=True)

    stage_a_report = None
    version_2_path = None
    if do_stage_a:
        progress("Version 2 - localized MIDI repair")
        try:
            stage_a_report = midi_repair.enhance(
                input_path,
                stage_a_dir,
                device=device,
                progress=progress,
            )
        finally:
            progress("Releasing Version 2 models before Version 3 loads")
            midi_repair.unload_models()

        if stage_a_report.final_audio is None or not stage_a_report.final_sr:
            raise RuntimeError("Version 2 completed without producing audio")
        version_2_path = save_audio(
            version_2_dir / f"{input_path.stem}_v2.wav",
            stage_a_report.final_audio,
            stage_a_report.final_sr,
        )
        stage_a_report.output_path = version_2_path
        handoff_audio, handoff_sr = stage_a_report.final_audio, stage_a_report.final_sr
    else:
        progress("Version 2 skipped - sending the native upload to Version 3")
        handoff_audio, handoff_sr = load_audio(input_path, target_sr=None)

    staged_input = save_audio(
        stage_b_input_dir / f"{input_path.stem}.wav",
        handoff_audio,
        handoff_sr,
    )
    progress(
        f"Version 3 input: {staged_input.name} at {handoff_sr}Hz "
        f"with {'1 channel' if handoff_audio.ndim == 1 else f'{handoff_audio.shape[1]} channels'}"
    )

    progress("Version 3 - tempo, denoise and bandwidth restoration")
    stage_b_report = suno_restore.restore(
        stage_b_input_dir,
        version_3_dir,
        do_tempo=do_tempo,
        do_denoise=do_denoise,
        do_bandwidth=do_bandwidth,
        apollo_repo=apollo_repo,
        device=device or "cpu",
        work_dir=stage_b_work_dir,
        progress=progress,
        on_tempo=on_tempo,
        on_stem=on_stem,
    )

    version_3_paths = [
        entry.output_path
        for entry in stage_b_report.stems.values()
        if entry.output_path is not None
    ]
    if len(version_3_paths) != 1:
        raise RuntimeError(
            f"Version 3 should produce exactly one file, produced {len(version_3_paths)}"
        )

    return CombinedReport(
        input_path=input_path,
        stage_a=stage_a_report,
        stage_b=stage_b_report,
        version_2_path=version_2_path,
        version_3_path=version_3_paths[0],
    )
