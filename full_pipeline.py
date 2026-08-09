"""Merged restoration pipeline: midi_repair (Stage A) -> suno_restore (Stage B).

Stage A (midi_repair, ported from rewrite-v2) assumes the input stem may have
one localized damaged span, finds it, and repairs only that span by
regenerating MIDI and re-rendering from real donor audio elsewhere in the
same stem. If nothing is flagged, Stage A is a no-op pass-through.

Stage B (suno_restore, this branch's own design) assumes three defects are
present across the whole stem regardless of Stage A's outcome -- tempo drift,
hiss/crackle, and bandwidth loss -- and corrects all three unconditionally.

Order matters for one concrete reason: Stage A's risk detector
(spectral flatness) is calibrated on the *input* signal. Running Stage B
first would denoise and bandwidth-extend the stem before Stage A ever sees
it, changing the flatness profile the detector was calibrated against.
Stage A therefore always runs first, on the rawest signal available.

Stage A works internally at 22050Hz (a rewrite-v2 constant, unchanged here).
Stage B requires 48kHz throughout -- its tempo/denoise/bandwidth models and
its own cross-stem sync guarantee are built around that rate. The handoff
resamples Stage A's output up to `suno_restore.TARGET_SR` before Stage B ever
sees it; skipping this step would feed Stage B a signal already missing
everything above ~10kHz (half of 22050), which bandwidth extension cannot
distinguish from a genuinely narrow-band stem.

GPU memory: Stage A's models (BS-RoFormer, MuScriptor, the AMT model) and
Stage B's models (beat_this, the denoiser, Apollo) are each individually
sized to fit an 8GB card, but were never tested resident at the same time.
`midi_repair.unload_models()` runs between stages so only one stage's models
occupy VRAM at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import midi_repair
import suno_restore
from suno_restore.audio_io import TARGET_SR, resample, save_audio
from suno_restore.pipeline import RunReport


@dataclass
class CombinedReport:
    stage_a: midi_repair.EnhanceReport | None
    stage_b: RunReport


def restore_from_stem(
    input_path: str | Path,
    output_dir: str | Path,
    apollo_repo: str | Path,
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
    """Run the full merged pipeline on one input stem.

    `input_path` is a single audio file -- a stem exported by Suno, or one of
    V1's 9 separated stems. `output_dir` receives Stage B's final output
    (`<name>_restored.wav`, 48kHz). `do_stage_a=False` skips straight to Stage
    B on the raw input, at its native sample rate resampled to 48kHz --
    useful when the input has no localized damage to speak of and only the
    three whole-track defects need correcting.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    work_dir = Path(work_dir) if work_dir else output_dir / "_work"
    stage_a_dir = work_dir / "stage_a"
    stage_b_input_dir = work_dir / "stage_b_input"
    stage_a_dir.mkdir(parents=True, exist_ok=True)
    stage_b_input_dir.mkdir(parents=True, exist_ok=True)

    stage_a_report = None
    if do_stage_a:
        progress("Stage A - localized MIDI repair")
        stage_a_report = midi_repair.enhance(input_path, stage_a_dir, device=device, progress=progress)

        progress("Releasing Stage A models before Stage B loads")
        midi_repair.unload_models()

        progress(f"Resampling Stage A output {stage_a_report.final_sr}Hz -> {TARGET_SR}Hz")
        resampled = resample(stage_a_report.final_audio, stage_a_report.final_sr, TARGET_SR)
        save_audio(stage_b_input_dir / f"{input_path.stem}.wav", resampled, TARGET_SR)
    else:
        progress("Stage A skipped - loading raw input for Stage B")
        from suno_restore.audio_io import load_audio

        audio, sr = load_audio(input_path, target_sr=TARGET_SR)
        save_audio(stage_b_input_dir / f"{input_path.stem}.wav", audio, sr)

    progress("Stage B - whole-track tempo/denoise/bandwidth correction")
    stage_b_report = suno_restore.restore(
        stage_b_input_dir,
        output_dir,
        do_tempo=do_tempo,
        do_denoise=do_denoise,
        do_bandwidth=do_bandwidth,
        apollo_repo=apollo_repo,
        device=device or "cpu",
        work_dir=work_dir,
        progress=progress,
        on_tempo=on_tempo,
        on_stem=on_stem,
    )

    return CombinedReport(stage_a=stage_a_report, stage_b=stage_b_report)
