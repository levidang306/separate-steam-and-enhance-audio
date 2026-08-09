"""Orchestration for the Suno stem restoration workflow.

Order is fixed and deliberate:

  1. Tempo correction -- the only step that touches the time axis, so it runs
     first and is computed once across all stems. Running it later would
     re-stretch whatever the other steps had just repaired.
  2. Denoise -- before bandwidth extension, because super-resolution models
     synthesise the new high band from what sits below the cutoff; feeding them
     noise extends the noise.
  3. Bandwidth extension -- last, and the only step that changes sample rate
     (Apollo emits 44.1kHz, which is resampled back to 48kHz).

Steps are individually switchable. On the reference stem set, measurement showed
the cutoff sits near 16kHz rather than the documented 12-13kHz, hiss is 49-58dB
down, and tempo holds to within 1% -- so each step has to justify itself per
track rather than being assumed necessary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import bandwidth, denoise, tempo
from .audio_io import TARGET_SR, is_effectively_silent, load_audio, save_audio
from .metrics import energy_above_hz, spectral_cliff_hz

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


@dataclass
class StemReport:
    name: str
    silent: bool = False
    denoise_residual_db: float | None = None
    denoise_note: str = ""
    cliff_before_hz: float | None = None
    cliff_after_hz: float | None = None
    energy_before_pct: float | None = None
    energy_after_pct: float | None = None
    output_path: Path | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    stems: dict[str, StemReport] = field(default_factory=dict)
    tempo_analysis: "tempo.TempoAnalysis | None" = None
    tempo_note: str = ""


def discover_stems(input_dir: str | Path) -> list[Path]:
    """List stem files in a directory, in filename order."""
    return sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def load_stems(paths: Iterable[Path], target_sr: int = TARGET_SR) -> dict[str, tuple[np.ndarray, int]]:
    return {p.stem: load_audio(p, target_sr=target_sr) for p in paths}


def restore(
    input_dir: str | Path,
    output_dir: str | Path,
    do_tempo: bool = True,
    do_denoise: bool = True,
    do_bandwidth: bool = True,
    apollo_repo: str | Path | None = None,
    device: str = "cpu",
    work_dir: str | Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
    on_tempo: Callable[["tempo.TempoAnalysis"], None] = lambda analysis: None,
    on_stem: Callable[[StemReport], None] = lambda entry: None,
) -> RunReport:
    """Restore every stem in `input_dir` and write the results to `output_dir`.

    `on_tempo` fires once the shared warp is known; `on_stem` fires as each stem
    finishes and is written. They exist so a UI can report results while the run
    is still going, rather than re-implementing this loop to get progress.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_stems(input_dir)
    if not paths:
        raise FileNotFoundError(f"No audio files found in {input_dir}")

    progress(f"Loading {len(paths)} stems")
    stems = load_stems(paths)
    report = RunReport()
    for name, (audio, sr) in stems.items():
        entry = StemReport(name=name, silent=is_effectively_silent(audio))
        entry.cliff_before_hz = spectral_cliff_hz(audio, sr)
        entry.energy_before_pct = energy_above_hz(audio, sr)
        if entry.silent:
            entry.notes.append("effectively silent -- passed through untouched")
        report.stems[name] = entry

    if do_tempo:
        progress("Step 1 - tempo correction")
        stems, analysis = tempo.correct_tempo(stems, device=device)
        report.tempo_analysis = analysis
        report.tempo_note = analysis.summary()
        on_tempo(analysis)

    separator = None
    if do_denoise:
        progress("Step 2 - loading denoise model")
        separator = denoise.load_separator(output_dir)

    for name, (audio, sr) in list(stems.items()):
        entry = report.stems[name]

        if do_denoise and separator is not None:
            progress(f"Step 2 - denoising {name}")
            result = denoise.denoise_stem(audio, sr, separator, work_dir=work_dir)
            audio, sr = result.audio, result.sr
            entry.denoise_residual_db = result.residual_db
            entry.denoise_note = (
                result.reason if result.skipped else denoise.describe_residual(result.residual_db)
            )

        if do_bandwidth and not entry.silent:
            progress(f"Step 3 - extending bandwidth of {name}")
            result = bandwidth.extend_stem(
                audio, sr, apollo_repo, work_dir=work_dir, device=device
            )
            audio, sr = result.audio, result.sr
            entry.cliff_after_hz = result.cliff_after_hz
            entry.energy_after_pct = result.energy_after_pct

        if entry.cliff_after_hz is None:
            entry.cliff_after_hz = spectral_cliff_hz(audio, sr)
            entry.energy_after_pct = energy_above_hz(audio, sr)

        entry.output_path = save_audio(output_dir / f"{name}_restored.wav", audio, sr)
        # Drop the audio once written: holding every restored stem costs ~110MB
        # each at full length, and nothing downstream needs them in memory.
        stems.pop(name, None)
        progress(f"Wrote {entry.output_path.name}")
        on_stem(entry)

    return report
