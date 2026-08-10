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

from . import bandwidth, chain, denoise, gate, tempo
from .audio_io import TARGET_SR, channel_count, is_effectively_silent, load_audio, save_audio
from .config import settings as default_settings
from .metrics import energy_above_hz, spectral_cliff_hz

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


class IntegrityError(RuntimeError):
    """A stage changed something it was not allowed to change."""


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
    input_channels: int = 0
    input_sr: int = 0
    output_channels: int = 0
    output_sr: int = 0
    stereo_strategy: str = ""
    outcome: "chain.RestoreOutcome | None" = None

    @property
    def channels_preserved(self) -> bool:
        return self.input_channels == self.output_channels

    @property
    def engine_config(self) -> dict:
        """The decisions and verification behind this stem, for the record."""
        return self.outcome.engine_config if self.outcome else {}


def check_integrity(
    stage: str, name: str, before: np.ndarray, after: np.ndarray, settings=None
) -> None:
    """Refuse to let a stage silently change the channel layout.

    Channel collapse is unrecoverable and invisible in every other metric the
    pipeline reports -- the previous version turned every stereo stem into a
    mono one and reported success. Checking after each stage is what makes the
    responsible stage identifiable instead of leaving it to be guessed at.
    """
    settings = settings or default_settings
    if not settings.enforce_channel_count:
        return
    if channel_count(after) != channel_count(before):
        raise IntegrityError(
            f"{stage} changed '{name}' from {channel_count(before)} to "
            f"{channel_count(after)} channels"
        )


@dataclass
class RunReport:
    stems: dict[str, StemReport] = field(default_factory=dict)
    tempo_analysis: "tempo.TempoAnalysis | None" = None
    tempo_note: str = ""
    tempo_decision: "gate.StepDecision | None" = None


def _decide_shared_tempo(
    stems: dict[str, tuple[np.ndarray, int]], requested: bool, device: str
) -> "gate.StepDecision":
    """One tempo decision for the whole stem set, on the best reference available.

    The reference is chosen from percussive stems only. The old priority list
    fell through to guitar and piano, which is how a solo guitar came to be
    beat-tracked against its own uneven picking and warped by 375ms.
    """
    if not requested:
        return gate.StepDecision(False, 0.0, "tempo not requested for this run", {})

    candidates = gate.rank_reference_stems(list(stems))
    if not candidates:
        return gate.StepDecision(
            run=False, confidence=0.0,
            reason=f"no percussive reference stem; need one of {gate.REFERENCE_PRIORITY}",
            measurements={"available_stems": sorted(stems)},
        )

    reference = candidates[0]
    audio, sr = stems[reference]
    return gate.decide_tempo(
        audio, sr, stem_name=reference, available_stems=list(stems), is_separated=True,
    )


def discover_stems(input_dir: str | Path) -> list[Path]:
    """List stem files in a directory, in filename order."""
    return sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def load_stems(
    paths: Iterable[Path], target_sr: int = TARGET_SR
) -> dict[str, tuple[np.ndarray, int]]:
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
    separator=None,
    progress: Callable[[str], None] = lambda message: None,
    on_tempo: Callable[["tempo.TempoAnalysis"], None] = lambda analysis: None,
    on_stem: Callable[[StemReport], None] = lambda entry: None,
) -> RunReport:
    """Restore every stem in `input_dir` and write the results to `output_dir`.

    `on_tempo` fires once the shared warp is known; `on_stem` fires as each stem
    finishes and is written. They exist so a UI can report results while the run
    is still going, rather than re-implementing this loop to get progress.

    `separator` accepts an already-loaded denoise model. The checkpoint is
    ~900MB and takes seconds to load, so a caller processing several folders
    should load it once; tests use the same door to supply a stand-in rather
    than downloading a model to check a channel count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_stems(input_dir)
    if not paths:
        raise FileNotFoundError(f"No audio files found in {input_dir}")

    progress(f"Loading {len(paths)} stems")
    # `RESTORE_OUTPUT_SAMPLE_RATE=0` means follow the source, so the load rate
    # follows it too: resampling on the way in and again on the way out is two
    # filters applied to material that asked for neither.
    load_sr = default_settings.output_sample_rate or None
    stems = load_stems(paths, target_sr=load_sr)
    report = RunReport()
    for name, (audio, sr) in stems.items():
        entry = StemReport(name=name, silent=is_effectively_silent(audio))
        entry.cliff_before_hz = spectral_cliff_hz(audio, sr)
        entry.energy_before_pct = energy_above_hz(audio, sr)
        entry.input_channels = channel_count(audio)
        entry.input_sr = sr
        if entry.silent:
            entry.notes.append("effectively silent -- passed through untouched")
        report.stems[name] = entry

    # Tempo is decided once, for the whole set, on the best available reference.
    # It has to stay a shared decision: beat-tracking each stem separately gives
    # each its own warp and drifts them apart, which defeats the point of having
    # stems. That is also why it sits outside `chain.restore_stem`, which is a
    # per-stem view and cannot see the others.
    tempo_decision = _decide_shared_tempo(stems, do_tempo, device)
    report.tempo_decision = tempo_decision
    if tempo_decision.run:
        progress("Step 1 - tempo correction")
        stems, analysis = tempo.correct_tempo(stems, device=device)
        report.tempo_analysis = analysis
        report.tempo_note = analysis.summary()
        on_tempo(analysis)
    else:
        report.tempo_note = f"skipped: {tempo_decision.reason}"
        progress(f"Step 1 - tempo skipped ({tempo_decision.reason})")

    if do_denoise and separator is None:
        progress("Step 2 - loading denoise model")
        separator = denoise.load_separator(output_dir)

    # Captured before the loop: stems are dropped as they are written, and the
    # tempo gate's view of what is available must not shrink as it goes.
    stem_names = list(stems)

    for name, (audio, sr) in list(stems.items()):
        entry = report.stems[name]
        before_chain = audio

        def denoise_model(signal: np.ndarray, rate: int, _name: str = name) -> np.ndarray:
            progress(f"Step 2 - denoising {_name}")
            result = denoise.denoise_stem(signal, rate, separator, work_dir=work_dir)
            entry.denoise_residual_db = result.residual_db
            entry.stereo_strategy = result.stereo_strategy
            entry.denoise_note = (
                result.reason if result.skipped
                else denoise.describe_residual(result.residual_db)
            )
            return result.audio

        def bandwidth_model(signal: np.ndarray, rate: int, _name: str = name) -> np.ndarray:
            progress(f"Step 3 - extending bandwidth of {_name}")
            result = bandwidth.extend_stem(
                signal, rate, apollo_repo, work_dir=work_dir, device=device
            )
            entry.cliff_after_hz = result.cliff_after_hz
            entry.energy_after_pct = result.energy_after_pct
            return result.audio

        enabled = tuple(
            step for step, wanted in
            (("denoise", do_denoise and separator is not None),
             ("bandwidth", do_bandwidth and not entry.silent))
            if wanted
        )
        outcome = chain.restore_stem(
            audio, sr,
            denoise_model=denoise_model if "denoise" in enabled else None,
            bandwidth_model=bandwidth_model if "bandwidth" in enabled else None,
            stem_name=name,
            available_stems=stem_names,
            enable=enabled,
        )
        audio, sr = outcome.audio, outcome.sr
        check_integrity("restoration chain", name, before_chain, audio)
        entry.outcome = outcome

        if entry.cliff_after_hz is None:
            entry.cliff_after_hz = spectral_cliff_hz(audio, sr)
            entry.energy_after_pct = energy_above_hz(audio, sr)

        entry.output_channels = channel_count(audio)
        entry.output_sr = sr
        if not entry.channels_preserved:
            raise IntegrityError(
                f"'{name}' went in with {entry.input_channels} channels and came out with "
                f"{entry.output_channels}"
            )
        entry.output_path = save_audio(output_dir / f"{name}_restored.wav", audio, sr)
        # Drop the audio once written: holding every restored stem costs ~110MB
        # each at full length, and nothing downstream needs them in memory.
        stems.pop(name, None)
        progress(f"Wrote {entry.output_path.name}")
        on_stem(entry)

    return report
