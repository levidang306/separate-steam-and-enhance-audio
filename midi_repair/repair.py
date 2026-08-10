"""Localized MIDI-based repair, ported from rewrite-v2's streamlit_app.py.

Assumption: the input is a single stem that may contain one contiguous
damaged span (noise-like, spectral-flatness detectable). Stem separation and
MIDI transcription build an understanding of the stem; a risk detector flags
at most one region; if one is flagged, it is regenerated in MIDI and
re-rendered from real donor notes found elsewhere in the same stem, then
crossfaded back in. If nothing is flagged, the input passes through
unchanged -- this stage never touches material it has no evidence about.

Model loaders are cached per-device with `functools.lru_cache`, same pattern
as `suno_restore.bandwidth.load_model`, so the caller decides when to evict
them (`unload_models()`) rather than a Streamlit-only cache keeping every
model resident for the life of the process -- with two pipelines' worth of
models now sharing one GPU, holding all of them at once risks OOM.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch

from suno_restore.audio_io import load_audio, resample

SR = 22050
HOP_LENGTH = 512

# See streamlit_app.py (rewrite-v2) for the calibration story: clean audio
# never exceeded ~0.20 flatness, injected-noise regions sat at 0.26-0.42.
FLATNESS_LOW, FLATNESS_HIGH = 0.15, 0.35
TIME_RESOLUTION, DUR_OFFSET, NOTE_OFFSET = 100, 10000, 11000


def _device(device: str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=1)
def _load_bs_roformer(device: str):
    import yaml
    from bs_roformer.download import ensure_model_assets
    from bs_roformer.inference import SafeLoaderWithTuple
    from bs_roformer.model_registry import DEFAULT_MODEL
    from bs_roformer.utils import get_model_from_config
    from ml_collections import ConfigDict

    model_path, config_path = ensure_model_assets(DEFAULT_MODEL)
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(_device(device))
    return model, config


@lru_cache(maxsize=1)
def _load_muscriptor(device: str):
    from muscriptor import TranscriptionModel

    dev = _device(device)
    # "large" (~1.37B params, fp32) needs ~5.5GB VRAM at load time on top of
    # BS-RoFormer -- OOMs on <12GB cards. Fall back to "medium" there.
    size = "large"
    if dev.type == "cuda" and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
        size = "medium"
    return TranscriptionModel.load_model(size)


@lru_cache(maxsize=1)
def _load_amt(device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")
    model.eval()
    model.to(_device(device))
    return model


def unload_models() -> None:
    """Drop cached models and free their VRAM.

    Call this between Stage A (this module) and Stage B (suno_restore) --
    both stages together can exceed an 8GB card if their models stay
    resident simultaneously.
    """
    _load_bs_roformer.cache_clear()
    _load_muscriptor.cache_clear()
    _load_amt.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _col_normalize(m):
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)


def _separate_stems(input_path, out_dir: Path, device: str) -> dict:
    import soundfile as sf
    from bs_roformer.utils import demix_track

    model, config = _load_bs_roformer(device)
    # Decode through the same ffmpeg path as V3. libsndfile can silently
    # truncate the low-bitrate VBR MP3 files this project is built for.
    mix, sr = load_audio(input_path, target_sr=None)
    mono = mix.ndim == 1
    mix_stereo = np.stack([mix, mix], axis=-1) if mono else mix
    mixture = torch.tensor(mix_stereo.T, dtype=torch.float32)

    stem_result, _ = demix_track(config, model, mixture, _device(device))
    stem_names = list(stem_result.keys())

    stem_paths, stem_rms = {}, {}
    for name in stem_names:
        audio = stem_result[name].T
        if mono:
            audio = audio[:, 0]
        out_path = out_dir / f"stem_{name}.wav"
        sf.write(out_path, audio, sr, subtype="FLOAT")
        stem_paths[name] = out_path
        stem_rms[name] = _rms(audio)

    target_stem = max(stem_names, key=lambda n: stem_rms[n])
    recon = sum(stem_result[name] for name in stem_names).T
    recon_error = _rms(recon - mix_stereo) / _rms(mix_stereo)
    return {
        "stem_paths": stem_paths,
        "stem_rms": stem_rms,
        "target_stem": target_stem,
        "recon_error": recon_error,
    }


def _transcribe(input_path, out_dir: Path, device: str) -> dict:
    import mido

    muscriptor_model = _load_muscriptor(device)
    midi_bytes = muscriptor_model.transcribe_to_midi(str(input_path))
    midi_path = out_dir / "transcription.mid"
    midi_path.write_bytes(midi_bytes)

    mid = mido.MidiFile(midi_path)
    note_on_events = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    return {"midi_path": midi_path, "n_notes": len(note_on_events), "midi_duration": mid.length}


def _analyze(input_path, midi_path) -> dict:
    from dtw import dtw
    from music21 import converter as m21_converter

    y_mono, sr_mono = load_audio(input_path, target_sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr_mono, hop_length=HOP_LENGTH)

    score = m21_converter.parse(Path(midi_path))
    detected_key = score.analyze("key")

    n_beats = int(np.ceil(score.highestTime))
    midi_chroma = np.zeros((12, max(n_beats, 1)))
    for n in score.flatten().notes:
        beat_idx = min(int(n.offset), n_beats - 1)
        for p in n.pitches if hasattr(n, "pitches") else [n.pitch]:
            midi_chroma[p.pitchClass, beat_idx] += 1

    alignment = dtw(_col_normalize(chroma).T, _col_normalize(midi_chroma).T, keep_internals=True)
    return {
        "detected_key": str(detected_key),
        "dtw_norm_distance": alignment.normalizedDistance,
    }


def compute_degradation_risk(audio_path_or_array, sr: int = SR, hop_length: int = HOP_LENGTH):
    if isinstance(audio_path_or_array, (str, Path)):
        y, _ = load_audio(audio_path_or_array, target_sr=sr, mono=True)
    else:
        y = audio_path_or_array

    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    rms_curve = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    n = len(rms_curve)
    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)

    flatness_risk = np.clip((flatness[:n] - FLATNESS_LOW) / (FLATNESS_HIGH - FLATNESS_LOW), 0, 1)
    return frame_times, flatness_risk


def detect_flagged_region(
    risk, frame_times, z=1.5, min_duration=1.0, pad=0.25, merge_gap=1.0, min_risk=0.4
):
    threshold = max(risk.mean() + z * risk.std(), min_risk)
    above = risk > threshold
    frame_dt = frame_times[1] - frame_times[0] if len(frame_times) > 1 else 0.0
    merge_gap_frames = int(merge_gap / frame_dt) if frame_dt > 0 else 0

    runs = []
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs:
        return None, threshold

    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= merge_gap_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    best_start, best_end = max(merged, key=lambda r: r[1] - r[0])
    start_t = frame_times[best_start]
    end_t = frame_times[min(best_end - 1, n - 1)]
    if end_t - start_t < min_duration:
        mid = (start_t + end_t) / 2
        start_t, end_t = mid - min_duration / 2, mid + min_duration / 2
    return (max(0.0, start_t - pad), end_t + pad), threshold


def decode_notes(events):
    notes = []
    for t, d, note in zip(events[0::3], events[1::3], events[2::3]):
        note_id = note - NOTE_OFFSET
        notes.append(
            {
                "onset": t / TIME_RESOLUTION,
                "dur": (d - DUR_OFFSET) / TIME_RESOLUTION,
                "pitch": note_id % 128,
                "instrument": note_id // 128,
            }
        )
    return notes


def _repair_midi(midi_path, region_start: float, region_end: float, device: str):
    from anticipation import ops, sample
    from anticipation.convert import midi_to_events

    amt_model = _load_amt(device)
    events = midi_to_events(str(midi_path))
    max_t = ops.max_time(events, seconds=True)
    before = ops.clip(events, 0, region_start, clip_duration=False)
    after = ops.clip(events, region_end, max_t, clip_duration=False)
    context = ops.sort(before + after)
    repaired_events = ops.sort(
        sample.generate(amt_model, region_start, region_end, inputs=context, top_p=0.95)
    )
    return repaired_events, events


def find_donor(target_pitch, library):
    exact = [n for n in library if n["pitch"] == target_pitch]
    if exact:
        return min(exact, key=lambda n: abs(n["dur"] - 0.3)), 0
    nearest = min(library, key=lambda n: abs(n["pitch"] - target_pitch))
    return nearest, target_pitch - nearest["pitch"]


def render_window(target_audio, sr, donor_library, target_notes, window_start, window_end):
    # Clamped to what is actually left in the file. The flagged region comes
    # from frame times that can round past the final sample, and a window longer
    # than the audio it replaces produces a patch that cannot be spliced back in.
    start_n = max(0, int(window_start * sr))
    window_len = max(0, min(int((window_end - window_start) * sr), len(target_audio) - start_n))
    rendered = np.zeros(window_len)
    placements = []
    for target in target_notes:
        donor, shift = find_donor(target["pitch"], donor_library)
        d_start = int(donor["onset"] * sr)
        d_end = int((donor["onset"] + max(donor["dur"], 0.05)) * sr)
        donor_audio = target_audio[d_start:d_end]
        if len(donor_audio) < 512:
            continue
        if shift != 0:
            donor_audio = librosa.effects.pitch_shift(donor_audio, sr=sr, n_steps=shift)
        target_dur = max(target["dur"], 0.05)
        rate = np.clip(len(donor_audio) / sr / target_dur, 0.25, 4.0)
        stretched = librosa.effects.time_stretch(donor_audio, rate=rate)
        rel_start = int((target["onset"] - window_start) * sr)
        end = min(rel_start + len(stretched), window_len)
        if rel_start >= window_len or end <= rel_start:
            continue
        rendered[rel_start:end] += stretched[: end - rel_start]
        placements.append({"pitch": target["pitch"], "shift": shift, "rel_start_s": rel_start / sr})
    peak = np.abs(rendered).max()
    if peak > 1.0:
        rendered = rendered / peak * 0.95
    return rendered, placements


def equal_power_crossfade_splice(base, patch, start_s, sr, fade_s=0.05):
    """Drop `patch` into `base` at `start_s`, crossfading both seams.

    The patch is trimmed to the room available rather than assumed to fit. Its
    length comes from the notes that were re-rendered, not from the file, so a
    region near the end of the track can produce more audio than there is space
    for. Previously the destination slice silently clipped at the end of `base`
    while the source slice did not, and the assignment failed on a shape
    mismatch -- 64629 samples into 60343.
    """
    out = base.copy()
    start_n = max(0, int(start_s * sr))
    if start_n >= len(base) or len(patch) == 0:
        return out

    patch = patch[: len(base) - start_n]
    end_n = start_n + len(patch)

    # Each seam needs a fade on both sides of it, so the two fades together
    # cannot be longer than the patch.
    fade_n = int(max(0, min(fade_s * sr, len(patch) // 2)))
    if fade_n == 0:
        out[start_n:end_n] = patch
        return out

    t = np.linspace(0, np.pi / 2, fade_n)
    fade_out, fade_in = np.cos(t), np.sin(t)

    out[start_n : start_n + fade_n] = (
        base[start_n : start_n + fade_n] * fade_out + patch[:fade_n] * fade_in
    )
    out[start_n + fade_n : end_n - fade_n] = patch[fade_n : len(patch) - fade_n]
    out[end_n - fade_n : end_n] = (
        patch[len(patch) - fade_n :] * fade_out + base[end_n - fade_n : end_n] * fade_in
    )
    return out


@dataclass
class EnhanceReport:
    target_stem: str | None = None
    region: tuple[float, float] | None = None
    risk_in_before: float | None = None
    risk_in_after: float | None = None
    final_audio: np.ndarray | None = None
    final_sr: int = 0
    is_enhanced: bool = False
    output_path: Path | None = None
    notes: list[str] = field(default_factory=list)


def preserve_source_layout(
    source_audio: np.ndarray,
    source_sr: int,
    analysis_audio: np.ndarray,
    repaired_audio: np.ndarray,
    analysis_sr: int,
) -> np.ndarray:
    """Apply V2's mono repair delta without replacing the source representation.

    V2 analyzes and renders at 22.05kHz mono, while V3 deliberately preserves the
    uploaded file's native sample rate and channel layout. Passing V2's render
    directly into V3 would therefore discard stereo and real high-frequency content
    before V3 could verify either. Applying only the repair delta to every source
    channel keeps the original stereo difference intact and leaves all untouched
    samples sourced from the original decode.
    """
    if analysis_audio.ndim != 1 or repaired_audio.ndim != 1:
        raise ValueError("V2 repair layout conversion expects mono analysis audio")
    if analysis_audio.shape != repaired_audio.shape:
        raise ValueError("V2 analysis and repaired audio must have the same shape")

    delta = (repaired_audio - analysis_audio).astype(np.float32)
    native_delta = resample(delta, analysis_sr, source_sr)
    fitted_delta = np.zeros(source_audio.shape[0], dtype=np.float32)
    sample_count = min(len(native_delta), len(fitted_delta))
    fitted_delta[:sample_count] = native_delta[:sample_count]

    restored = source_audio.astype(np.float32, copy=True)
    if restored.ndim == 1:
        restored += fitted_delta
    else:
        restored += fitted_delta[:, None]
    return np.clip(restored, -1.0, 1.0).astype(np.float32)


def enhance(
    input_path: str | Path,
    work_dir: str | Path,
    device: str | None = None,
    progress: Callable[[str], None] = lambda message: None,
) -> EnhanceReport:
    """Run Stage A on one stem: separate, transcribe, detect, repair, splice.

    Returns the input unchanged (as loaded audio) when no region is flagged --
    this stage only ever touches a span it has evidence for.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = EnhanceReport()
    source_audio, source_sr = load_audio(input_path, target_sr=None)

    progress("Stage A / 1a - stem separation (BS-RoFormer)")
    sep = _separate_stems(input_path, work_dir, device)
    report.target_stem = sep["target_stem"]

    progress("Stage A / 1b - MIDI transcription (MuScriptor)")
    trans = _transcribe(input_path, work_dir, device)

    progress("Stage A / 1c-1e - analysis (self-similarity, key, DTW alignment)")
    analysis = _analyze(input_path, trans["midi_path"])
    report.notes.append(f"detected key: {analysis['detected_key']}")

    progress("Stage A / 2a - degradation detection")
    frame_times, risk = compute_degradation_risk(str(input_path))
    region, threshold = detect_flagged_region(risk, frame_times)
    report.region = region

    target_audio, sr_t = load_audio(input_path, target_sr=SR, mono=True)
    if region is None:
        report.notes.append("no region cleared the risk threshold -- passed through unchanged")
        report.final_audio, report.final_sr = source_audio, source_sr
        return report

    progress(f"Stage A / 2b - MIDI repair for [{region[0]:.2f}s, {region[1]:.2f}s)")
    repaired_events, full_events = _repair_midi(trans["midi_path"], region[0], region[1], device)

    progress("Stage A / 3a - re-render from donor notes")
    orig_notes = decode_notes(full_events)
    repaired_notes = decode_notes(repaired_events)
    donor_library = [n for n in orig_notes if not (region[0] <= n["onset"] < region[1])]
    target_notes = [n for n in repaired_notes if region[0] <= n["onset"] < region[1]]
    if not donor_library:
        report.notes.append(
            "no donor notes available outside the flagged region -- passed through unchanged"
        )
        report.final_audio, report.final_sr = source_audio, source_sr
        return report
    rendered_window, _placements = render_window(
        target_audio, sr_t, donor_library, target_notes, region[0], region[1]
    )

    progress("Stage A / 3b - crossfade splice + re-verify")
    final_stem = equal_power_crossfade_splice(target_audio, rendered_window, region[0], sr_t)

    frame_times_after, risk_after = compute_degradation_risk(final_stem)
    mask = (frame_times >= region[0]) & (frame_times < region[1])
    mask_after = (frame_times_after >= region[0]) & (frame_times_after < region[1])
    report.risk_in_before = float(risk[mask].mean()) if mask.any() else None
    report.risk_in_after = float(risk_after[mask_after].mean()) if mask_after.any() else None

    if (
        report.risk_in_before is None
        or report.risk_in_after is None
        or report.risk_in_after >= report.risk_in_before
    ):
        report.notes.append(
            "candidate repair did not lower measured risk -- passed through unchanged"
        )
        report.final_audio, report.final_sr = source_audio, source_sr
        return report

    report.final_audio = preserve_source_layout(
        source_audio,
        source_sr,
        target_audio,
        final_stem,
        sr_t,
    )
    report.final_sr = source_sr
    report.is_enhanced = True
    return report
