
import shutil
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import mido
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import yaml
from dtw import dtw
from ml_collections import ConfigDict
from music21 import converter as m21_converter

st.set_page_config(page_title="Enhanced Stem Workflow", page_icon="🎹", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 0.6rem;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WORK_DIR = Path("data/interim/streamlit")
WORK_DIR.mkdir(parents=True, exist_ok=True)
SR = 22050
HOP_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _col_normalize(m):
    return m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-9)


# ==================================================================================
# Cached model loaders -- loaded once per server process, reused across runs/reruns
# ==================================================================================
@st.cache_resource(show_spinner="Loading BS-RoFormer model...")
def load_bs_roformer():
    from bs_roformer.download import ensure_model_assets
    from bs_roformer.inference import SafeLoaderWithTuple
    from bs_roformer.model_registry import DEFAULT_MODEL
    from bs_roformer.utils import get_model_from_config

    model_path, config_path = ensure_model_assets(DEFAULT_MODEL)
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    model.to(DEVICE)
    return model, config


@st.cache_resource(show_spinner="Loading MuScriptor model (large)...")
def load_muscriptor():
    from muscriptor import TranscriptionModel

    return TranscriptionModel.load_model("large")


@st.cache_resource(show_spinner="Loading AMT model (stanford-crfm/music-small-800k)...")
def load_amt():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("stanford-crfm/music-small-800k")
    model.eval()
    model.to(DEVICE)
    return model


# ==================================================================================
# Pipeline steps -- each mirrors a cell in notebooks/enhanced_stem_workflow.ipynb
# ==================================================================================
def step_1a_separate_stems(input_path, out_dir):
    from bs_roformer.utils import demix_track

    model, config = load_bs_roformer()
    mix, sr = sf.read(input_path)
    mono = mix.ndim == 1
    mix_stereo = np.stack([mix, mix], axis=-1) if mono else mix
    mixture = torch.tensor(mix_stereo.T, dtype=torch.float32)

    stem_result, _ = demix_track(config, model, mixture, DEVICE)
    stem_names = list(stem_result.keys())

    stem_paths, stem_rms = {}, {}
    for name in stem_names:
        audio = stem_result[name].T
        if mono:
            audio = audio[:, 0]
        out_path = out_dir / f"stem_{name}.wav"
        sf.write(out_path, audio, sr, subtype="FLOAT")
        stem_paths[name] = out_path
        stem_rms[name] = rms(audio)

    target_stem = max(stem_names, key=lambda n: stem_rms[n])
    recon = sum(stem_result[name] for name in stem_names).T
    recon_error = rms(recon - mix_stereo) / rms(mix_stereo)
    return {
        "stem_paths": stem_paths, "stem_rms": stem_rms, "target_stem": target_stem,
        "mix_stereo": mix_stereo, "sr": sr, "recon_error": recon_error,
    }


def step_1b_transcribe(input_path, out_dir):
    muscriptor_model = load_muscriptor()
    midi_bytes = muscriptor_model.transcribe_to_midi(str(input_path))
    midi_path = out_dir / "transcription.mid"
    midi_path.write_bytes(midi_bytes)

    mid = mido.MidiFile(midi_path)
    note_on_events = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    programs = sorted({m.program for t in mid.tracks for m in t if m.type == "program_change"})
    return {"midi_path": midi_path, "n_notes": len(note_on_events), "programs": programs,
            "midi_duration": mid.length}


def step_1c_1d_1e_analysis(input_path, midi_path):
    y_mono, sr_mono = librosa.load(input_path, sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr_mono, hop_length=HOP_LENGTH)

    score = m21_converter.parse(Path(midi_path))
    detected_key = score.analyze("key")

    n_beats = int(np.ceil(score.highestTime))
    midi_chroma = np.zeros((12, n_beats))
    for n in score.flatten().notes:
        beat_idx = min(int(n.offset), n_beats - 1)
        for p in (n.pitches if hasattr(n, "pitches") else [n.pitch]):
            midi_chroma[p.pitchClass, beat_idx] += 1

    alignment = dtw(_col_normalize(chroma).T, _col_normalize(midi_chroma).T, keep_internals=True)
    corr = np.corrcoef(alignment.index1, alignment.index2)[0, 1]

    return {"chroma": chroma, "detected_key": str(detected_key),
            "key_confidence": detected_key.correlationCoefficient,
            "dtw_norm_distance": alignment.normalizedDistance, "dtw_index_corr": corr}


def compute_degradation_risk(audio_path_or_array, midi_path, sr=SR, hop_length=HOP_LENGTH):
    if isinstance(audio_path_or_array, (str, Path)):
        y, _ = librosa.load(audio_path_or_array, sr=sr, mono=True)
    else:
        y = audio_path_or_array
    audio_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(audio_chroma.shape[1]), sr=sr, hop_length=hop_length)
    n = audio_chroma.shape[1]

    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0][:n]
    rms_curve = librosa.feature.rms(y=y, hop_length=hop_length)[0][:n]
    rms_local_med = np.array([np.median(rms_curve[max(0, i - 20):i + 20]) for i in range(n)])
    rms_dropout_risk = np.clip(1 - rms_curve / (rms_local_med + 1e-6), 0, 1)
    flatness_risk = np.clip(
        (flatness - np.percentile(flatness, 50)) / (np.percentile(flatness, 95) - np.percentile(flatness, 50) + 1e-9),
        0, 1,
    )
    audio_artifact_risk = np.maximum(flatness_risk, rms_dropout_risk)

    sim = librosa.segment.recurrence_matrix(audio_chroma, mode="affinity", sym=True)
    frame_dt = hop_length / sr
    gap = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) * frame_dt
    best_match = np.where(gap > 3.0, sim, -1).max(axis=1)
    best_match[best_match < 0] = np.nan
    local_med = np.array([np.nanmedian(best_match[max(0, i - 50):i + 50]) for i in range(n)])
    pattern_break_risk = np.nan_to_num(np.clip(local_med - best_match, 0, None))
    pattern_break_risk = pattern_break_risk / (pattern_break_risk.max() + 1e-9)

    score = m21_converter.parse(Path(midi_path))
    n_beats = int(np.ceil(score.highestTime))
    midi_chroma = np.zeros((12, n_beats))
    for note in score.flatten().notes:
        beat_idx = min(int(note.offset), n_beats - 1)
        for p in (note.pitches if hasattr(note, "pitches") else [note.pitch]):
            midi_chroma[p.pitchClass, beat_idx] += 1

    alignment = dtw(_col_normalize(audio_chroma).T, _col_normalize(midi_chroma).T, keep_internals=True)
    beat_for_frame = np.full(n, -1)
    for a_idx, b_idx in zip(alignment.index1, alignment.index2):
        beat_for_frame[a_idx] = b_idx

    audio_n, midi_n = _col_normalize(audio_chroma), _col_normalize(midi_chroma)
    harmonic_agree = np.array([
        float(audio_n[:, i] @ midi_n[:, beat_for_frame[i]]) if beat_for_frame[i] >= 0 else np.nan
        for i in range(n)
    ])
    harmonic_risk = np.nan_to_num(np.clip(1 - harmonic_agree, 0, 1))

    note_density = np.array([
        sum(1 for note in score.flatten().notes if int(note.offset) == b) for b in range(n_beats)
    ])
    density_med = np.median(note_density) if len(note_density) else 0
    density_dev = np.abs(note_density - density_med) / (density_med + 1e-6)
    density_risk_per_beat = np.clip(density_dev / (density_dev.max() + 1e-9), 0, 1)
    midi_confidence_risk = np.array([
        density_risk_per_beat[beat_for_frame[i]] if beat_for_frame[i] >= 0 else 0.0
        for i in range(n)
    ])

    fused = np.mean([audio_artifact_risk, pattern_break_risk, harmonic_risk, midi_confidence_risk], axis=0)
    return frame_times, fused


def detect_flagged_region(risk, frame_times, z=1.5, min_duration=1.0, pad=0.25, merge_gap=1.0):
    threshold = risk.mean() + z * risk.std()
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


TIME_RESOLUTION, DUR_OFFSET, NOTE_OFFSET = 100, 10000, 11000


def decode_notes(events):
    notes = []
    for t, d, note in zip(events[0::3], events[1::3], events[2::3]):
        note_id = note - NOTE_OFFSET
        notes.append({"onset": t / TIME_RESOLUTION, "dur": (d - DUR_OFFSET) / TIME_RESOLUTION,
                      "pitch": note_id % 128, "instrument": note_id // 128})
    return notes


def step_2b_repair_midi(midi_path, region_start, region_end):
    from anticipation import ops, sample
    from anticipation.convert import events_to_midi, midi_to_events

    amt_model = load_amt()
    events = midi_to_events(str(midi_path))
    max_t = ops.max_time(events, seconds=True)
    before = ops.clip(events, 0, region_start, clip_duration=False)
    after = ops.clip(events, region_end, max_t, clip_duration=False)
    context = ops.sort(before + after)
    repaired_events = ops.sort(sample.generate(amt_model, region_start, region_end, inputs=context, top_p=0.95))
    return repaired_events, events


def find_donor(target_pitch, library):
    exact = [n for n in library if n["pitch"] == target_pitch]
    if exact:
        return min(exact, key=lambda n: abs(n["dur"] - 0.3)), 0
    nearest = min(library, key=lambda n: abs(n["pitch"] - target_pitch))
    return nearest, target_pitch - nearest["pitch"]


def render_window(target_audio, sr, donor_library, target_notes, window_start, window_end):
    window_len = int((window_end - window_start) * sr)
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
    fade_n = int(fade_s * sr)
    start_n = int(start_s * sr)
    end_n = start_n + len(patch)  # derive from patch's actual length, not a second independent
                                   # int(end_s * sr) rounding -- the two can disagree by 1 sample
    t = np.linspace(0, np.pi / 2, fade_n)
    fade_out, fade_in = np.cos(t), np.sin(t)
    out = base.copy()
    a_seg, b_seg = base[start_n:start_n + fade_n], patch[:fade_n]
    out[start_n:start_n + fade_n] = a_seg * fade_out[:len(a_seg)] + b_seg * fade_in[:len(a_seg)]
    out[start_n + fade_n:end_n - fade_n] = patch[fade_n:len(patch) - fade_n]
    a_seg, b_seg = patch[len(patch) - fade_n:], base[end_n - fade_n:end_n]
    out[end_n - fade_n:end_n] = a_seg * fade_out[:len(a_seg)] + b_seg * fade_in[:len(a_seg)]
    return out


# ==================================================================================
# Full pipeline runner -- executes every step and streams progress via st.status
# ==================================================================================
def run_pipeline(input_path, clean_reference_path, out_dir):
    result = {"clean_reference_path": clean_reference_path}

    with st.status("Step 1a — Stem separation (BS-RoFormer)", expanded=True) as status:
        st.write("Goal: split the audio into separate instrument stems and identify the dominant one (TARGET_STEM).")
        t0 = time.time()
        r = step_1a_separate_stems(input_path, out_dir)
        result.update(r)
        st.write(f"Stems: {list(r['stem_rms'].keys())}")
        st.write(f"**TARGET_STEM: `{r['target_stem']}`** (RMS {r['stem_rms'][r['target_stem']]:.4f})")
        st.write(f"Reconstruction error (sum of stems vs original): {100*r['recon_error']:.2f}%")
        status.update(label=f"Step 1a — Stem separation: done ({time.time()-t0:.0f}s), TARGET_STEM = {r['target_stem']}", state="complete")

    with st.status("Step 1b — MIDI transcription (MuScriptor)", expanded=True) as status:
        st.write("Goal: transcribe the full audio to MIDI. Runs independently, in parallel with Step 1a.")
        t0 = time.time()
        r = step_1b_transcribe(input_path, out_dir)
        result.update(r)
        st.write(f"Notes: {r['n_notes']}, instruments (GM program): {r['programs']}, "
                 f"MIDI duration: {r['midi_duration']:.1f}s")
        status.update(label=f"Step 1b — MIDI transcription: done ({time.time()-t0:.0f}s), {r['n_notes']} notes", state="complete")

    with st.status("Step 1c/1d/1e — Audio analysis + harmony + time alignment", expanded=True) as status:
        st.write("Self-similarity matrix (librosa), harmonic analysis (music21), DTW audio↔MIDI alignment.")
        t0 = time.time()
        r = step_1c_1d_1e_analysis(input_path, result["midi_path"])
        result.update(r)
        st.write(f"Detected key: **{r['detected_key']}** (confidence {r['key_confidence']:.2f})")
        st.write(f"DTW: normalizedDistance={r['dtw_norm_distance']:.3f}, "
                 f"beat↔audio-time correlation={r['dtw_index_corr']:.3f}")
        status.update(label=f"Step 1c/1d/1e — Analysis: done ({time.time()-t0:.0f}s), key={r['detected_key']}", state="complete")

    with st.status("Step 2a — Degradation detection (4-signal fusion)", expanded=True) as status:
        st.write("Fuses audio-artifact + pattern-break + harmonic-agreement + MIDI-density into one risk curve, then auto-flags the suspected degraded region.")
        t0 = time.time()
        frame_times, risk = compute_degradation_risk(str(input_path), str(result["midi_path"]))
        region, threshold = detect_flagged_region(risk, frame_times)
        result["frame_times_before"], result["risk_before"] = frame_times, risk
        result["region"], result["threshold"] = region, threshold
        if region:
            st.write(f"Flagged region: **[{region[0]:.2f}s, {region[1]:.2f}s)** (risk threshold {threshold:.3f})")
            status.update(label=f"Step 2a — Detection: done ({time.time()-t0:.0f}s), region [{region[0]:.1f}s, {region[1]:.1f}s)", state="complete")
        else:
            st.write("No region exceeded the risk threshold -- audio looks fine, no repair needed.")
            status.update(label="Step 2a — Detection: no clear degraded region found", state="complete")

    if region is None:
        result["stopped_early"] = True
        return result

    with st.status("Step 2b — MIDI repair (AMT infilling)", expanded=True) as status:
        st.write("Removes the MIDI segment in the flagged region and regenerates it with AMT (conditioned on both past and future context).")
        t0 = time.time()
        repaired_events, full_events = step_2b_repair_midi(result["midi_path"], region[0], region[1])
        result["repaired_events"], result["full_events"] = repaired_events, full_events
        repaired_midi_path = out_dir / "repaired.mid"
        from anticipation.convert import events_to_midi
        events_to_midi(repaired_events).save(str(repaired_midi_path))
        result["repaired_midi_path"] = repaired_midi_path
        st.write(f"Regenerated MIDI for the flagged region, saved to `{repaired_midi_path}`.")
        status.update(label=f"Step 2b — MIDI repair: done ({time.time()-t0:.0f}s)", state="complete")

    with st.status("Step 3a — Re-render with real audio (concatenative)", expanded=True) as status:
        st.write(f"For each repaired note, finds a REAL note of the same pitch in the `{result['target_stem']}` stem, "
                 "pitch-shifts/time-stretches it, and splices it into place.")
        t0 = time.time()
        target_audio, sr_t = librosa.load(result["stem_paths"][result["target_stem"]], sr=SR, mono=True)
        orig_notes = decode_notes(full_events)
        repaired_notes = decode_notes(repaired_events)
        donor_library = [n for n in orig_notes if not (region[0] <= n["onset"] < region[1])]
        target_notes = [n for n in repaired_notes if region[0] <= n["onset"] < region[1]]
        rendered_window, placements = render_window(target_audio, sr_t, donor_library, target_notes, region[0], region[1])
        result.update({"target_audio": target_audio, "sr_t": sr_t, "rendered_window": rendered_window, "placements": placements})
        st.write(f"Rendered {len(placements)}/{len(target_notes)} target notes.")
        status.update(label=f"Step 3a — Render: done ({time.time()-t0:.0f}s), {len(placements)} notes", state="complete")

    with st.status("Step 3b — Crossfade splice + re-verification", expanded=True) as status:
        st.write("Splices the rendered segment into the stem with an equal-power crossfade, then reruns the detector to confirm risk actually dropped.")
        t0 = time.time()
        final_stem = equal_power_crossfade_splice(target_audio, rendered_window, region[0], sr_t)
        final_stem_path = out_dir / "final_stem.wav"
        sf.write(final_stem_path, final_stem, sr_t)
        result["final_stem"], result["final_stem_path"] = final_stem, final_stem_path

        frame_times_after, risk_after = compute_degradation_risk(final_stem, result["repaired_midi_path"])
        result["frame_times_after"], result["risk_after"] = frame_times_after, risk_after

        mask = (frame_times >= region[0]) & (frame_times < region[1])
        mask_after = (frame_times_after >= region[0]) & (frame_times_after < region[1])
        risk_in_before = risk[mask].mean()
        risk_in_after = risk_after[mask_after].mean()
        result["risk_in_before"], result["risk_in_after"] = risk_in_before, risk_in_after
        improvement = 100 * (risk_in_before - risk_in_after) / risk_in_before if risk_in_before else 0
        result["improvement"] = improvement

        trend = "decreased" if improvement > 0 else "increased"
        st.write(f"Risk in the repaired region: **{risk_in_before:.3f} → {risk_in_after:.3f}** ({trend} {abs(improvement):.0f}%)")
        status.update(label=f"Step 3b — Splice + verify: done ({time.time()-t0:.0f}s), risk {trend} {abs(improvement):.0f}%", state="complete")

    result["stopped_early"] = False
    return result


# ==================================================================================
# UI
# ==================================================================================
st.title("🎹 Enhanced Stem Workflow")
st.caption(
    "Interactive demo of `docs/enhanced-stem-workflow.docx`: stem separation, degradation "
    "detection, MIDI repair, and re-render with real audio. Each step shows its result as soon "
    "as it finishes."
)

with st.sidebar:
    st.header("⚙️ Settings")

    if DEVICE.type == "cuda":
        st.success(f"🚀 GPU detected\n\n**{torch.cuda.get_device_name(0)}**")
    else:
        st.warning("🖥️ No GPU detected -- running on CPU")

    st.divider()

    degrade_choice = st.radio(
        "Degradation handling",
        ["Inject simulated degradation (corrupt a section)", "Use as-is (no degradation)"],
        help="Choose 'Inject simulated degradation' if the audio is clean and you want to see the "
             "pipeline detect and repair a self-inflicted glitch. Choose 'Use as-is' if the audio is "
             "a real Suno stem that may already be degraded on its own.",
    )
    max_seconds = st.number_input(
        "Clip length limit (seconds)", min_value=10, max_value=600, value=90, step=10,
        help="Shorter clips run faster -- BS-RoFormer is the slowest step, especially on CPU.",
    )

    st.divider()
    st.caption(
        "A ~60-90s clip takes roughly 10-20 minutes end to end on CPU (BS-RoFormer alone is the "
        "majority of that). A GPU speeds this up substantially."
    )

with st.container(border=True):
    st.subheader("1. Upload audio")
    uploaded = st.file_uploader(
        "Input audio", type=["wav", "mp3", "flac", "m4a", "ogg"], label_visibility="collapsed"
    )
    if uploaded is not None:
        st.audio(uploaded)
    run_clicked = st.button(
        "▶ Run pipeline", type="primary", disabled=uploaded is None, use_container_width=True
    )

if uploaded is None:
    st.info("👆 Upload an audio file to get started.")
    st.stop()

if run_clicked:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = WORK_DIR / f"uploaded{Path(uploaded.name).suffix}"
    raw_path.write_bytes(uploaded.getvalue())

    y_raw, sr_raw = librosa.load(raw_path, sr=None, mono=False)
    if y_raw.ndim == 1:
        y_raw = y_raw[np.newaxis, :]
    n_samples = min(y_raw.shape[1], int(max_seconds * sr_raw))
    y_raw = y_raw[:, :n_samples]

    clean_reference_path = WORK_DIR / "clean_reference.wav"
    sf.write(clean_reference_path, y_raw.T if y_raw.shape[0] > 1 else y_raw[0], sr_raw)

    if degrade_choice.startswith("Inject simulated degradation"):
        degraded = y_raw.copy()
        duration_s = n_samples / sr_raw
        d_start, d_end = duration_s * 0.30, duration_s * 0.38
        lo, hi = int(d_start * sr_raw), int(d_end * sr_raw)
        rng = np.random.default_rng(0)
        for ch in range(degraded.shape[0]):
            degraded[ch, lo:hi] = degraded[ch, lo:hi] * 0.2 + rng.normal(0, 0.3, hi - lo)
        input_path = WORK_DIR / "pipeline_input.wav"
        sf.write(input_path, degraded.T if degraded.shape[0] > 1 else degraded[0], sr_raw)
        st.info(f"Injected simulated degradation at [{d_start:.1f}s, {d_end:.1f}s) and saved it as the pipeline input.")
    else:
        input_path = clean_reference_path
        clean_reference_path = None  # no known-clean baseline when using the file as-is

    st.session_state["pipeline_result"] = run_pipeline(input_path, clean_reference_path, WORK_DIR)
    st.session_state["input_path"] = input_path

# ==================================================================================
# Display results (persists across reruns once computed)
# ==================================================================================
result = st.session_state.get("pipeline_result")
if result is None:
    st.stop()

st.divider()
st.header("📊 Results")

if result.get("stopped_early"):
    st.success("✅ No clear degradation detected -- audio looks fine, pipeline stopped after Step 2a.")
    st.audio(str(st.session_state["input_path"]))
    st.stop()

region = result["region"]
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target stem", result["target_stem"])
    col2.metric("Repaired region", f"{region[0]:.1f}s – {region[1]:.1f}s")
    col3.metric("Risk before repair", f"{result['risk_in_before']:.3f}")
    col4.metric("Risk after repair", f"{result['risk_in_after']:.3f}",
                delta=f"{-result['improvement']:.0f}%", delta_color="inverse")

st.subheader("🎧 Listen")
with st.container(border=True):
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Original / reference**")
        st.audio(str(result.get("clean_reference_path") or st.session_state["input_path"]))
    with p2:
        st.markdown("**Pipeline input** (degraded if simulated)")
        st.audio(str(st.session_state["input_path"]))
    with p3:
        st.markdown("**Repaired** (`final_stem.wav`)")
        st.audio(str(result["final_stem_path"]))


def plot_waveform(ax, y, sr, title, region=None):
    t = np.arange(len(y)) / sr
    ax.plot(t, y, linewidth=0.4, color="#4C78A8")
    if region:
        ax.axvspan(region[0], region[1], color="red", alpha=0.15)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1, 1)


tab_wave, tab_risk = st.tabs(["🌊 Waveform", "⚠️ Risk curve"])

with tab_wave:
    y_input, _ = librosa.load(st.session_state["input_path"], sr=SR, mono=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.5), sharex=True)
    plot_waveform(axes[0], y_input, SR, "Pipeline input", region=region)
    plot_waveform(axes[1], result["final_stem"], result["sr_t"], "Repaired (final_stem.wav)", region=region)
    axes[-1].set_xlabel("seconds")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with tab_risk:
    fig2, ax = plt.subplots(figsize=(11, 3))
    ax.plot(result["frame_times_before"], result["risk_before"], label="Before repair", color="#E45756")
    ax.plot(result["frame_times_after"], result["risk_after"], label="After repair", color="#54A24B")
    ax.axhline(result["threshold"], color="gray", linestyle="--", linewidth=1, label=f"threshold ({result['threshold']:.2f})")
    ax.axvspan(region[0], region[1], color="red", alpha=0.12, label="repaired region")
    ax.set_xlabel("seconds")
    ax.set_ylabel("risk [0,1]")
    ax.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)

st.caption(
    "Pipeline payoff: risk in the repaired region should drop clearly (green line lower than the "
    "red line, inside the shaded region). Don't expect risk to hit zero -- AMT generates new "
    "music, it doesn't reconstruct the exact original."
)
