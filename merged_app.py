"""Interactive runner for the merged restoration pipeline.

Stage A (midi_repair, ported from rewrite-v2) -> Stage B (suno_restore).
Takes one uploaded stem, not a folder -- Stage A only makes sense on a single
file it can transcribe and analyze as one piece. Orchestration lives in
`full_pipeline.restore_from_stem`; this module only renders it, same
separation `streamlit_app.py` (the Stage-B-only UI) already uses.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import full_pipeline
from suno_restore import bandwidth, tempo
from suno_restore.audio_io import FFmpegNotFound, load_audio
from suno_restore.metrics import energy_above_hz, spectral_cliff_hz

DEFAULT_OUTPUT = Path("data/interim/restored_merged")
DEFAULT_APOLLO = Path("vendor/Apollo")
UPLOAD_DIR = Path("data/interim/uploaded_merged")

st.set_page_config(page_title="Merged Stem Restoration", layout="wide")
st.title("Merged Stem Restoration")
st.caption(
    "Stage A — localized MIDI repair (stem separation → transcribe → detect → "
    "AMT infill → donor re-render). Stage B — whole-track tempo/denoise/"
    "bandwidth correction. Stage A runs first and only touches a span it has "
    "evidence for; Stage B always runs on the full result."
)

WAVE_BG = "#0d0d0d"
WAVE_PANEL = "#141414"
WAVE_BAR = "#e8833c"
WAVE_GHOST = "#4a3320"
WAVE_TEXT = "#f2f0ed"
WAVE_TEXT_2 = "#9a9691"
N_BARS = 260
LINE_ORIGINAL = "#3987e5"
LINE_RESTORED = "#d95926"


def format_hz(value: float | None) -> str:
    return f"{value:.0f}Hz" if value else "none"


def envelope(audio: np.ndarray, bars: int = N_BARS) -> np.ndarray:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    edges = np.linspace(0, len(mono), bars + 1, dtype=int)
    return np.array([
        np.abs(mono[lo:hi]).max() if hi > lo else 0.0
        for lo, hi in zip(edges, edges[1:])
    ])


def waveform_figure(pairs: list[tuple[str, np.ndarray, int]]):
    reference = envelope(pairs[0][1])
    peak = reference.max() or 1.0
    duration = pairs[0][1].shape[0] / pairs[0][2]
    x = np.arange(N_BARS)

    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 2.1 * len(pairs)), facecolor=WAVE_BG)
    axes = np.atleast_1d(axes)
    for index, (ax, (label, audio, _sr)) in enumerate(zip(axes, pairs)):
        env = envelope(audio) / peak
        ax.set_facecolor(WAVE_PANEL)
        if index > 0:
            ax.bar(x, reference / peak, width=0.62, bottom=-(reference / peak) / 2,
                   color=WAVE_GHOST, linewidth=0)
        ax.bar(x, env, width=0.62, bottom=-env / 2, color=WAVE_BAR, linewidth=0)
        ax.set_xlim(-2, N_BARS + 1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ticks = np.linspace(0, N_BARS, 7)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / N_BARS * duration:.0f}s" for t in ticks], color=WAVE_TEXT_2, fontsize=8)
        ax.tick_params(length=0, pad=4)
        ax.text(0, 1.16, label, transform=ax.transAxes, color=WAVE_TEXT, fontsize=11, fontweight="bold", va="top")
        if index > 0:
            ax.text(1, 1.16, "dim bars = original", transform=ax.transAxes, color=WAVE_TEXT_2, fontsize=8.5, va="top", ha="right")
    fig.tight_layout(h_pad=2.2)
    return fig


def spectrum_figure(pairs: list[tuple[str, np.ndarray, int]]):
    import librosa

    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=WAVE_BG)
    ax.set_facecolor(WAVE_PANEL)
    for (label, audio, sr), color in zip(pairs, (LINE_ORIGINAL, LINE_RESTORED)):
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        stft = np.abs(librosa.stft(mono, n_fft=8192, hop_length=2048))
        spec = np.percentile(stft, 95, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        ax.semilogx(freqs, 20 * np.log10(spec + 1e-12), label=label, linewidth=1.6, color=color)
    ax.set_xlim(100, 24000)
    ax.set_xlabel("Hz", color=WAVE_TEXT_2)
    ax.set_ylabel("dB", color=WAVE_TEXT_2)
    ax.tick_params(colors=WAVE_TEXT_2, labelsize=8)
    ax.grid(alpha=0.12, linewidth=0.6, color=WAVE_TEXT_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(WAVE_TEXT_2)
    legend = ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    for text in legend.get_texts():
        text.set_color(WAVE_TEXT_2)
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload one stem",
        type=["wav", "mp3", "flac", "aiff", "aif", "ogg", "m4a"],
        help="A single audio file: a stem exported by Suno, or one of V1's 9 separated stems.",
    )
    output_dir = Path(st.text_input("Output folder", str(DEFAULT_OUTPUT)))

    st.header("Stage A — localized repair")
    do_stage_a = st.checkbox(
        "Run Stage A", value=True,
        help="Stem separation, MIDI transcription, degradation detection, AMT infill, donor re-render. "
             "Turn off to send the raw upload straight to Stage B.",
    )

    st.header("Stage B — whole-track correction")
    do_tempo = st.checkbox("1 — Tempo correction", value=True)
    do_denoise = st.checkbox("2 — Denoise", value=True)
    do_bandwidth = st.checkbox("3 — Bandwidth extension", value=True)

    apollo_repo = Path(st.text_input("Apollo checkout", str(DEFAULT_APOLLO)))

    cuda = torch.cuda.is_available()
    device = st.selectbox("Device", ["cuda", "cpu"] if cuda else ["cpu"])
    st.caption(f"GPU: **{torch.cuda.get_device_name(0)}**" if cuda else "No CUDA device detected")
    if cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        st.caption(f"VRAM free: {free_b / 1e9:.1f} GB / {total_b / 1e9:.1f} GB")

    run = st.button("Run merged pipeline", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload a stem in the sidebar, then run.")
    st.stop()

upload_dir = Path(UPLOAD_DIR)
shutil.rmtree(upload_dir, ignore_errors=True)
upload_dir.mkdir(parents=True, exist_ok=True)
input_path = upload_dir / uploaded.name
input_path.write_bytes(uploaded.getbuffer())
st.caption(f"Using uploaded **{uploaded.name}**")

with st.expander("What each stage does, and why", expanded=False):
    st.markdown(
        """
**Stage A — localized repair** assumes at most one contiguous damaged span exists in the stem.
It separates the stem (BS-RoFormer), transcribes it to MIDI (MuScriptor), flags a candidate
region by spectral flatness, regenerates that region's MIDI (AMT infilling), then re-renders it
from real donor notes found elsewhere in the *same* stem and crossfades it back in. If nothing
is flagged, Stage A changes nothing.

**Stage B — whole-track correction** assumes three specific defects (tempo drift, hiss/crackle,
bandwidth loss) are present across the whole stem, confirmed against Suno's own documented
output — and corrects all three unconditionally, regardless of what Stage A did or didn't find.

**Order.** Stage A always runs first: its degradation detector is calibrated on the flatness of
the *input* signal, and running Stage B first would denoise/bandwidth-extend the stem before
Stage A ever measures it.

**Sample rate.** Stage A works at 22050Hz internally; its output is resampled to 48kHz before
Stage B, which requires 48kHz throughout.

**GPU memory.** Stage A's and Stage B's models are unloaded between stages so only one stage's
models occupy VRAM at a time — the two together were sized individually for an 8GB card, not
tested resident simultaneously.
"""
    )

if not run:
    st.stop()

try:
    load_audio(input_path)
except FFmpegNotFound as error:
    st.error(str(error))
    st.stop()

stage_a_status = st.status(
    "Stage A — localized MIDI repair" if do_stage_a else "Stage A — skipped",
    expanded=do_stage_a,
)
tempo_status = st.status(
    "Stage B / Step 1 — Tempo correction" if do_tempo else "Stage B / Step 1 — Tempo correction (skipped)",
    expanded=False,
)
stem_status = st.status("Stage B / Steps 2-3 — Restoring", expanded=True)
log_area = stage_a_status.empty()
log_lines: list[str] = []


def on_progress(message: str) -> None:
    log_lines.append(message)
    with stage_a_status:
        st.write(message)


def render_tempo(analysis: tempo.TempoAnalysis) -> None:
    with tempo_status:
        st.write(f"**{analysis.summary()}**")
    tempo_status.update(label=f"Stage B / Step 1 — {analysis.summary()}", state="complete")


completed = []


def render_stem(entry) -> None:
    completed.append(entry)
    stem_status.update(label=f"Stage B / Steps 2-3 — {len(completed)} stem restored", state="complete")


try:
    combined = full_pipeline.restore_from_stem(
        input_path,
        output_dir,
        apollo_repo=apollo_repo,
        device=device,
        do_stage_a=do_stage_a,
        do_tempo=do_tempo,
        do_denoise=do_denoise,
        do_bandwidth=do_bandwidth,
        progress=on_progress,
        on_tempo=render_tempo,
        on_stem=render_stem,
    )
except tempo.NoReferenceStem as error:
    tempo_status.update(label="Stage B / Step 1 — no usable reference stem", state="error")
    st.error(str(error))
    st.stop()
except bandwidth.ApolloUnavailable as error:
    stem_status.update(label="Stage B / Steps 2-3 — Apollo unavailable", state="error")
    st.error(str(error))
    st.stop()

stage_a_status.update(
    label="Stage A — done" if do_stage_a else "Stage A — skipped",
    state="complete",
)

st.subheader("Stage A result")
if combined.stage_a is None:
    st.info("Stage A was skipped; Stage B ran on the raw upload.")
else:
    a = combined.stage_a
    cols = st.columns(4)
    cols[0].metric("Target stem", a.target_stem or "—")
    cols[1].metric("Flagged region", f"{a.region[0]:.1f}s–{a.region[1]:.1f}s" if a.region else "none")
    cols[2].metric("Risk before", f"{a.risk_in_before:.3f}" if a.risk_in_before is not None else "—")
    cols[3].metric(
        "Risk after", f"{a.risk_in_after:.3f}" if a.risk_in_after is not None else "—",
        delta=(f"{-(100 * (a.risk_in_before - a.risk_in_after) / a.risk_in_before):.0f}%"
               if a.risk_in_before else None),
        delta_color="inverse",
    )
    if a.notes:
        st.caption(" · ".join(a.notes))

st.subheader("Stage B result")
for name, entry in combined.stage_b.stems.items():
    if entry.output_path is None:
        continue
    with st.expander(f"{name} → `{entry.output_path.name}`", expanded=True):
        st.audio(str(entry.output_path))
        restored_audio, restored_sr = load_audio(entry.output_path, target_sr=None)
        original_audio, original_sr = load_audio(input_path)
        pairs = [
            ("Uploaded input", original_audio, original_sr),
            ("Restored (Stage A + B)" if do_stage_a else "Restored (Stage B only)", restored_audio, restored_sr),
        ]
        st.pyplot(waveform_figure(pairs))
        st.pyplot(spectrum_figure(pairs))
        st.caption(
            f"Cutoff {format_hz(entry.cliff_before_hz)} → {format_hz(entry.cliff_after_hz)} · "
            f"energy >13kHz {entry.energy_before_pct:.1f}% → {entry.energy_after_pct:.1f}%"
        )

st.success(f"Wrote restored output to `{output_dir}`")
