"""Streamlit UI for the combined Version 2 -> Version 3 restoration flow."""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import full_pipeline
from suno_restore import bandwidth, tempo
from suno_restore.audio_io import FFmpegNotFound, load_audio

DEFAULT_OUTPUT = Path("data/interim/restored_merged")
DEFAULT_APOLLO = Path("vendor/Apollo")
UPLOAD_DIR = Path("data/interim/uploaded_merged")

WAVE_BG = "#0d0d0d"
WAVE_PANEL = "#141414"
WAVE_BAR = "#e8833c"
WAVE_GHOST = "#4a3320"
WAVE_TEXT = "#f2f0ed"
WAVE_TEXT_2 = "#9a9691"
LINE_ORIGINAL = "#3987e5"
LINE_VERSION_2 = "#e59b39"
LINE_VERSION_3 = "#d95926"
N_BARS = 260


def format_hz(value: float | None) -> str:
    return f"{value:.0f}Hz" if value else "none"


def format_pct(value: float | None) -> str:
    return f"{value:.2f}%" if value is not None else "—"


def channel_count(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def envelope(audio: np.ndarray, bars: int = N_BARS) -> np.ndarray:
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    edges = np.linspace(0, len(mono), bars + 1, dtype=int)
    return np.array(
        [np.abs(mono[lo:hi]).max() if hi > lo else 0.0 for lo, hi in zip(edges, edges[1:])]
    )


def waveform_figure(pairs: list[tuple[str, np.ndarray, int]]):
    """Draw Original, V2, and V3 on one shared amplitude scale."""
    reference = envelope(pairs[0][1])
    peak = reference.max() or 1.0
    duration = pairs[0][1].shape[0] / pairs[0][2]
    x = np.arange(N_BARS)
    fig, axes = plt.subplots(
        len(pairs),
        1,
        figsize=(11, 2.0 * len(pairs)),
        facecolor=WAVE_BG,
    )
    axes = np.atleast_1d(axes)

    for index, (ax, (label, audio, _sr)) in enumerate(zip(axes, pairs)):
        env = envelope(audio) / peak
        ax.set_facecolor(WAVE_PANEL)
        if index > 0:
            ax.bar(
                x,
                reference / peak,
                width=0.62,
                bottom=-(reference / peak) / 2,
                color=WAVE_GHOST,
                linewidth=0,
            )
        ax.bar(x, env, width=0.62, bottom=-env / 2, color=WAVE_BAR, linewidth=0)
        ax.set_xlim(-2, N_BARS + 1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ticks = np.linspace(0, N_BARS, 7)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{t / N_BARS * duration:.0f}s" for t in ticks],
            color=WAVE_TEXT_2,
            fontsize=8,
        )
        ax.tick_params(length=0, pad=4)
        ax.text(
            0,
            1.16,
            label,
            transform=ax.transAxes,
            color=WAVE_TEXT,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        if index > 0:
            ax.text(
                1,
                1.16,
                "dim bars = original",
                transform=ax.transAxes,
                color=WAVE_TEXT_2,
                fontsize=8.5,
                va="top",
                ha="right",
            )
    fig.tight_layout(h_pad=2.2)
    return fig


def spectrum_figure(pairs: list[tuple[str, np.ndarray, int]]):
    """Draw all available version spectra on one axis."""
    import librosa

    colors = {
        "Original": LINE_ORIGINAL,
        "Version 2": LINE_VERSION_2,
        "Version 3": LINE_VERSION_3,
    }
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=WAVE_BG)
    ax.set_facecolor(WAVE_PANEL)
    for label, audio, sr in pairs:
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        stft = np.abs(librosa.stft(mono, n_fft=8192, hop_length=2048))
        spec = np.percentile(stft, 95, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        ax.semilogx(
            freqs,
            20 * np.log10(spec + 1e-12),
            label=label,
            linewidth=1.6,
            color=colors[label],
        )
    ax.set_xlim(100, max(101, min(24000, max(sr for _, _, sr in pairs) / 2)))
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


st.set_page_config(page_title="V2 + V3 Audio Restoration", layout="wide")
st.title("V2 + V3 Audio Restoration")
st.caption(
    "One file flows through Version 2 localized MIDI repair, then the enhanced "
    "Version 3 gated and verified restoration. Both version outputs are retained "
    "and shown separately."
)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload one audio file",
        type=["wav", "mp3", "flac", "aiff", "aif", "ogg", "m4a"],
        accept_multiple_files=False,
        help="The combined flow currently handles exactly one file per run.",
    )
    output_dir = Path(st.text_input("Output folder", str(DEFAULT_OUTPUT)))

    st.header("Version 2 — localized repair")
    do_version_2 = st.checkbox(
        "Run Version 2",
        value=True,
        help=(
            "Detect one damaged span, regenerate its MIDI, and re-render it from "
            "donor notes while preserving the upload's rate and channels."
        ),
    )

    st.header("Version 3 — verified restoration")
    do_tempo = st.checkbox("Tempo correction", value=True)
    do_denoise = st.checkbox("Denoise", value=True)
    do_bandwidth = st.checkbox("Bandwidth extension", value=True)
    apollo_repo = Path(st.text_input("Apollo checkout", str(DEFAULT_APOLLO)))

    cuda = torch.cuda.is_available()
    device = st.selectbox("Device", ["cuda", "cpu"] if cuda else ["cpu"])
    st.caption(f"GPU: **{torch.cuda.get_device_name(0)}**" if cuda else "No CUDA device detected")
    if cuda:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        st.caption(f"VRAM free: {free_bytes / 1e9:.1f} GB / {total_bytes / 1e9:.1f} GB")

    run = st.button(
        "Run V2 + V3",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

if uploaded is None:
    st.info("Upload one audio file in the sidebar to begin.")
    st.stop()

upload_dir = Path(UPLOAD_DIR)
shutil.rmtree(upload_dir, ignore_errors=True)
upload_dir.mkdir(parents=True, exist_ok=True)
uploaded_name = Path(uploaded.name).name
input_path = upload_dir / uploaded_name
input_path.write_bytes(uploaded.getbuffer())

st.subheader("Original input")
st.caption(f"`{input_path.name}`")
st.audio(str(input_path))

with st.expander("How the renewed merge works", expanded=False):
    st.markdown(
        """
**Version 2 — localized MIDI repair** analyzes the file at 22.05kHz mono, but only its
repair delta is applied back to the native upload. This keeps the original sample rate,
channel layout, stereo difference, and untouched high-frequency content. If its candidate
repair does not reduce measured risk, Version 2 rolls back to the original audio.

**Version 3 — gated and verified restoration** receives the Version 2 artifact at that
native rate and channel layout. Tempo, denoise, and bandwidth extension run only when the
damage gates justify them. The wet result is blended conservatively and is replaced by the
dry Version 2 signal if verification fails.

Outputs are kept separately under **`v2/`** and **`v3/`**, so the result of either version
can be auditioned independently.
"""
    )

if not run:
    st.info("Choose the versions and steps in the sidebar, then run.")
    st.stop()

try:
    load_audio(input_path, target_sr=None)
except FFmpegNotFound as error:
    st.error(str(error))
    st.stop()

version_2_status = st.status(
    "Version 2 — localized MIDI repair" if do_version_2 else "Version 2 — skipped",
    expanded=do_version_2,
)
version_3_status = st.status("Version 3 — gated and verified restoration", expanded=True)
tempo_status = st.status("Version 3 / Tempo — evaluating", expanded=False)


def on_progress(message: str) -> None:
    target = (
        version_2_status
        if message.startswith(("Version 2", "Stage A", "Releasing Version 2"))
        else version_3_status
    )
    with target:
        st.write(message)


def render_tempo(analysis: tempo.TempoAnalysis) -> None:
    with tempo_status:
        st.write(f"**{analysis.summary()}**")
    tempo_status.update(label=f"Version 3 / Tempo — {analysis.summary()}", state="complete")


completed = []


def render_file(entry) -> None:
    completed.append(entry)
    version_3_status.update(label="Version 3 — output written")


try:
    combined = full_pipeline.restore_from_stem(
        input_path,
        output_dir,
        apollo_repo=apollo_repo,
        device=device,
        do_stage_a=do_version_2,
        do_tempo=do_tempo,
        do_denoise=do_denoise,
        do_bandwidth=do_bandwidth,
        progress=on_progress,
        on_tempo=render_tempo,
        on_stem=render_file,
    )
except tempo.NoReferenceStem as error:
    tempo_status.update(label="Version 3 / Tempo — no usable reference", state="error")
    st.error(str(error))
    st.stop()
except bandwidth.ApolloUnavailable as error:
    version_3_status.update(label="Version 3 — Apollo unavailable", state="error")
    st.error(str(error))
    st.stop()
except ModuleNotFoundError as error:
    version_2_status.update(label="Version 2 — dependency missing", state="error")
    st.error(
        f"Version 2 needs the missing package `{error.name}`. "
        "Install the merged dependencies with `pip install -r requirements.txt`."
    )
    st.stop()

version_2_status.update(
    label="Version 2 — complete" if do_version_2 else "Version 2 — skipped",
    state="complete",
)
if not do_tempo:
    # `render_tempo` only fires once the warp has been computed, so an unticked
    # step would otherwise leave this spinner running for the rest of the run.
    tempo_status.update(label="Version 3 / Tempo — not requested", state="complete")
version_3_status.update(label="Version 3 — complete", state="complete")

original_audio, original_sr = load_audio(combined.input_path, target_sr=None)
version_3_audio, version_3_sr = load_audio(combined.version_3_path, target_sr=None)
comparison = [("Original", original_audio, original_sr)]

st.subheader("Audition each version")
original_col, version_2_col, version_3_col = st.columns(3)
with original_col:
    st.markdown("#### Original")
    st.audio(str(combined.input_path))
    st.caption(f"{original_sr}Hz · {channel_count(original_audio)} channel(s)")

version_2_audio = None
version_2_sr = None
with version_2_col:
    st.markdown("#### Version 2 output")
    if combined.version_2_path is None:
        st.info("Version 2 was skipped.")
    else:
        version_2_audio, version_2_sr = load_audio(combined.version_2_path, target_sr=None)
        st.audio(str(combined.version_2_path))
        st.caption(
            f"`v2/{combined.version_2_path.name}` · {version_2_sr}Hz · "
            f"{channel_count(version_2_audio)} channel(s)"
        )
        comparison.append(("Version 2", version_2_audio, version_2_sr))

with version_3_col:
    st.markdown("#### Version 3 output")
    st.audio(str(combined.version_3_path))
    st.caption(
        f"`v3/{combined.version_3_path.name}` · {version_3_sr}Hz · "
        f"{channel_count(version_3_audio)} channel(s)"
    )

comparison.append(("Version 3", version_3_audio, version_3_sr))
st.pyplot(waveform_figure(comparison))
st.pyplot(spectrum_figure(comparison))

st.subheader("Version 2 decision")
if combined.stage_a is None:
    st.info("Skipped — Version 3 received the original upload.")
else:
    version_2 = combined.stage_a
    cols = st.columns(4)
    cols[0].metric("Outcome", "Enhanced" if version_2.is_enhanced else "Unchanged")
    cols[1].metric("Target stem", version_2.target_stem or "—")
    cols[2].metric(
        "Flagged region",
        f"{version_2.region[0]:.1f}s–{version_2.region[1]:.1f}s" if version_2.region else "none",
    )
    cols[3].metric(
        "Risk before → after",
        (
            f"{version_2.risk_in_before:.3f} → {version_2.risk_in_after:.3f}"
            if version_2.risk_in_before is not None and version_2.risk_in_after is not None
            else "—"
        ),
    )
    if version_2.notes:
        st.caption(" · ".join(version_2.notes))

st.subheader("Version 3 result")
version_3_entry = next(iter(combined.stage_b.stems.values()))
steps_run = [
    label for label, ticked in
    (("tempo", do_tempo), ("denoise", do_denoise), ("bandwidth", do_bandwidth))
    if ticked
]
cols = st.columns(4)
cols[0].metric("Steps run", " + ".join(steps_run) if steps_run else "none")
cols[1].metric(
    "Channels",
    f"{version_3_entry.input_channels} → {version_3_entry.output_channels}",
)
cols[2].metric("Sample rate", f"{version_3_entry.output_sr}Hz")
cols[3].metric(
    "Cutoff",
    f"{format_hz(version_3_entry.cliff_before_hz)} → {format_hz(version_3_entry.cliff_after_hz)}",
)
st.caption(
    f"Energy above 13kHz: {format_pct(version_3_entry.energy_before_pct)} → "
    f"{format_pct(version_3_entry.energy_after_pct)}"
)
if combined.stage_b.tempo_note:
    st.caption(f"Tempo: {combined.stage_b.tempo_note}")
if version_3_entry.denoise_note:
    st.caption(f"Denoise: {version_3_entry.denoise_note}")

# Every enabled step ran at full strength, so "it ran" is not the same as "it
# helped". The plots above are the check; this is the pointer to the numbers.
st.caption(
    "Each ticked step ran unconditionally. To confirm the result is actually "
    "better, measure it: `python scripts/ab_quality.py <original> <restored>`."
)

if combined.version_2_path is None:
    st.success(f"Version 2 was skipped. Saved Version 3 to `{output_dir / 'v3'}`.")
else:
    st.success(f"Saved Version 2 to `{output_dir / 'v2'}` and Version 3 to `{output_dir / 'v3'}`.")
