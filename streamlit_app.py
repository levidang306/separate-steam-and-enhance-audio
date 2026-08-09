"""Interactive runner for the Suno stem restoration workflow.

Three defects, one step each: tempo drift, hiss, bandwidth loss. Every step is
switchable, because on the reference stem set only the bandwidth loss showed up
clearly -- the measured cutoff sat near 16kHz, hiss was 49-58dB down, and tempo
held to within a percent. What each step reports is meant to be read, not
assumed.

Orchestration lives in `suno_restore.pipeline.restore`; this module only renders
it. Driving the steps from here as well would be a second copy of the sequence,
free to drift from the one the Python API uses.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from suno_restore import bandwidth, denoise, pipeline, tempo
from suno_restore.audio_io import (
    FFmpegNotFound,
    TARGET_SR,
    is_effectively_silent,
    load_audio,
)
from suno_restore.metrics import energy_above_hz, spectral_cliff_hz

DEFAULT_INPUT = Path("data/input/Another Heartbreak Stems")
DEFAULT_OUTPUT = Path("data/interim/restored")
DEFAULT_APOLLO = Path("vendor/Apollo")
UPLOAD_DIR = Path("data/interim/uploaded")

st.set_page_config(page_title="Suno Stem Restoration", layout="wide")
st.title("Suno Stem Restoration")
st.caption(
    "Tempo correction (Beat This!) → denoise (Mel-Roformer-Denoise) → bandwidth "
    "extension (Apollo). Stems in, restored stems out."
)


WAVE_BG = "#0d0d0d"
WAVE_PANEL = "#141414"
WAVE_BAR = "#e8833c"
WAVE_GHOST = "#4a3320"
WAVE_TEXT = "#f2f0ed"
WAVE_TEXT_2 = "#9a9691"
N_BARS = 260

# The waveform's ghost brown is deliberately near-invisible behind the bars, so it
# cannot double as a line colour: on the dark surface it fails both the chroma
# floor (reads gray) and contrast (1.48, against a 3:1 minimum). The spectrum uses
# the palette's dark-mode blue/orange pair, which clears every check.
LINE_ORIGINAL = "#3987e5"
LINE_RESTORED = "#d95926"


def format_hz(value: float | None) -> str:
    return f"{value:.0f}Hz" if value else "none"


def envelope(audio: np.ndarray, bars: int = N_BARS) -> np.ndarray:
    """Peak amplitude per bin — what a player's bar display shows."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    edges = np.linspace(0, len(mono), bars + 1, dtype=int)
    return np.array([
        np.abs(mono[lo:hi]).max() if hi > lo else 0.0
        for lo, hi in zip(edges, edges[1:])
    ])


def waveform_figure(pairs: list[tuple[str, np.ndarray, int]]):
    """Player-style bar waveforms, one panel per signal.

    Both panels are scaled by the same peak and the original is drawn behind the
    restored one in a dim tone, so the two panels can be compared directly rather
    than each being normalised to its own maximum.
    """
    reference = envelope(pairs[0][1])
    peak = reference.max() or 1.0
    duration = pairs[0][1].shape[0] / pairs[0][2]
    x = np.arange(N_BARS)

    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 2.1 * len(pairs)),
                             facecolor=WAVE_BG)
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
        ax.set_xticklabels([f"{t / N_BARS * duration:.0f}s" for t in ticks],
                           color=WAVE_TEXT_2, fontsize=8)
        ax.tick_params(length=0, pad=4)
        ax.text(0, 1.16, label, transform=ax.transAxes, color=WAVE_TEXT,
                fontsize=11, fontweight="bold", va="top")
        if index > 0:
            ax.text(1, 1.16, "dim bars = original", transform=ax.transAxes,
                    color=WAVE_TEXT_2, fontsize=8.5, va="top", ha="right")

    fig.tight_layout(h_pad=2.2)
    return fig


def spectrum_figure(pairs: list[tuple[str, np.ndarray, int]]):
    """Average spectra on one axis — the view that shows bandwidth extension."""
    import librosa

    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=WAVE_BG)
    ax.set_facecolor(WAVE_PANEL)
    for (label, audio, sr), color in zip(pairs, (LINE_ORIGINAL, LINE_RESTORED)):
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        stft = np.abs(librosa.stft(mono, n_fft=8192, hop_length=2048))
        spec = np.percentile(stft, 95, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
        ax.semilogx(freqs, 20 * np.log10(spec + 1e-12), label=label,
                    linewidth=1.6, color=color)
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
        "Upload an audio file",
        type=["wav", "mp3", "flac", "aiff", "aif", "ogg", "m4a"],
        help="Restores this one file. Leave empty to use the folder below instead.",
    )
    input_dir = Path(st.text_input("…or a stem folder", str(DEFAULT_INPUT)))
    output_dir = Path(st.text_input("Output folder", str(DEFAULT_OUTPUT)))

    if uploaded is not None:
        # The pipeline works on a folder, so an upload becomes a folder of one.
        # It is cleared each time so a previous upload cannot be picked up too.
        upload_dir = Path(UPLOAD_DIR)
        shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / uploaded.name).write_bytes(uploaded.getbuffer())
        input_dir = upload_dir
        st.caption(f"Using uploaded **{uploaded.name}**")

    st.header("Steps")
    do_tempo = st.checkbox("1 — Tempo correction", value=True)
    do_denoise = st.checkbox("2 — Denoise", value=True)
    do_bandwidth = st.checkbox("3 — Bandwidth extension", value=True)

    apollo_repo = Path(st.text_input("Apollo checkout", str(DEFAULT_APOLLO)))

    cuda = torch.cuda.is_available()
    device = st.selectbox(
        "Device", ["cuda", "cpu"] if cuda else ["cpu"],
        help="Apollo needs CUDA; on CPU it is far slower.",
    )
    st.caption(
        f"GPU: **{torch.cuda.get_device_name(0)}**" if cuda else "No CUDA device detected"
    )

    run = st.button("Run restoration", type="primary", use_container_width=True)

if not input_dir.is_dir():
    st.warning(f"Stem folder not found: `{input_dir}`")
    st.stop()

try:
    paths = pipeline.discover_stems(input_dir)
except FFmpegNotFound as error:
    st.error(str(error))
    st.stop()

if not paths:
    st.warning(f"No audio files in `{input_dir}`")
    st.stop()

with st.expander("What each step uses, and why", expanded=False):
    st.markdown(
        """
| Step | Fixes | Model | Why this one |
|---|---|---|---|
| **1 — Tempo correction** | BPM wandering across the track | [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, MIT) | Beats are tracked on **one** reference stem — drums, else bass, else instrumental — and the resulting time-warp is applied to every stem. Tracking each stem separately would give each its own warp and drift them apart. |
| **2 — Denoise** | Hiss, crackle, "AI shimmer" | Mel-Roformer-Denoise-Aufr33, via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | A music source-separation model that splits a clean track from the noise. **DeepFilterNet was rejected**: it calls itself a *Speech* Enhancement Framework, and on bass or drums it suppresses musical content it cannot recognise as speech. |
| **3 — Bandwidth extension** | Frequencies missing above the export cutoff | [Apollo](https://github.com/JusperLee/Apollo) (JusperLee) | Trained on MP3 codec artifacts and evaluated on separated-stem data, which matches this input. **AudioSR was rejected**: it needs ~7.2GB and does not fit an 8GB card, and it is trained on synthetic low-pass rather than codec cutoffs. **UniverSR was rejected**: its highest input mode assumes content stops at 12kHz, but these stems carry real content to 16–18.5kHz, which it would discard first. |

**Decoding.** Everything is read through ffmpeg, not soundfile. Suno's exports are
low-bitrate MP3s with unreliable headers, and libsndfile silently truncates most of
them — it read 87 seconds of a 294-second stem while reporting that as the full
length, so nothing downstream could notice.

**Order.** Tempo runs first because it is the only step that touches the time axis;
running it later would re-stretch what the other steps just repaired. Denoise runs
before bandwidth extension because super-resolution synthesises the new high band
from what sits below the cutoff — feeding it noise extends the noise.

**Sample rate.** Stems are handled at 48kHz. Apollo is built for 44.1kHz and does no
resampling of its own, so audio is converted down for it and back afterwards; without
that its output returns 8.8% sharp and stems end up at mixed rates.
"""
    )

st.subheader(f"{len(paths)} stem{'s' if len(paths) != 1 else ''} found")
st.write(", ".join(f"`{p.name}`" for p in paths))

if not run:
    st.info("Set the steps you want in the sidebar, then run.")
    st.stop()

source_by_name = {p.stem: p for p in paths}

with st.status("Loading stems…", expanded=True) as load_status:
    rows = []
    for path in paths:
        audio, sr = load_audio(path)
        rows.append({
            "stem": path.stem,
            "duration": f"{audio.shape[0]/sr:.1f}s",
            "cutoff": format_hz(spectral_cliff_hz(audio, sr)),
            "energy >13kHz": f"{energy_above_hz(audio, sr):.2f}%",
            "silent": is_effectively_silent(audio),
        })
        del audio
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "**cutoff** is where the spectrum falls off a cliff — the lower it is, the more "
        "treble is missing. **energy >13kHz** is how much of the signal lives above that "
        "line. A **silent** stem is passed through untouched: there is nothing to restore."
    )
    load_status.update(label=f"Inspected {len(paths)} stems at {TARGET_SR}Hz", state="complete")

tempo_status = st.status(
    "Step 1 — Tempo correction" if do_tempo else "Step 1 — Tempo correction (skipped)",
    expanded=do_tempo,
)
if do_tempo:
    with tempo_status:
        st.write(
            "Beat-tracks one reference stem and applies that single time-warp to "
            "every stem, so they stay aligned with each other."
        )

stem_status = st.status("Steps 2-3 — Restoring stems", expanded=True)
with stem_status:
    st.write(
        "Each stem is denoised and bandwidth-extended, then written immediately. "
        "The residual figure is how loud the removed noise was — far below the "
        "signal means the stem was already clean."
    )
    stem_lines = st.empty()

completed: list[pipeline.StemReport] = []


def render_tempo(analysis: tempo.TempoAnalysis) -> None:
    with tempo_status:
        st.write(f"**{analysis.summary()}**")
        st.write(
            f"Detected {int(analysis.before['n_beats'])} beats · "
            f"spread {analysis.before['min_bpm']:.1f}–{analysis.before['max_bpm']:.1f} BPM "
            f"(σ {analysis.before['std_bpm']:.2f})"
        )
    tempo_status.update(label=f"Step 1 — {analysis.summary()}", state="complete")


def render_stem(entry: pipeline.StemReport) -> None:
    completed.append(entry)
    lines = []
    for done in completed:
        parts = [f"**{done.name}**"]
        if done.denoise_residual_db is not None:
            residual = (
                "—" if done.denoise_residual_db == -np.inf
                else f"{done.denoise_residual_db:.1f}dB"
            )
            parts.append(f"residual {residual} · {done.denoise_note}")
        if done.cliff_before_hz or done.cliff_after_hz:
            parts.append(
                f"cutoff {format_hz(done.cliff_before_hz)} → {format_hz(done.cliff_after_hz)}"
            )
        if done.notes:
            parts.append("; ".join(done.notes))
        lines.append(" — ".join(parts))
    stem_lines.markdown("\n\n".join(f"- {line}" for line in lines))
    stem_status.update(label=f"Steps 2-3 — {len(completed)}/{len(paths)} stems restored")


try:
    report = pipeline.restore(
        input_dir,
        output_dir,
        do_tempo=do_tempo,
        do_denoise=do_denoise,
        do_bandwidth=do_bandwidth,
        apollo_repo=apollo_repo,
        device=device,
        work_dir=output_dir,
        on_tempo=render_tempo,
        on_stem=render_stem,
    )
except tempo.NoReferenceStem as error:
    tempo_status.update(label="Step 1 — no usable reference stem", state="error")
    st.error(str(error))
    st.stop()
except bandwidth.ApolloUnavailable as error:
    stem_status.update(label="Steps 2-3 — Apollo unavailable", state="error")
    st.error(str(error))
    st.stop()

if not do_tempo:
    tempo_status.update(state="complete")
stem_status.update(label=f"Steps 2-3 — {len(completed)} stems restored", state="complete")

st.subheader("Results")
st.caption(
    "The **waveform** shows level over time, on a scale shared between the two panels — "
    "restoration barely moves levels, so near-identical panels are the expected result, "
    "not a sign that nothing ran. The **spectrum** is where the work shows: the gap "
    "between the two lines at the top end is the content bandwidth extension rebuilt."
)
for name, entry in report.stems.items():
    if entry.output_path is None:
        continue
    with st.expander(f"{name} → `{entry.output_path.name}`", expanded=False):
        left, right = st.columns(2)
        with left:
            st.caption("Original")
            st.audio(str(source_by_name[name]))
        with right:
            st.caption("Restored")
            st.audio(str(entry.output_path))

        # Re-read both rather than holding originals in memory for the whole run:
        # at full length a duplicate set of stems costs several hundred MB.
        original_audio, original_sr = load_audio(source_by_name[name])
        restored_audio, restored_sr = load_audio(entry.output_path, target_sr=None)
        pairs = [
            ("Original", original_audio, original_sr),
            ("Restored", restored_audio, restored_sr),
        ]
        st.pyplot(waveform_figure(pairs))
        # Levels barely move under restoration, so the waveform alone would imply
        # nothing happened; the spectrum is where bandwidth extension shows up.
        st.pyplot(spectrum_figure(pairs))

st.success(f"Wrote {len(completed)} restored stems to `{output_dir}`")
