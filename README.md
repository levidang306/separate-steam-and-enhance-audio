# Suno Stem Restoration

Restores stems exported by Suno. Three defects, one step each:

| Step | Defect | Model |
|---|---|---|
| 1 | Tempo drift | [Beat This!](https://github.com/CPJKU/beat_this) |
| 2 | Hiss / crackle | Mel-Roformer-Denoise, via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) |
| 3 | Bandwidth loss | [Apollo](https://github.com/JusperLee/Apollo) |

Stems in, restored stems out — same set of stems, same channel count, same sample rate, still
aligned with each other.

Every enabled step runs unconditionally and at full strength:

```
INPUT -> TEMPO -> DENOISE -> BANDWIDTH -> OUTPUT
```

There is no decision layer and no safety net. A step that is ticked runs, its
output replaces the signal, and the result is written whatever it measures like.
That is a deliberate choice, and it is worth knowing what it costs: this exact
configuration warped a solo guitar by 375ms, expanded its quiet frames down by
15.75dB, took 3.4dB out of the 5-10kHz band carrying pick attack, and folded the
stereo image to mono -- while reporting success.

So check the result rather than assuming it. [`quality.py`](suno_restore/quality.py)
measures what a run actually did -- timing drift, per-band level, transients,
dynamics, stereo, artifacts -- and `scripts/ab_quality.py` prints it:

```bash
.venv/Scripts/python scripts/ab_quality.py original.wav restored.wav
```

An output identical to its input means the step did nothing. An output whose
2-second correlation has collapsed means it did far too much.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

winget install Gyan.FFmpeg                                    # required by Step 2
git clone https://github.com/JusperLee/Apollo vendor/Apollo   # Step 3, no pip package
```

The cu118 pin is deliberate. Plain `pip install torch` resolves to a CPU-only wheel, which drops
every step onto the CPU — denoising one stem goes from under a minute to well over ten.

## Usage

### Combined Version 2 + Version 3

```bash
.venv/Scripts/streamlit run merged_app.py
```

Upload one file and run the combined flow. Version 2 performs localized MIDI repair first;
enhanced Version 3 then evaluates, selectively restores, blends, and verifies that result. The UI
shows separate audio players and plots for the original, Version 2, and Version 3 outputs.

Artifacts are deliberately separate:

```text
data/interim/restored_merged/
├── v2/<name>_v2.wav
└── v3/<name>_restored.wav
```

From Python, `full_pipeline.restore_from_stem(...)` returns a `CombinedReport` with
`version_2_path`, `version_3_path`, and the detailed reports from both versions.

### Version 3 only

```bash
.venv/Scripts/streamlit run streamlit_app.py
```

Upload one audio file, choose which steps to run, and read what each reports.

Or from Python:

```python
from suno_restore import restore

report = restore(
    "data/input/Another Heartbreak Stems",
    "data/interim/restored",
    apollo_repo="vendor/Apollo",
    device="cuda",
)
```

## Reading the output

Every step reports what it did rather than asserting it worked, because on real material not all
three defects show up:

- **Tempo** — how many inter-beat segments were stretched and by how much. Watch the length change:
  the step warps only what falls inside the detected beat grid and copies the rest, so on material
  where the grid is thin the uncorrected regions inherit the drift instead of losing it.
- **Denoise** — the level of what was removed, relative to the stem. Below about -40dB means the
  stem was already clean and the step changed nothing audible.
- **Bandwidth** — where the spectral cliff sat before and after.

None of these says whether the result is *better*. For that, measure it:

```bash
.venv/Scripts/python scripts/ab_quality.py original.wav restored.wav
```

The numbers to read first are 2-second correlation (below ~0.9 means the timing or spectrum has
been disturbed), timing drift, and the per-band deltas.

## Notes

- These MP3 exports are low bitrate with unreliable headers, and libsndfile silently truncates most
  of them — it read 87 seconds of a 294-second stem. Everything decodes through ffmpeg instead.
- Apollo's bundled `inference.py` does not work as shipped and is not used; the model is driven
  directly. See `suno_restore/bandwidth.py` for the details.
- Mastering (EQ, compression, loudness) is deliberately out of scope. Doing it per stem and then
  recombining stacks compression; it belongs in one pass on the finished mix.

Design and measurements: [`docs/design/2026-08-09-suno-stem-restoration-design.md`](docs/design/2026-08-09-suno-stem-restoration-design.md)
