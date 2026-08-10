# Suno Stem Restoration

Restores stems exported by Suno. Three defects, one step each:

| Step | Defect | Model |
|---|---|---|
| 1 | Tempo drift | [Beat This!](https://github.com/CPJKU/beat_this) |
| 2 | Hiss / crackle | Mel-Roformer-Denoise, via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) |
| 3 | Bandwidth loss | [Apollo](https://github.com/JusperLee/Apollo) |

Stems in, restored stems out — same set of stems, same channel count, same sample rate, still
aligned with each other.

**No step runs unless it can show it should.** An enhancement chain that runs unconditionally will
make good audio worse, and it does so silently: on a solo guitar this one warped the timing by
375ms, expanded the quiet frames down by 15.75dB, took 3.4dB out of the 5–10kHz band where pick
attack lives, and folded the stereo image to mono — while reporting success. So every step now has
to make its case first, against measurements, before it is allowed to touch anything:

```
INPUT ─► ANALYSIS ─► DAMAGE DETECTION ─► STEP DECISION
      ─► SELECTIVE RESTORATION ─► WET/DRY BLEND
      ─► QUALITY VERIFICATION ─► ACCEPT or DRY ─► OUTPUT
```

| Module | Question it answers |
|---|---|
| [`quality.py`](suno_restore/quality.py) | What actually changed? Timing, bands, transients, dynamics, stereo, artifacts |
| [`gate.py`](suno_restore/gate.py) | Should this step run on this material at all? |
| [`damage.py`](suno_restore/damage.py) | Where is the damage, in time and frequency? |
| [`blend.py`](suno_restore/blend.py) | How much of the model's output survives, and where? |
| [`verify.py`](suno_restore/verify.py) | Is the result safe, and is it actually better? |
| [`chain.py`](suno_restore/chain.py) | All of the above, in order, with the dry signal held as fallback |

The dry signal is kept from start to finish and is never written over, so any step that turns out
to have made things worse can be undone. A stem that fails verification is replaced by its input
and reported as `is_enhanced=false` with the reason recorded — never as a successful enhancement.

Configuration lives in [`.env.example`](.env.example); every default is the cautious one.

## Checking it

```bash
.venv/Scripts/python scripts/ab_quality.py original.wav processed.wav   # objective A/B
.venv/Scripts/python scripts/quality_gate.py                            # the CI gate
.venv/Scripts/python scripts/ablation.py                                # per-step matrix
.venv/Scripts/python scripts/benchmark.py                               # before/after
```

`scripts/quality_gate.py` is what CI runs. It fails if clean input is modified, if the reference
guitar regresses, or if any combination of steps damages clean material — and it runs against
stand-in models that reproduce the *measured misbehaviour* of the real ones, so passing means the
architecture holds even when what it is given is as bad as what was originally observed.

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
three defects show up — and a skip is a result, not a failure:

- **Tempo** — usually skipped, and says why. It only helps material with a real, steady, percussive
  pulse, and is off by default. Where it does run: how many inter-beat segments were corrected and
  by how much.
- **Denoise** — the level of what was removed, relative to the stem, and the confidence it was
  removed at. Below about -40dB means the stem was already clean and the step changed nothing.
- **Bandwidth** — whether a real cutoff was found, and where. No cliff means no restoration:
  generating a high band for material that never had one is invention, not repair.

`report.stems[name].engine_config` carries the whole record — every decision, the measurements
behind it, and the verification verdict — so a run can be explained after the fact rather than
guessed at.

## Notes

- These MP3 exports are low bitrate with unreliable headers, and libsndfile silently truncates most
  of them — it read 87 seconds of a 294-second stem. Everything decodes through ffmpeg instead.
- Apollo's bundled `inference.py` does not work as shipped and is not used; the model is driven
  directly. See `suno_restore/bandwidth.py` for the details.
- Mastering (EQ, compression, loudness) is deliberately out of scope. Doing it per stem and then
  recombining stacks compression; it belongs in one pass on the finished mix.

Design and measurements: [`docs/design/2026-08-09-suno-stem-restoration-design.md`](docs/design/2026-08-09-suno-stem-restoration-design.md)
