# Merged Restoration Pipeline — Design

## Context

Two pipelines existed on separate branches, built on opposite assumptions about
where damage in a stem lives:

- `rewrite-v2` ("Enhanced Stem Workflow") assumes one localized, findable
  damaged span per stem. It separates, transcribes to MIDI, detects a
  candidate region by spectral flatness, regenerates that region's MIDI, and
  re-renders it from real donor notes elsewhere in the same stem.
- `neural-restoration-redesign` (`suno_restore`) assumes three specific,
  global defects are present across the whole stem regardless of any
  detector's opinion, and corrects all three unconditionally: tempo drift,
  hiss/crackle, bandwidth loss.

Requested flow: run the localized-repair pipeline first to produce a
best-effort enhanced audio, then run the whole-track pipeline on that audio
for final correction. This document records what changed to make that work
in one process.

## Dependency merge

Checked with real `pip install --dry-run` (not just reading metadata) against
this repo's existing `.venv`, which already has `torch==2.7.1+cu118` and the
rest of `suno_restore`'s stack installed. Result: no resolver conflicts, once
one change was made.

**`huggingface-hub` pin relaxed from `==0.24.6` to `>=1.5,<2.0`.** The
`requirements.txt` comment and this repo's own README claimed Apollo needs
exactly `0.24.6` because it imports `PyTorchModelHubMixin` from the
`huggingface_hub.hub_mixin` submodule. Checked the actual code
(`vendor/Apollo/look2hear/models/base_model.py:10`): it imports the stable
top-level `huggingface_hub.PyTorchModelHubMixin` export, not the submodule.
Verified directly in this repo's `.venv` (already drifted to
`huggingface_hub==1.27.0` independent of this change) that both the import
and `BaseModel.from_pretrain` still work. `transformers==5.14.1` (needed for
Stage A's AMT model) requires `huggingface-hub<2.0,>=1.5.0`; relaxing the pin
to that range satisfies both sides. No transformers downgrade needed.

**`torch` standardized on `2.7.1+cu118`.** This is the newest CUDA build
compatible with the local dev machine's driver (512.72 / CUDA 11.6); cu118
wheels stop at exactly this version. Checked every Stage A package's own
minimum torch requirement (`muscriptor>=2.0`, `bs-roformer-infer>=2.0`,
`beat-this>=2`) — all satisfied. It also runs fine on the production server's
Tesla T4 (newer driver, CUDA 13.x userspace) since newer drivers stay
backward-compatible with older CUDA runtimes.

**Everything else Stage A needs** (`bs-roformer-infer`, `muscriptor`, `mido`,
`music21`, `dtw-python`, `ml_collections`, `anticipation` from git) installed
alongside the existing stack with zero version overlaps — `numpy`, `librosa`,
`soundfile`, `soxr`, `matplotlib`, `streamlit`, `beat-this` were already
pinned identically on both branches.

## Code structure

- `midi_repair/` — Stage A, ported from `rewrite-v2`'s `streamlit_app.py`.
  Same models, same thresholds, same order of operations; only the Streamlit
  calls were removed, replaced with the `progress` callback convention
  `suno_restore` already uses. Model loaders cache per-device
  (`functools.lru_cache`, matching `suno_restore.bandwidth.load_model`), and
  `midi_repair.unload_models()` evicts them explicitly.
- `full_pipeline.py` — orchestrates Stage A → Stage B on one input stem.

## Two things the merge had to resolve that weren't library conflicts

**Sample rate.** Stage A works internally at 22050Hz (a `rewrite-v2`
constant) and its output (`final_stem.wav` in the original prototype) was
written at that rate. Stage B requires 48kHz throughout — its tempo/denoise/
bandwidth models and its cross-stem sync guarantee assume it. `full_pipeline.
restore_from_stem` resamples Stage A's output to `suno_restore.TARGET_SR`
(48000) before handing it to Stage B. Skipping this would feed Stage B a
signal already missing everything above ~10kHz, which bandwidth extension
cannot tell apart from a genuinely narrow-band stem.

**GPU memory.** Stage A's models (BS-RoFormer, MuScriptor, the AMT model) and
Stage B's models (beat_this, the denoiser, Apollo) were each sized
individually to fit an 8GB card, but never tested resident together.
`full_pipeline.restore_from_stem` calls `midi_repair.unload_models()` between
stages so only one stage's models occupy VRAM at a time.

## Order

Stage A runs before Stage B, not after. Stage A's degradation detector is
calibrated on spectral flatness of the *input* signal; running Stage B first
would denoise and bandwidth-extend the stem before Stage A ever sees it,
changing the exact signal property the detector measures.

## Verified in this session

- `pip install --dry-run` then a real install of every Stage A package
  against the existing Stage B environment: no conflicts.
- Every Stage A and Stage B module imports together in one interpreter.
- Stage A's pure logic (`compute_degradation_risk`, `detect_flagged_region`,
  `decode_notes`, `render_window`, `equal_power_crossfade_splice`) smoke-
  tested against synthetic audio with an injected noisy region: the detector
  found the injected region, splicing preserved array shape, and risk in the
  splice window measurably dropped.

## Not verified — needs a GPU run before shipping

An actual end-to-end run was not performed here: it requires downloading and
loading BS-RoFormer, MuScriptor ("large", ~1.37B params), the AMT model
(`stanford-crfm/music-small-800k`), `beat_this`, the Mel-Roformer denoiser,
and Apollo's checkpoint — multiple GB and several models resident across two
stages, which is exactly the scenario `unload_models()` exists for but has
not been measured. Before this ships, run it once on the production Tesla T4
(16GB, currently showing ~13.6GB already in use by another process per the
`nvidia-smi` output shared in chat — confirm that's not contending for the
same GPU before scheduling a real run) and once on the 8GB dev card, and
watch peak VRAM at the Stage A → Stage B boundary specifically.
