# Version 2 + Version 3 Restoration — Renewed Merge

## Context

The repository has two complementary restoration systems:

- **Version 2 (`midi_repair`)** looks for one localized damaged span. It separates and
  transcribes the input, detects a candidate from spectral flatness, regenerates MIDI for that
  span, and re-renders it from donor notes elsewhere in the file.
- **Version 3 (`suno_restore`)** measures tempo, noise, and bandwidth damage. A step runs only
  when its gate can justify it; the wet result is blended conservatively and verification either
  accepts it or returns the dry input.

The combined order is Version 2 → Version 3. Version 2's detector is calibrated on the raw input,
so running Version 3 first would change the spectrum that detector is supposed to measure.

## Renewed merge after the Version 3 update

The first integration at `0ebe9f99b8beb4d9ec4d77cb34687f511d9e6620` predated Version 3's
damage gates, blending, native-rate preservation, and verification. Its orchestrator forced the
22.05kHz mono Version 2 render through a 48kHz resample before Version 3. That enlarged the sample
rate but could not restore stereo information or real content already discarded above 11.025kHz.
It also described Version 3 as unconditional, which is no longer true.

The renewed merge preserves the updated Version 3 contracts:

1. Version 2 still analyzes and renders internally at 22.05kHz mono.
2. Only the difference between Version 2's repaired and analysis renders is resampled and applied
   to every channel of the native decoded upload.
3. Adding the same delta to every channel preserves the original stereo difference; samples and
   high-frequency content outside the repair remain sourced from the native upload.
4. If the candidate repair does not reduce Version 2's measured risk, Version 2 rolls back to the
   native upload.
5. Version 3 receives this native-rate, native-channel Version 2 artifact. Its existing gates,
   wet/dry blend, verification, and dry fallback remain authoritative.

## Outputs

`full_pipeline.restore_from_stem` processes exactly one input file and keeps both artifacts:

```text
<output_dir>/
├── v2/<name>_v2.wav
├── v3/<name>_restored.wav
└── _work/
```

`CombinedReport.version_2_path` and `version_3_path` make the artifacts explicit. The Streamlit
UI in `merged_app.py` presents Original, Version 2, and Version 3 audio players side by side,
draws all available waveforms and spectra, and reports each version's decision separately.

## GPU memory

Version 2's BS-RoFormer, MuScriptor, and AMT models and Version 3's restoration models must not
remain resident together on smaller GPUs. `midi_repair.unload_models()` runs in a `finally` block
at the Version 2 → Version 3 boundary, including when Version 2 raises an error.

## Dependencies

The Version 2 packages are appended to `requirements.txt`: `bs-roformer-infer`, `muscriptor`,
`mido`, `music21`, `dtw-python`, and `anticipation`. Heavy/optional Version 2 imports are lazy so
the combined UI can render and explain setup even before those packages are installed; running
Version 2 still requires the complete requirements file and model downloads.

## Verification

Automated integration tests prove that:

- the mono Version 2 repair delta preserves the native stereo difference;
- Version 2 and Version 3 write to distinct, exposed paths;
- Version 3 receives the Version 2 audio at the same sample rate and channel layout;
- Version 2 models are released even when Version 2 fails.

The enhanced Version 3 quality, gate, blend, signal-integrity, and verification suites continue to
cover the downstream stage.

An actual model-backed end-to-end run still requires the multi-gigabyte checkpoints and a suitable
GPU. Before production release, run the combined path on the deployment GPU and record peak VRAM
at the Version 2 → Version 3 boundary.
