# Audio Restoration — Requirements & Working Agreement

**Version 1.0 — 10 August 2026**
**Audience:** everyone involved (technical and non-technical)
**Purpose:** fix what we are building, what we need to receive, and what we deliver — so the workflow does not change mid-cycle again.

---

## 1. Why this document exists

Last week the workflow changed while work was in progress. Effort went up, output quality did not.

This document is the agreement that prevents that. It defines three things in advance:

1. **Input** — what we need from you before work starts.
2. **Output** — what you get back, and how it is judged.
3. **Change control** — how a change of direction is handled if one is genuinely needed.

Once agreed, this is the reference. Requests outside it are handled as a new cycle, not as an in-flight change (see §7).

---

## 2. What we are fixing

Suno exports carry three known defects. We measured all three on the reference material. **They are not equally real, and this matters for where effort goes.**

| # | Defect | Verdict | Effort |
|---|---|---|---|
| 3 | **Bandwidth loss** — audio has no content above ~16–18.5 kHz | **Real and significant** | **Primary** |
| 1 | **Tempo drift** — timing wanders slightly | Real but small | Secondary |
| 2 | **Hiss / noise** | **Not detectable** on this material | None for now |

### Defect 3 — Bandwidth loss (primary target)

Every stem has a hard ceiling where audio content stops. Above that line there is nothing.

| Stem | Cut-off | Energy above 16 kHz |
|---|---|---|
| Lead Vocal | 18,328 Hz | 1.22% |
| Bass | 17,836 Hz | 0.01% |
| Drum Kit | 18,311 Hz | 1.18% |
| Backing Vocals | 18,217 Hz | 0.29% |
| Piano | 15,896 Hz | 0.00% |
| Acoustic Guitar | 18,533 Hz | 3.57% |

Restoration (Apollo) lifts the ceiling to ~21.5–21.9 kHz. **This is the only step that produces an audible, meaningful difference.**

> **Correction to earlier documentation:** the Neural Antalog notes describe a cut-off at 12–13 kHz. Our measurements show 16–18.5 kHz. The defect is real, but milder than documented. Plan against the measured numbers.

### Defect 1 — Tempo drift (secondary)

Reference track: 68.18 BPM, 190 beats. Correction adjusted 121 of 191 segments, **median adjustment 0.56%**, total length change **−0.23%**.

> **Do not trust the "drift vs tempo ±100%" line in the current report.** It is an artefact of the beat tracker misreading the tempo by one octave (it reported 136.4 BPM = exactly 2 × 68.18, splitting one interval in half). The reliable figure is the **median 0.56% correction**.

### Defect 2 — Hiss (no action)

The denoiser removed content sitting **−49 to −57 dB** below the signal — inaudible. Noise-floor measurement above 16 kHz returned **−240 dB (absolute zero) on 5 of 6 stems**: above the cut-off there is nothing at all, not even noise.

**Conclusion: on this material the denoise step does essentially nothing.** It stays available but is not a priority, and it should not be reported as a benefit it did not deliver.

---

## 3. What we need from you (Input)

Work does not start until these are supplied. Incomplete input is the single largest cause of wasted effort.

### 3.1 Required with every request

| Item | Detail needed |
|---|---|
| **Audio files** | Separated stems preferred (vocal, bass, drums, etc.). WAV preferred; MP3 accepted. |
| **Which defect to fix** | Bandwidth / tempo / noise — name it. "Make it better" is not a specification. |
| **What "wrong" sounds like** | One or two sentences in plain language: *"the vocal sounds dull and muffled"*, *"the drums slip out of time after the first chorus"*. |
| **Where in the track** | Timestamps if the problem is localised (e.g. 1:15–1:40). "Whole track" is a valid answer. |
| **Reference, if any** | A version that sounds correct, if one exists. Hugely valuable — see §5. |
| **What must not change** | E.g. "do not change the length", "keep it stereo", "do not touch the vocal". |

### 3.2 Fixed guarantees (we do not change these)

Whatever else happens, the output keeps:

- the same set of stems,
- the same channel count (stereo stays stereo),
- the same sample rate,
- stems still aligned with each other.

### 3.3 Explicitly out of scope

Mastering — EQ, compression, loudness. Applied per stem it stacks up wrongly; it belongs in one pass on the finished mix, not here.

---

## 4. What you get back (Output)

### 4.1 Deliverables

1. **Restored audio** — one file per input stem, same format guarantees as §3.2.
2. **A change report** — what each step actually did, in numbers.
3. **A quality comparison** — original vs restored, measured.

### 4.2 The change report

Each step reports what it **did**, not that it "worked":

- **Tempo** — how many segments were stretched and by how much; total length change.
- **Denoise** — how loud the removed material was. Below −40 dB means the file was already clean and nothing audible changed.
- **Bandwidth** — where the ceiling sat before and after.

### 4.3 How we judge success

Three numbers decide whether a result is accepted:

| Metric | Meaning | Pass |
|---|---|---|
| **2-second correlation** | Is it still the same performance? | ≥ 0.90 |
| **Timing drift** | Did it stay in time? | No new drift |
| **Per-band level change** | Did the tone change? | Within tolerance |

Two failure modes matter equally:

- **Output identical to input** → the step did nothing. Not a success.
- **Correlation collapsed** → the step did far too much. Not a success.

Restoration must **fix what is broken and leave alone what is not**. A clean input coming back essentially unchanged is a correct result, not a failure.

---

## 5. Training data — two supported paths

To improve the models we need training data. There are exactly two ways to supply it.

### Case 1 — You have both bad and good versions

You provide **matched pairs**: same song, same lyrics, same arrangement, differing only in quality.

```
Bad version  ─┐
              ├─→  Paired dataset  ─→  Training
Good version ─┘
```

**This is the preferred path.** The model learns directly from real defects. Requirement: the two versions must be the *same performance*, not two different renders of the same song.

### Case 2 — You only have good versions

You provide **clean audio only**. We generate the damaged counterpart ourselves by applying the defects listed in §2.

```
Good audio ─→  Apply known defects  ─→  Paired dataset  ─→  Training
               (bandwidth cut,
                tempo drift,
                noise)
```

The defects we simulate are exactly the ones we measured — bandwidth cut at the measured 16–18.5 kHz, tempo drift at the measured scale, not invented numbers.

**Either path works. What does not work is unlabelled audio with no indication of which category it belongs to.** Every file supplied must be marked as *good*, *bad*, or *paired*.

---

## 6. Priority

| Priority | Work | Rationale |
|---|---|---|
| **1** | Bandwidth restoration | The only measured defect with real impact |
| **2** | Tempo correction | Real but small (0.56% median) |
| **3** | Denoise | No measurable effect on current material |
| **—** | Mastering | Out of scope |

---

## 7. Change control

This is the part that protects the schedule.

1. **Within a cycle, the scope in §3 is fixed.** New findings are recorded, not acted on immediately.
2. **A change of direction requires a written change request**, stating what changes and what is dropped to make room. Adding without dropping is what caused last week's outcome.
3. **Measurements beat assumptions.** If documentation and measurement disagree — as with the 12–13 kHz figure — the measurement wins, and the documentation is corrected.
4. **No step is reported as a success it did not deliver.** If denoise changed nothing, the report says it changed nothing.

---

## 8. Definition of done

A cycle is complete when:

- Every requested defect has been addressed **or** documented as not present in the material.
- The output meets the §3.2 format guarantees.
- The §4.3 quality metrics pass.
- Clean input material comes back essentially unchanged.
- The change report matches what actually happened.

Not "the process ran". Not "it sounds more processed". Measurably better where broken, untouched where it was not.
