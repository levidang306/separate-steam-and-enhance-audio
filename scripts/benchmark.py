"""Before/after benchmark: the old unconditional chain against the gated one.

    python scripts/benchmark.py

"Before" reproduces the V2+V3 chain as it was measured to behave: all three
steps run unconditionally, at full strength, with the denoise output's channel
count adopted from the model. "After" is the current chain, given the *same*
badly-behaved models -- so the comparison isolates the architecture rather than
crediting it with a change of model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suno_restore import chain, quality  # noqa: E402
from tests import corpus, models  # noqa: E402

SR = corpus.SR


def legacy_chain(audio: np.ndarray, sr: int) -> np.ndarray:
    """The old behaviour: every step, every time, at full strength.

    Including the channel collapse -- the denoise model wrote a mono file and
    the loader adopted the file's channel count, so a stereo stem came back
    mono with the side information gone.
    """
    warped = models.measured_tempo(audio, sr)
    denoised = models.measured_denoise(warped, sr)
    collapsed = denoised.mean(axis=1).astype(np.float32) if denoised.ndim == 2 else denoised
    return models.measured_bandwidth(collapsed, sr)


def _row(label: str, report: quality.QualityReport, extra: str = "") -> str:
    band = report.spectral.band_delta_db
    return (
        f"  {label:<8}"
        f"dur {report.timing.duration_delta_ms:+8.1f}ms  "
        f"drift {report.timing.offset_drift_ms:7.1f}ms  "
        f"corr {report.timing.correlation_min:6.3f}  "
        f"5-8k {band.get('5k_8k', float('nan')):+6.2f}  "
        f"8-10k {band.get('8k_10k', float('nan')):+6.2f}  "
        f"atk {report.transient.attack_slope_delta_db:+5.2f}  "
        f"exp {report.dynamic.expansion_spread_db:5.2f}  "
        f"ch {report.stereo.channels_original}->{report.stereo.channels_processed}"
        f"{extra}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    mix = corpus.full_mix(args.seconds)
    guitar = corpus.solo_guitar(args.seconds)
    cases = {
        "guitar (reference)": guitar,
        "clean / null": corpus.clean_full_band(args.seconds),
        "noisy": corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0),
        "bandwidth-limited": corpus.band_limit(mix, SR, cutoff_hz=8000.0),
    }

    print(f"Both columns use the same badly-behaved models; {args.seconds:.0f}s per case.\n")
    for name, audio in cases.items():
        print(name)
        before = quality.compare(audio, legacy_chain(audio, SR), SR, SR)
        outcome = chain.restore_stem(audio, SR, **models.MEASURED, stem_name=name)
        after = quality.compare(audio, outcome.audio, SR, outcome.sr)

        print(_row("before", before))
        ran = "+".join(step for step, o in outcome.steps.items() if o.ran) or "nothing"
        print(_row("after", after, f"   ran [{ran}]"))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
