"""The CI quality gate: run the checks that must never regress, and exit non-zero.

    python scripts/quality_gate.py

Three things are checked, in the order they matter:

  1. **Null test.** Clean material must come back essentially unchanged. This is
     first because it is the failure that would otherwise go unnoticed -- most
     material is already fine, and quietly degrading it is invisible until
     somebody listens closely to something they cared about.
  2. **Reference regression.** The solo-guitar case must still skip tempo, skip
     bandwidth, apply denoise gently, and hold 5-10kHz inside +/-1dB.
  3. **Ablation safety.** No combination of steps may damage clean material.

All three run against models that reproduce the *measured misbehaviour* of the
real ones, so a pass means the architecture holds even when what it is given is
as bad as what was originally observed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suno_restore import chain, quality  # noqa: E402
from tests import ablation, corpus, models  # noqa: E402

SR = corpus.SR

NULL_MIN_CORRELATION = 0.98
NULL_MAX_BAND_DEVIATION_DB = 0.5
NULL_MAX_DRIFT_MS = 5.0
PRESENCE_TOLERANCE_DB = 1.0


class Failure(Exception):
    pass


def _check(condition: bool, message: str, failures: list[str]) -> bool:
    if not condition:
        failures.append(message)
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    return condition


def null_test(seconds: float, failures: list[str]) -> None:
    print("\nNull test -- clean material must come back unchanged")
    for name in ("clean_full_band", "drums", "vocal"):
        clean = corpus.CLEAN_CLASSES[name](seconds)
        outcome = chain.restore_stem(clean, SR, **models.MEASURED, stem_name=name)
        report = quality.compare(clean, outcome.audio, SR, outcome.sr)
        worst = max(abs(value) for value in report.spectral.band_delta_db.values())

        _check(
            report.timing.correlation_min > NULL_MIN_CORRELATION,
            f"{name}: correlation {report.timing.correlation_min:.4f} > {NULL_MIN_CORRELATION}",
            failures,
        )
        _check(
            worst <= NULL_MAX_BAND_DEVIATION_DB,
            f"{name}: worst band {worst:.3f} dB <= {NULL_MAX_BAND_DEVIATION_DB}",
            failures,
        )
        _check(
            report.timing.offset_drift_ms <= NULL_MAX_DRIFT_MS,
            f"{name}: drift {report.timing.offset_drift_ms:.2f} ms <= {NULL_MAX_DRIFT_MS}",
            failures,
        )
        _check(report.stereo.channels_preserved, f"{name}: channels preserved", failures)


def reference_test(seconds: float, failures: list[str]) -> None:
    print("\nReference guitar -- the Phase 9 targets")
    guitar = corpus.solo_guitar(seconds)
    outcome = chain.restore_stem(
        guitar, SR, **models.MEASURED, stem_name="guitar", available_stems=["guitar"]
    )
    report = quality.compare(guitar, outcome.audio, SR, outcome.sr)

    _check(not outcome.steps["tempo"].ran, "tempo skipped", failures)
    _check(bool(outcome.steps["tempo"].decision.reason), "tempo skip has a reason", failures)
    _check(not outcome.steps["bandwidth"].ran, "bandwidth skipped", failures)
    _check(
        bool(outcome.steps["bandwidth"].decision.reason),
        "bandwidth skip has a reason",
        failures,
    )
    _check(outcome.steps["denoise"].ran, "denoise applied", failures)
    _check(
        outcome.steps["denoise"].decision.confidence < 0.5,
        f"denoise gentle (confidence {outcome.steps['denoise'].decision.confidence:.2f} < 0.5)",
        failures,
    )

    for band in ("5k_8k", "8k_10k"):
        delta = report.spectral.band_delta_db[band]
        _check(
            abs(delta) <= PRESENCE_TOLERANCE_DB,
            f"{band} {delta:+.3f} dB within +/-{PRESENCE_TOLERANCE_DB}",
            failures,
        )

    _check(
        abs(report.timing.duration_delta_ms) < 5.0,
        f"duration {report.timing.duration_delta_ms:+.2f} ms",
        failures,
    )
    _check(
        report.timing.offset_drift_ms < 5.0,
        f"timing drift {report.timing.offset_drift_ms:.2f} ms",
        failures,
    )
    _check(report.stereo.channels_preserved, "stereo preserved", failures)
    _check(
        report.dynamic.expansion_spread_db <= 6.0,
        f"expansion spread {report.dynamic.expansion_spread_db:.2f} dB <= 6.0",
        failures,
    )
    _check(
        report.transient.attack_slope_delta_db > -0.5,
        f"attack slope {report.transient.attack_slope_delta_db:+.2f} dB > -0.5",
        failures,
    )


def ablation_test(seconds: float, failures: list[str]) -> None:
    print("\nAblation -- no combination of steps may damage clean material")
    clean = {"clean_full_band": corpus.clean_full_band(seconds)}
    rows = ablation.run_matrix(clean, models.MEASURED)
    for row in rows:
        _check(
            row.correlation > NULL_MIN_CORRELATION
            and abs(row.worst_band_db) <= NULL_MAX_BAND_DEVIATION_DB
            and row.drift_ms <= NULL_MAX_DRIFT_MS
            and row.channels_preserved,
            f"[{row.label}] corr {row.correlation:.4f}, band {row.worst_band_db:+.3f} dB, "
            f"drift {row.drift_ms:.2f} ms",
            failures,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args(argv)

    failures: list[str] = []
    null_test(args.seconds, failures)
    reference_test(args.seconds, failures)
    ablation_test(args.seconds, failures)

    print()
    if failures:
        print(f"QUALITY GATE FAILED -- {len(failures)} check(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
