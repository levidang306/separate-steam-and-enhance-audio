"""Objective A/B comparison of two audio files, or two directories of stems.

    python scripts/ab_quality.py original.wav processed.wav
    python scripts/ab_quality.py data/input data/restored --json report.json

Directories are matched by stem name, tolerating the `_restored` suffix the
pipeline writes. Every number printed comes from `suno_restore.quality`; this
script only arranges them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/ab_quality.py` from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suno_restore import quality  # noqa: E402
from suno_restore.pipeline import AUDIO_SUFFIXES  # noqa: E402

SUFFIXES = ("_restored", "_dry", "_wet", "_processed")


def _canonical(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _pair_directories(original: Path, processed: Path) -> list[tuple[Path, Path]]:
    def index(root: Path) -> dict[str, Path]:
        return {
            _canonical(p.stem): p
            for p in sorted(root.iterdir())
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        }

    left, right = index(original), index(processed)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise SystemExit(
            f"No matching stem names between {original} and {processed}.\n"
            f"  {original.name}: {sorted(left)}\n  {processed.name}: {sorted(right)}"
        )
    for name in sorted(set(left) ^ set(right)):
        print(f"warning: '{name}' has no counterpart and was skipped", file=sys.stderr)
    return [(left[name], right[name]) for name in shared]


def _verdict(report: quality.QualityReport) -> list[str]:
    """Flag the conditions the Phase 4 gate treats as failures."""
    problems = []
    if not report.stereo.channels_preserved:
        problems.append("channel count changed")
    if abs(report.timing.duration_delta_ms) > 5.0:
        problems.append(f"duration moved {report.timing.duration_delta_ms:+.1f} ms")
    if report.timing.offset_drift_ms > 5.0:
        problems.append(f"timing drifted {report.timing.offset_drift_ms:.1f} ms")
    if report.timing.correlation_min < 0.90:
        problems.append(f"2s correlation fell to {report.timing.correlation_min:.3f}")
    for name, delta in report.spectral.band_delta_db.items():
        low = float(name.split("_")[0].replace("k", "000"))
        if 100 <= low <= 10000 and abs(delta) > 1.5:
            problems.append(f"{name} moved {delta:+.2f} dB")
    if report.transient.attack_slope_delta_db < -0.5:
        problems.append(f"attack slope lost {report.transient.attack_slope_delta_db:.2f} dB")
    if report.dynamic.expansion_spread_db > 6.0:
        problems.append(f"level-dependent gain spans {report.dynamic.expansion_spread_db:.2f} dB")
    if report.processed.clipped_samples > report.original.clipped_samples:
        problems.append(f"{report.processed.clipped_samples} clipped samples")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("original", type=Path, help="reference file or directory")
    parser.add_argument("processed", type=Path, help="file or directory to measure")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict lines")
    args = parser.parse_args(argv)

    if args.original.is_dir() != args.processed.is_dir():
        parser.error("both arguments must be files, or both directories")
    pairs = (
        _pair_directories(args.original, args.processed)
        if args.original.is_dir()
        else [(args.original, args.processed)]
    )

    payload, failed = {}, False
    for reference, candidate in pairs:
        report = quality.compare_files(reference, candidate)
        problems = _verdict(report)
        failed = failed or bool(problems)

        print(f"\n=== {reference.name}  ->  {candidate.name}")
        if not args.quiet:
            print(report.summary())
        if problems:
            print("VERDICT    FAIL")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("VERDICT    pass")
        payload[reference.stem] = {"report": report.to_dict(), "problems": problems}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
