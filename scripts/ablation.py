"""Run the per-step ablation matrix and print the table.

    python scripts/ablation.py                 # measured (badly-behaved) models
    python scripts/ablation.py --well-behaved  # control run
    python scripts/ablation.py --json out.json

Every subset of {tempo, denoise, bandwidth} is run against every corpus class
and compared with the dry input, so a regression can be attributed to a step
rather than to "the pipeline".
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import ablation, models  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--well-behaved", action="store_true",
        help="use models that behave, to isolate the chain's own effect",
    )
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args(argv)

    model_set = models.WELL_BEHAVED if args.well_behaved else models.MEASURED
    which = "well-behaved" if args.well_behaved else "measured (reproducing the reported defects)"
    print(f"Models: {which}\n")

    rows = ablation.run_matrix(ablation.default_cases(args.seconds), model_set)
    print(ablation.format_table(rows))

    full = [row for row in rows if row.steps == ablation.chain.STEPS]
    print("\nFull chain, per case:")
    for row in full:
        verdict = "fell back to dry" if row.fell_back else (
            "enhanced" if row.is_enhanced else "kept, no measurable gain"
        )
        print(f"  {row.case:<20} ran [{'+'.join(row.ran) or 'nothing'}] -> {verdict}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(row) for row in rows], indent=2, default=float), encoding="utf-8"
        )
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
