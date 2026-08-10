"""Benchmark the wet/dry ratios and frequency weights, rather than assuming them.

    python scripts/sweep.py

The plan supplies starting values -- denoise 0.3/0.4/0.5/0.6, bandwidth
0.5/0.7/0.9, and a 5-10kHz denoise weight of 0.40 -- and says explicitly that
they are starting values to be benchmarked, not settled facts. This is that
benchmark.

Each setting is scored on both sides of the trade it makes:

  * **Preservation** is measured with the badly-behaved model, because that is
    what the ratio has to protect against. Worst band movement inside the
    protected range, note decay, and level-dependent gain spread.
  * **Restoration** is measured with the well-behaved model, because a ratio
    that protects perfectly by doing nothing is not a good ratio. How much of
    the high-band noise bed actually came out.

A setting is only admissible if preservation stays inside the Phase 4 budget;
among those, more restoration is better.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suno_restore import chain, quality  # noqa: E402
from suno_restore.config import Settings  # noqa: E402
from tests import corpus, models  # noqa: E402

SR = corpus.SR

BAND_BUDGET_DB = 1.5
DECAY_BUDGET_DB = -1.0
EXPANSION_BUDGET_DB = 6.0


def _worst_protected_band(report: quality.QualityReport) -> float:
    worst = 0.0
    for name, delta in report.spectral.band_delta_db.items():
        low = float(name.split("_")[0].replace("k", "000"))
        if 100 <= low < 10000 and abs(delta) > abs(worst):
            worst = delta
    return worst


def _run(audio: np.ndarray, model_set: dict, settings: Settings,
         enable: tuple[str, ...]) -> quality.QualityReport:
    outcome = chain.restore_stem(audio, SR, **model_set, enable=enable, settings=settings)
    return quality.compare(audio, outcome.audio, SR, outcome.sr)


def _isolated(audio: np.ndarray, model, settings: Settings) -> quality.QualityReport:
    """The parameter's own effect, with confidence and the damage mask removed.

    In the assembled chain these two dominate: the gate scales the wet amount by
    its confidence and the mask then confines it to where damage was found, so a
    ratio changing from 0.3 to 1.0 moves the output by a fraction of a dB. That
    is the correct behaviour and a poor benchmark -- it measures the gate, not
    the ratio. Here the step runs at full confidence over the whole stem, which
    is what the ratio would do if the evidence ever fully justified it.
    """
    from suno_restore import blend

    wet = model(audio, SR)
    blended = blend.blend_denoise(audio, wet, SR, confidence=1.0, settings=settings).audio
    return quality.compare(audio, blended, SR, SR)


def sweep_denoise_wet(seconds: float) -> None:
    print("\nDenoise wet/dry ratio")
    print("  isolated = full confidence, no mask (the ratio's own effect,"
          " against the destructive model)")
    print("  in system = after gating and masking, on the reference guitar")
    print(f"  {'wet':>5} | {'iso band':>9}{'iso decay':>10}{'iso expand':>11}{'iso 5-8k':>10}"
          f" | {'sys band':>9}{'hiss out':>10}  verdict")

    guitar = corpus.solo_guitar(seconds)
    noisy = corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0)

    for wet in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        settings = replace(Settings(), denoise_wet=wet)
        iso = _isolated(guitar, models.measured_denoise, settings)
        sys_harm = _run(guitar, models.MEASURED, settings, ("denoise",))
        good = _run(noisy, models.WELL_BEHAVED, settings, ("denoise",))

        iso_band = _worst_protected_band(iso)
        removed = -good.spectral.band_delta_db.get("10k_16k", 0.0)
        safe = (
            abs(iso_band) <= BAND_BUDGET_DB
            and iso.transient.decay_delta_db >= DECAY_BUDGET_DB
            and iso.dynamic.expansion_spread_db <= EXPANSION_BUDGET_DB
        )
        print(f"  {wet:>5.1f} | {iso_band:>9.2f}{iso.transient.decay_delta_db:>10.2f}"
              f"{iso.dynamic.expansion_spread_db:>11.2f}"
              f"{iso.spectral.band_delta_db.get('5k_8k', float('nan')):>10.2f}"
              f" | {_worst_protected_band(sys_harm):>9.2f}{removed:>10.2f}"
              f"  {'ok' if safe else 'OVER BUDGET'}")


def sweep_presence_weight(seconds: float) -> None:
    print("\nDenoise weight in 5-10kHz (the presence band)")
    print("  isolated columns show what the weight costs when the step runs at full strength")
    print(f"  {'weight':>7} | {'iso 5-8k':>10}{'iso 8-10k':>11}"
          f" | {'sys 5-8k':>10}{'sys 8-10k':>11}{'hiss out':>10}  verdict")

    guitar = corpus.solo_guitar(seconds)
    noisy = corpus.add_hiss(guitar, level_db=-30.0, sr=SR, high_pass_hz=13000.0)

    for weight in (0.0, 0.2, 0.4, 0.6, 1.0):
        settings = replace(Settings(), denoise_weight_5k_10k=weight)
        iso = _isolated(guitar, models.measured_denoise, settings)
        sys_harm = _run(guitar, models.MEASURED, settings, ("denoise",))
        good = _run(noisy, models.WELL_BEHAVED, settings, ("denoise",))

        iso_five = iso.spectral.band_delta_db.get("5k_8k", float("nan"))
        iso_eight = iso.spectral.band_delta_db.get("8k_10k", float("nan"))
        removed = -good.spectral.band_delta_db.get("10k_16k", 0.0)
        # The Phase 9 target for the reference guitar is +/-1dB here.
        safe = abs(iso_five) <= 1.0 and abs(iso_eight) <= 1.0
        print(f"  {weight:>7.1f} | {iso_five:>10.2f}{iso_eight:>11.2f}"
              f" | {sys_harm.spectral.band_delta_db.get('5k_8k', float('nan')):>10.2f}"
              f"{sys_harm.spectral.band_delta_db.get('8k_10k', float('nan')):>11.2f}"
              f"{removed:>10.2f}"
              f"  {'ok' if safe else 'OVER the +/-1dB guitar target'}")


def sweep_bandwidth_wet(seconds: float) -> None:
    print("\nBandwidth wet/dry ratio")
    print(f"  {'wet':>5}{'worst band':>12}{'HF added':>10}{'quality':>9}  verdict")

    limited = corpus.band_limit(corpus.full_mix(seconds), SR, cutoff_hz=8000.0)

    for wet in (0.5, 0.7, 0.9, 1.0):
        settings = replace(Settings(), bandwidth_wet=wet)
        outcome = chain.restore_stem(
            limited, SR, **models.WELL_BEHAVED, enable=("bandwidth",), settings=settings
        )
        report = quality.compare(limited, outcome.audio, SR, outcome.sr)

        band = _worst_protected_band(report)
        added = report.spectral.band_delta_db.get("10k_16k", 0.0)
        score = outcome.verdict.quality_score if outcome.verdict else 0.0
        safe = abs(band) <= BAND_BUDGET_DB and not outcome.fell_back
        print(f"  {wet:>5.1f}{band:>12.2f}{added:>10.2f}{score:>9.3f}"
              f"  {'ok' if safe else 'rejected'}")


def sweep_attenuation_cap(seconds: float) -> None:
    print("\nMaximum attenuation cap")
    print("  isolated: the cap is a backstop, so it only shows up when the step runs at"
          " full strength")
    print(f"  {'dB':>5} | {'iso quiet':>10}{'iso expand':>11}{'iso decay':>10}"
          f" | {'sys expand':>11}  verdict")

    guitar = corpus.solo_guitar(seconds)
    for cap in (3.0, 6.0, 9.0, 12.0, 20.0):
        settings = replace(Settings(), max_attenuation_db=cap)
        iso = _isolated(guitar, models.measured_denoise, settings)
        sys_harm = _run(guitar, models.MEASURED, settings, ("denoise",))
        safe = iso.dynamic.expansion_spread_db <= EXPANSION_BUDGET_DB
        print(f"  {cap:>5.1f} | {iso.dynamic.quiet_gain_db:>10.2f}"
              f"{iso.dynamic.expansion_spread_db:>11.2f}"
              f"{iso.transient.decay_delta_db:>10.2f}"
              f" | {sys_harm.dynamic.expansion_spread_db:>11.2f}"
              f"  {'ok' if safe else 'OVER BUDGET'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=14.0)
    args = parser.parse_args(argv)

    sweep_denoise_wet(args.seconds)
    sweep_presence_weight(args.seconds)
    sweep_bandwidth_wet(args.seconds)
    sweep_attenuation_cap(args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
