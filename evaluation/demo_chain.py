"""Demo: the alcohol -> sleep cross-system chain (brief 2.5).

Answers a query like "how will N drinks before bed affect my sleep tonight?" by
chaining hepatic -> (declared edge) -> sleep. Illustrative, not pass/fail.

Run:  python -m evaluation.demo_chain
"""

from __future__ import annotations

from modules.hepatic import Drink
from modules.sleep import SleepModel
from pipeline.graph import run_alcohol_then_sleep

PROFILE = dict(weight_kg=80, sex="male", age=35, bedtime_hour=2.0)


def main():
    model = SleepModel().fit()  # train once, reuse across scenarios
    print("Scenario: drinks consumed at t=0, bedtime 2 h later (80 kg male, 35 y)\n")
    print(f"{'drinks':>6} {'bedtime BAC':>12} {'alc g/kg':>9} "
          f"{'REM %':>16} {'awakenings':>16}")
    print("-" * 64)
    for n in (0, 1, 2, 4, 6):
        drinks = [Drink.standard(n, hour=0.0)] if n else []
        res = run_alcohol_then_sleep(drinks=drinks, sleep_model=model, **PROFILE)
        rem_lo, rem_hi = res.sleep.intervals["rem_pct"]
        awk_lo, awk_hi = res.sleep.intervals["awakenings"]
        print(f"{n:>6} {res.bedtime_bac:>12.4f} {res.alcohol_gkg_bedtime:>9.3f} "
              f"{res.sleep.metrics['rem_pct']:>7.1f} [{rem_lo:.0f},{rem_hi:.0f}] "
              f"{res.sleep.metrics['awakenings']:>8.2f} [{awk_lo:.1f},{awk_hi:.1f}]")
    print("-" * 64)
    print(f"evidence: {res.evidence.value}")
    print(f"label   : {res.confidence_label}")
    print("\nMore bedtime alcohol -> higher bedtime BAC -> more g/kg burden ->\n"
          "lower REM and more awakenings, chained through the declared edge.")


if __name__ == "__main__":
    main()
