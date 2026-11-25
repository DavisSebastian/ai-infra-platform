"""Week 7: how many requests does green need to serve before the bad card is noticed? And how often does it cry wolf?

Re-draws smaller windows from captured scenarios many times and counts how often the ALERT condition holds for the
check that matters in scenario C, answer length: green's answers differ from BLUE's (same users, same period) AND
from the baseline's, both by K-S at p < 0.05. That is decide()'s "one check confirmed by both comparisons" rule.

    uv run python week07-drift/scripts/detection_curve.py

  scenario C (green on card v3)  -> detection rate: how often a window of n green requests would raise the alert
  scenario A (both on card v1)   -> false-alarm rate: how often the same rule fires with nothing wrong

It uses scipy's K-S test directly (the test Evidently runs for numeric columns) so it can redo the comparison
thousands of times in seconds; it ignores decide()'s 50-record minimum on purpose, to show where that minimum sits.
"""

import random
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

DATA = Path(__file__).resolve().parent.parent / "data"
SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 60]
DRAWS = 200
warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")  # it falls back, which is fine


def words(window: pd.DataFrame) -> list[int]:
    return window["answer"].fillna("").str.split().str.len().tolist()


def alert_rate(scenario: pd.DataFrame, baseline: list[int], n: int) -> float:
    green, blue = words(scenario[scenario["side"] == "green"]), words(scenario[scenario["side"] == "blue"])
    hits = 0
    for draw in range(DRAWS):
        rng = random.Random(draw)
        g, b = rng.sample(green, n), rng.sample(blue, n)
        hits += ks_2samp(g, b).pvalue < 0.05 and ks_2samp(g, baseline).pvalue < 0.05
    return hits / DRAWS


def main() -> None:
    baseline = words(pd.read_parquet(DATA / "baseline.parquet"))
    bad = pd.read_parquet(DATA / "scenario-c-green-silent-failure.parquet")
    fine = pd.read_parquet(DATA / "scenario-a-canary-no-change.parquet")
    g, b = bad[bad["side"] == "green"], bad[bad["side"] == "blue"]
    print(f"answer words, median: baseline {pd.Series(baseline).median():.0f}, "
          f"blue (v1) {pd.Series(words(b)).median():.0f}, green (v3) {pd.Series(words(g)).median():.0f}\n")  # fmt: skip
    header = ("green requests", "alert raised (card v3)", "false alarm (no change)", "total at 10% canary")
    print("  " + "  ".join(f"{h:>23}" for h in header))
    for n in SIZES:
        row = (f"{n}", f"{alert_rate(bad, baseline, n):.0%}", f"{alert_rate(fine, baseline, n):.0%}", f"{n * 10}")
        print("  " + "  ".join(f"{v:>23}" for v in row))


if __name__ == "__main__":
    main()
