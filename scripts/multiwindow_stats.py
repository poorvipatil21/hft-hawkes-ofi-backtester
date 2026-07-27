#!/usr/bin/env python3
"""
Computes across-window statistics for branching ratio (and other
per-window metrics) from real multi-day/multi-window calibration
results -- the actual statistical upgrade this project's multi-window
data collection was for.

SCOPE, STATED HONESTLY: this treats each window/day as one independent
observation and computes a t-distribution-based confidence interval on
the across-window mean (appropriate for small n, here n=15). This is
NOT a full two-level (within-window + between-window variance)
random-effects meta-analysis -- that would require a per-window standard
error (from bootstrapping each window individually), which costs real
additional compute time we're not spending right now. What this DOES
give you, honestly: a real, defensible confidence interval on "what is
the branching ratio, treating days as the unit of replication" -- a
substantial upgrade over a single point estimate, even without the
full two-level model.

USAGE
    python3 scripts/multiwindow_stats.py data/binance_days_summary.csv
    python3 scripts/multiwindow_stats.py data/multi_window_summary.csv --columns ofi_corr,branching_diag,branching_offdiag
"""
import argparse
import csv
import math
import sys

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def t_critical(df: float, alpha: float = 0.05) -> float:
    """95% two-sided t critical value. Uses scipy if available, else a
    small lookup table for common small-n df (good enough for n<=30),
    falling back to the normal-distribution value (1.96) for larger df
    or if scipy is unavailable and df isn't in the table -- flagged
    explicitly in the output so this approximation is never silent."""
    if HAVE_SCIPY:
        return float(scipy_stats.t.ppf(1 - alpha / 2, df))
    # Common df -> two-sided 95% t critical value (Student's t table)
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
             19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}
    if df in table:
        return table[df]
    print(f"  [note: exact t-critical for df={df} not in fallback table and scipy "
          f"unavailable; using normal approximation 1.96 -- install scipy for exact value]")
    return 1.96


def analyze_column(values, label):
    values = [v for v in values if v is not None and v == v]  # drop None/NaN
    n = len(values)
    if n < 2:
        print(f"{label}: insufficient data (n={n})")
        return
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    tcrit = t_critical(df)
    ci_lo = mean - tcrit * se
    ci_hi = mean + tcrit * se
    print(f"{label}:")
    print(f"  n = {n}, mean = {mean:.4f}, std = {sd:.4f}, SE of mean = {se:.4f}")
    print(f"  range = [{min(values):.4f}, {max(values):.4f}]")
    print(f"  95% CI on across-window mean (t-dist, df={df}): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  coefficient of variation (std/mean): {sd/mean:.3f}" if mean != 0 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--columns", default=None,
                     help="comma-separated column names to analyze (default: auto-detect "
                          "numeric columns other than obvious id/count columns)")
    args = ap.parse_args()

    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"No rows in {args.csv_path}")
        return 1

    if args.columns:
        columns = args.columns.split(",")
    else:
        skip = {"window", "day", "csv", "n_marks", "converged"}
        columns = [c for c in rows[0].keys() if c not in skip]

    print(f"Analyzing {len(rows)} windows/days from {args.csv_path}")
    print(f"Columns: {columns}\n")
    print("=" * 70)
    for col in columns:
        values = []
        for r in rows:
            raw = r.get(col, "")
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                values.append(None)
        analyze_column(values, col)
        print("-" * 70)

    if not HAVE_SCIPY:
        print("\nNOTE: scipy not installed -- t-critical values for df not in the "
              "built-in table fall back to the normal approximation (1.96). Install "
              "scipy (pip install scipy --break-system-packages) for exact values at "
              "any sample size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
