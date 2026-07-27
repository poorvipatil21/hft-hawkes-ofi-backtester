#!/usr/bin/env python3
"""
Batch-calibrates Hawkes across multiple already-converted Binance day
CSVs (from crypto_trades_to_csv.py) and aggregates multi-window
statistics -- the Binance-day counterpart to process_all_captures.py
(which handles raw Kraken .jsonl captures that still need conversion;
these Binance CSVs are already converted, so this script skips straight
to calibration).

USAGE
    python3 scripts/calibrate_all_binance_days.py
    python3 scripts/calibrate_all_binance_days.py --max-iters 20000 --pattern "data/binance_days/*_feed.csv"

OUTPUT
    Per-day table + aggregate mean/std/min/max across all days, and
    data/binance_days_summary.csv with per-day results.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import statistics as stats


def calibrate_day(csv_path: str, max_iters: int, lr: float, mu_reg: float):
    cmd = ["./bin/calibrate_hawkes", "0", "0", "-1", csv_path, str(max_iters), str(lr), str(mu_reg)]
    print(f"  calibrating: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  [ERROR] calibration failed for {csv_path}:\n{result.stderr}")
        return None
    out = result.stdout

    def find(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    converged = find(r"converged=(\d)", int)
    n_marks = find(r"\((\d+) aggressor/Hawkes marks\)", int)
    branching = re.search(
        r"branching ratio n\s*:\s*\[\[([\d.]+),\s*([\d.]+)\],\s*\[([\d.]+),\s*([\d.]+)\]\]", out)
    diag_mean = offdiag_mean = None
    if branching:
        n00, n01, n10, n11 = (float(x) for x in branching.groups())
        diag_mean = (n00 + n11) / 2.0
        offdiag_mean = (n01 + n10) / 2.0
    gof_means = re.findall(r"residual GOF mark \d+\s*:.*?mean=([\d.]+)", out)
    gof_mean_avg = sum(float(x) for x in gof_means) / len(gof_means) if gof_means else None

    return {
        "csv": csv_path,
        "n_marks": n_marks,
        "converged": converged,
        "branching_diag": diag_mean,
        "branching_offdiag": offdiag_mean,
        "gof_mean": gof_mean_avg,
    }


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    d = {"n": len(values), "mean": stats.mean(values), "min": min(values), "max": max(values)}
    d["std"] = stats.stdev(values) if len(values) > 1 else 0.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu-reg", type=float, default=0.5)
    ap.add_argument("--pattern", default="data/binance_days/*_feed.csv")
    args = ap.parse_args()

    csv_files = sorted(glob.glob(args.pattern))
    if not csv_files:
        print(f"No files matched {args.pattern}")
        return 1
    print(f"Found {len(csv_files)} day(s):")
    for f in csv_files:
        print(f"  {f}")

    results = []
    for csv_path in csv_files:
        print(f"\n=== Processing {csv_path} ===")
        res = calibrate_day(csv_path, args.max_iters, args.lr, args.mu_reg)
        if res is None:
            continue
        res["day"] = os.path.basename(csv_path)
        results.append(res)
        print(f"  n_marks={res['n_marks']}  converged={res['converged']}  "
              f"diag={res['branching_diag']}  offdiag={res['branching_offdiag']}  gof={res['gof_mean']}")

    if not results:
        print("\nNo days processed successfully.")
        return 1

    print(f"\n{'='*80}\nPER-DAY RESULTS ({len(results)} days)\n{'='*80}")
    print(f"{'day':<45} {'n_marks':>8} {'conv':>5} {'diag':>7} {'offdiag':>8} {'gof':>6}")
    for r in results:
        print(f"{r['day']:<45} {r['n_marks'] or 0:>8} {r['converged'] or 0:>5} "
              f"{r['branching_diag'] or 0:>7.4f} {r['branching_offdiag'] or 0:>8.4f} {r['gof_mean'] or 0:>6.3f}")

    print(f"\n{'='*80}\nAGGREGATE ACROSS {len(results)} DAYS\n{'='*80}")
    for label, key in [("Branching ratio (diagonal)", "branching_diag"),
                        ("Branching ratio (off-diagonal)", "branching_offdiag"),
                        ("GOF residual mean", "gof_mean")]:
        s = summarize([r[key] for r in results])
        if s:
            print(f"{label:<32}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}]  (n={s['n']})")
        else:
            print(f"{label:<32}: no data")

    n_converged = sum(1 for r in results if r["converged"] == 1)
    print(f"\nConverged: {n_converged}/{len(results)} days")
    if n_converged < len(results):
        print("WARNING: not all days converged -- consider rerunning those with "
              "higher --max-iters before trusting their individual numbers.")

    summary_path = "data/binance_days_summary.csv"
    with open(summary_path, "w") as f:
        f.write("day,n_marks,converged,branching_diag,branching_offdiag,gof_mean\n")
        for r in results:
            f.write(f"{r['day']},{r['n_marks']},{r['converged']},"
                    f"{r['branching_diag']},{r['branching_offdiag']},{r['gof_mean']}\n")
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
