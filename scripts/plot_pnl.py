#!/usr/bin/env python3
"""Plot an equity curve written by run_backtest (CSV with columns: ts,equity).

Usage:  python scripts/plot_pnl.py equity_curve.csv [--out equity.png]
Requires matplotlib. If it is not installed the script prints a summary instead.
"""
import argparse
import csv
import sys


def load(path):
    ts, eq = [], []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            ts.append(int(row["ts"]))
            eq.append(float(row["equity"]))
    return ts, eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="equity.png")
    args = ap.parse_args()

    ts, eq = load(args.csv)
    if not eq:
        print("empty curve")
        return

    pnl = [e - eq[0] for e in eq]
    print(f"points={len(eq)}  start=${eq[0]:,.2f}  end=${eq[-1]:,.2f}  pnl=${pnl[-1]:,.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; printed summary only", file=sys.stderr)
        return

    xs = [t / 1e9 for t in ts]  # ns -> s
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, eq, lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("equity ($)")
    ax.set_title("Backtest equity curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
