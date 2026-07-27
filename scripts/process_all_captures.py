#!/usr/bin/env python3
"""
Batch-processes every Kraken capture window in data/*.jsonl end to end:
convert -> calibrate -> aggregate multi-window statistics. This is the
script that turns Tier-1 (multiple independent capture windows) into
actual reportable numbers -- a mean/range across windows instead of the
single-window point estimates every result in this project has been
until now.

Run this once all (or however many) capture_loop.py windows have
finished. It's safe to re-run: already-converted files are skipped
(unless --force), so you can run it partway through a long capture_loop
run to see preliminary results, then re-run later as more windows finish.

USAGE
    python3 scripts/process_all_captures.py
    python3 scripts/process_all_captures.py --max-iters 20000 --force

OUTPUT
    - Converts each data/kraken_ws_capture_*.jsonl -> data/*_converted.csv
      (via kraken_ws_to_csv.py, same --tick-size/--qty-scale used
      throughout this project: 0.1 / 1e6)
    - Runs calibrate_hawkes on each converted CSV, parsing its output for
      the two headline statistics (OFI correlation, branching ratio matrix)
    - Prints a per-window table plus aggregate mean/std/min/max across all
      successfully processed windows
    - Writes data/multi_window_summary.csv with per-window results, ready
      to drop into the paper as a real multi-window table
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import statistics as stats


def convert_window(jsonl_path: str, tick_size: float, qty_scale: float, force: bool) -> str:
    base = os.path.splitext(os.path.basename(jsonl_path))[0]
    out_csv = f"data/{base}_converted.csv"
    if os.path.exists(out_csv) and not force:
        print(f"  [skip] {out_csv} already exists (use --force to reconvert)")
        return out_csv
    cmd = [sys.executable, "scripts/kraken_ws_to_csv.py", jsonl_path, out_csv,
           "--tick-size", str(tick_size), "--qty-scale", str(qty_scale)]
    print(f"  converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] conversion failed for {jsonl_path}:\n{result.stderr}")
        return None
    print(f"  -> {out_csv}")
    return out_csv


def calibrate_window(csv_path: str, max_iters: int, lr: float, mu_reg: float):
    cmd = ["./bin/calibrate_hawkes", "0", "0", "-1", csv_path, str(max_iters), str(lr), str(mu_reg)]
    print(f"  calibrating: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  [ERROR] calibration failed for {csv_path}:\n{result.stderr}")
        return None
    out = result.stdout

    # Forward any warnings calibrate_hawkes printed -- these were previously
    # silently swallowed since only specific fields below were surfaced,
    # which meant the degenerate-fit diagnostic (added specifically to catch
    # this) never reached anyone actually looking at this script's output.
    for line in out.splitlines():
        if "WARNING" in line or line.strip().startswith("***"):
            print(f"  {line}")

    def find(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    ofi_corr = find(r"OFI vs fwd-return.*?corr\s*:\s*(-?[\d.]+)")
    converged = find(r"converged=(\d)", int)
    n_marks = find(r"\((\d+) aggressor/Hawkes marks\)", int)
    branching = re.search(
        r"branching ratio n\s*:\s*\[\[([\d.]+),\s*([\d.]+)\],\s*\[([\d.]+),\s*([\d.]+)\]\]", out)
    diag_mean = offdiag_mean = None
    is_degenerate = False
    if branching:
        n00, n01, n10, n11 = (float(x) for x in branching.groups())
        diag_mean = (n00 + n11) / 2.0
        offdiag_mean = (n01 + n10) / 2.0
        # Same suspiciously-uniform check as calibrate_hawkes.cpp's own
        # diagnostic, computed independently here so it can't be silently
        # missed even if a raw WARNING line is somehow not forwarded above.
        vals = [n00, n01, n10, n11]
        mean_v = sum(vals) / 4.0
        sd_v = (sum((v - mean_v) ** 2 for v in vals) / 4.0) ** 0.5
        is_degenerate = sd_v < 0.02 and mean_v > 0.05

    return {
        "csv": csv_path,
        "n_marks": n_marks,
        "ofi_corr": ofi_corr,
        "converged": converged,
        "branching_diag": diag_mean,
        "branching_offdiag": offdiag_mean,
        "is_degenerate": is_degenerate,
        "raw_output": out,
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
    ap.add_argument("--tick-size", type=float, default=0.1)
    ap.add_argument("--qty-scale", type=float, default=1e6)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mu-reg", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="reconvert/recalibrate even if output already exists")
    ap.add_argument("--pattern", default="data/kraken_ws_capture_*.jsonl")
    args = ap.parse_args()

    jsonl_files = sorted(glob.glob(args.pattern))
    if not jsonl_files:
        print(f"No files matched {args.pattern}")
        return 1
    print(f"Found {len(jsonl_files)} capture window(s):")
    for f in jsonl_files:
        print(f"  {f}")

    results = []
    for jsonl_path in jsonl_files:
        print(f"\n=== Processing {jsonl_path} ===")
        csv_path = convert_window(jsonl_path, args.tick_size, args.qty_scale, args.force)
        if csv_path is None:
            continue
        res = calibrate_window(csv_path, args.max_iters, args.lr, args.mu_reg)
        if res is None:
            continue
        res["window"] = os.path.basename(jsonl_path)
        results.append(res)
        print(f"  n_marks={res['n_marks']}  OFI_corr={res['ofi_corr']}  "
              f"converged={res['converged']}  diag={res['branching_diag']}  "
              f"offdiag={res['branching_offdiag']}")

    if not results:
        print("\nNo windows processed successfully.")
        return 1

    print(f"\n{'='*70}\nPER-WINDOW RESULTS ({len(results)} windows)\n{'='*70}")
    print(f"{'window':<45} {'n_marks':>8} {'OFI_corr':>9} {'conv':>5} {'diag':>7} {'offdiag':>8}  {'status'}")
    for r in results:
        if not r["converged"]:
            status = "UNCONVERGED -- exclude"
        elif r["is_degenerate"]:
            status = "DEGENERATE -- exclude"
        else:
            status = "ok"
        print(f"{r['window']:<45} {r['n_marks'] or 0:>8} {r['ofi_corr'] or 0:>9.4f} "
              f"{r['converged'] or 0:>5} {r['branching_diag'] or 0:>7.4f} {r['branching_offdiag'] or 0:>8.4f}  {status}")

    trustworthy = [r for r in results if r["converged"] == 1 and not r["is_degenerate"]]
    excluded = [r for r in results if r not in trustworthy]

    print(f"\n{'='*70}\nAGGREGATE -- TRUSTWORTHY SUBSET ONLY ({len(trustworthy)}/{len(results)} windows: "
          f"converged AND not flagged degenerate)\n{'='*70}")
    if excluded:
        print("Excluded from this aggregate:")
        for r in excluded:
            reason = "unconverged" if not r["converged"] else "degenerate (suspiciously uniform branching ratio)"
            print(f"  {r['window']} -- {reason}")
        print()
    for label, key in [("OFI correlation", "ofi_corr"),
                        ("Branching ratio (diagonal)", "branching_diag"),
                        ("Branching ratio (off-diagonal)", "branching_offdiag")]:
        s = summarize([r[key] for r in trustworthy])
        if s:
            print(f"{label:<32}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}]  (n={s['n']})")
        else:
            print(f"{label:<32}: no trustworthy data")

    if len(trustworthy) < 3:
        print(f"\nNOTE: only {len(trustworthy)} trustworthy window(s) remain -- too few for a "
              f"meaningful aggregate statistic. Consider rerunning excluded windows with a much "
              f"higher --max-iters, or gathering additional independent captures.")

    summary_path = "data/multi_window_summary.csv"
    with open(summary_path, "w") as f:
        f.write("window,n_marks,ofi_corr,converged,branching_diag,branching_offdiag,is_degenerate,trustworthy\n")
        for r in results:
            is_trustworthy = r in trustworthy
            f.write(f"{r['window']},{r['n_marks']},{r['ofi_corr']},{r['converged']},"
                    f"{r['branching_diag']},{r['branching_offdiag']},{r['is_degenerate']},{is_trustworthy}\n")
    print(f"\nSummary written to {summary_path} (includes is_degenerate and trustworthy columns -- "
          f"filter to trustworthy=True before computing any statistic you plan to report)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
