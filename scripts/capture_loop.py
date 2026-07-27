#!/usr/bin/env python3
"""
Runs multiple independent Kraken capture windows back-to-back, fully
automated -- start it once and let it run, rather than manually
re-launching capture_kraken_ws.py for each window.

WHY THIS MATTERS FOR THIS PROJECT
    Every headline statistic (branching ratio, OFI correlation) so far is
    from a single capture window per venue (n=1). Getting from n=1 to
    n=5-7 independent windows (different times of day, different days) is
    the single highest-value remaining improvement to the paper's
    empirical section -- this script is what actually produces that data
    without you needing to sit at the terminal re-typing a command every
    few hours.

RESILIENCE
    If one window's capture process crashes or the connection can't be
    re-established even after capture_kraken_ws.py's own internal
    reconnect logic gives up, this script logs the failure and moves on
    to the NEXT window rather than stopping the whole run -- a single bad
    window shouldn't cost you all the others.

USAGE
    python3 scripts/capture_loop.py --count 5 --duration 21600 --gap 300
        runs 5 windows of 6 hours each, with a 5-minute gap between them
        (needed so each window's WebSocket connection is DID close before
        rate-limit-sensitive reconnection could look automated/abusive to
        Kraken -- a small courtesy gap, not a requirement, but cheap to
        include). Total wall-clock: ~30.4 hours.

    python3 scripts/capture_loop.py --count 3 --duration 7200
        3 windows of 2 hours each, back-to-back (no gap).

IMPORTANT -- THIS SCRIPT ONLY HELPS IF YOUR MACHINE STAYS ON.
    Python cannot prevent your laptop from sleeping when the lid closes --
    that's an OS-level power setting. On Windows, run this once (as
    Administrator, in PowerShell) before starting a long capture run:

        powercfg /change standby-timeout-ac 0
        powercfg /change hibernate-timeout-ac 0
        powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0

    The last line specifically stops "closing the lid" from sleeping the
    machine while it's plugged into power (LIDACTION 0 = "do nothing").
    Make sure it's plugged in -- these settings only affect the "AC"
    (plugged-in) power profile, not battery, so the laptop will still
    sleep on battery as normal, which is what you want (don't want it
    silently draining/overheating in a bag). To undo after your capture
    run is done: powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, required=True, help="number of independent capture windows to run")
    ap.add_argument("--duration", type=float, required=True, help="seconds per window")
    ap.add_argument("--gap", type=float, default=60, help="seconds to wait between windows (default 60)")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--depth", type=int, default=25, choices=[10, 25, 100, 500, 1000])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--script", default="scripts/capture_kraken_ws.py",
                     help="path to the underlying single-capture script")
    args = ap.parse_args()

    total_hours = (args.count * args.duration + max(0, args.count - 1) * args.gap) / 3600
    log(f"Starting {args.count} capture windows of {args.duration/3600:.2f}h each, "
        f"{args.gap}s gap between -- total wall-clock ~{total_hours:.1f}h")

    results = []
    for i in range(1, args.count + 1):
        log(f"=== Window {i}/{args.count}: starting ===")
        cmd = [
            sys.executable, args.script,
            "--symbol", args.symbol,
            "--depth", str(args.depth),
            "--duration", str(args.duration),
            "--out-dir", args.out_dir,
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=args.duration + 120)  # small grace period over the target duration
            elapsed = time.time() - t0
            if proc.returncode == 0:
                log(f"=== Window {i}/{args.count}: finished OK after {elapsed:.0f}s ===")
                results.append((i, "ok", elapsed))
            else:
                log(f"=== Window {i}/{args.count}: exited with code {proc.returncode} after {elapsed:.0f}s -- "
                    f"continuing to next window ===")
                results.append((i, f"exit_code_{proc.returncode}", elapsed))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log(f"=== Window {i}/{args.count}: TIMED OUT after {elapsed:.0f}s (hung past duration+grace) -- "
                f"killed, continuing to next window ===")
            results.append((i, "timeout", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            log(f"=== Window {i}/{args.count}: exception {e!r} after {elapsed:.0f}s -- continuing ===")
            results.append((i, f"exception_{type(e).__name__}", elapsed))

        if i < args.count:
            log(f"Sleeping {args.gap}s before next window...")
            time.sleep(args.gap)

    log("=== All windows attempted. Summary: ===")
    for i, status, elapsed in results:
        log(f"  window {i}: {status} ({elapsed:.0f}s)")
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    log(f"{n_ok}/{args.count} windows completed successfully.")
    log(f"Check {args.out_dir}/*.jsonl for the resulting capture files.")


if __name__ == "__main__":
    main()
