#!/usr/bin/env python3
"""Generate a synthetic L2 market-event CSV the engine can replay via load_csv_feed().

Columns: ts,type,side,price_ticks,qty,id
  type : add | cancel | trade      side : buy | sell
Prices are integer ticks. The mid follows a bounded random walk; resting liquidity
is posted a couple of ticks off the touch and aggressor trades sweep inward.

Usage:  python scripts/generate_data.py --events 500000 --out data/feed.csv --seed 7
"""
import argparse
import csv
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=500_000)
    ap.add_argument("--out", default="data/feed.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-tick", type=int, default=10_000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    mid = args.start_tick
    ts = 0
    next_id = (1 << 40) + 1  # keep above the engine's market-id base

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ts", "type", "side", "price_ticks", "qty", "id"])
        for _ in range(args.events):
            ts += rng.randint(50, 500)                 # ns between events
            mid = max(100, mid + rng.choice((-1, 0, 0, 1)))
            roll = rng.random()
            if roll < 0.55:                            # add resting liquidity off the touch
                side = "buy" if rng.random() < 0.5 else "sell"
                off = 2 + rng.randint(0, 6)
                px = mid - off if side == "buy" else mid + off
                w.writerow([ts, "add", side, px, rng.randint(1, 20), next_id])
                next_id += 1
            else:                                      # aggressor trade sweeping the touch
                side = "buy" if rng.random() < 0.5 else "sell"
                w.writerow([ts, "trade", side, mid, rng.randint(1, 15), next_id])
                next_id += 1

    print(f"wrote {args.events} events -> {args.out}")


if __name__ == "__main__":
    main()
