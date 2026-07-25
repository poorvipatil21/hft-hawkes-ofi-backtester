#!/usr/bin/env python3
"""
Convert a Kaggle/Binance-style aggTrades CSV into the tape schema this
project's load_csv_feed() expects: ts,type,side,price_ticks,qty,id

Free data sources this targets:
  - Kaggle "Binance Full History" / any "aggTrades" dump
    (columns: agg_trade_id,price,quantity,first_trade_id,last_trade_id,
     transact_time,is_buyer_maker[,is_best_match])
  - Binance's own public data.binance.vision aggTrades archives (same schema,
    free, no Kaggle account needed either)

IMPORTANT LIMITATION: aggTrades-only data has no order-book depth, so this
gives you a genuine (clustered) Trade/aggressor-arrival stream for Hawkes
calibration, but OFI needs Add/Cancel depth events, which this data doesn't
have. All rows here are emitted as type=trade; there are no Add/Cancel rows.
This is fine for `calibrate_hawkes` (it drives the Hawkes fit off Trade
marks) but calibrate_hawkes's OFI-vs-forward-return correlation number will
be ~0 / meaningless on output from this script, since RollingOFI never sees
real depth changes. To get real OFI you need an L2 depth-snapshot dataset
(e.g. Binance's own depth-snapshot archives, or a Kaggle L2-orderbook set --
different from aggTrades) and a second, depth-aware converter.

Usage:
    python scripts/crypto_trades_to_csv.py trades.csv data/crypto_feed.csv \
        --tick-size 0.01

Then:
    ./bin/calibrate_hawkes 0 0 -1 data/crypto_feed.csv
    (n/seed args are ignored when a csv path is given; beta=-1 auto-scales)
"""
import argparse
import csv
import sys


def convert(in_path: str, out_path: str, tick_size: float, qty_scale: float, price_col: str,
            qty_col: str, ts_col: str, maker_col: str):
    with open(in_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["ts", "type", "side", "price_ticks", "qty", "id"])

        missing = [c for c in (price_col, qty_col, ts_col, maker_col) if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(
                f"Column(s) {missing} not found in {in_path}. "
                f"Available columns: {reader.fieldnames}. "
                f"Pass --price-col/--qty-col/--ts-col/--maker-col to match your file."
            )

        n = 0
        for i, row in enumerate(reader):
            price = float(row[price_col])
            qty = float(row[qty_col])
            ts_raw = row[ts_col]
            # Binance transact_time is epoch millis; keep as integer ns-ish
            # (just needs to be monotonic increasing -- Hawkes/OFI code only
            # cares about relative time deltas, not the unit).
            ts = int(float(ts_raw))

            is_buyer_maker = str(row[maker_col]).strip().lower() in ("true", "1")
            # is_buyer_maker=True  -> buyer was resting (maker) -> aggressor sold -> side=sell
            # is_buyer_maker=False -> buyer was the taker (aggressor)            -> side=buy
            side = "sell" if is_buyer_maker else "buy"

            price_ticks = round(price / tick_size)
            qty_scaled = round(qty * qty_scale)
            if qty_scaled <= 0:
                continue   # dust trade that rounds to 0 units -- not representable, skip
            writer.writerow([ts, "trade", side, price_ticks, qty_scaled, i + 1])
            n += 1

        print(f"wrote {n} trade rows to {out_path}")
        print("NOTE: trade-only tape -- Hawkes calibration is valid, OFI will read as ~0.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--qty-scale", type=float, default=1e6,
                     help="multiply raw qty by this before rounding to an integer "
                          "unit (engine's Quantity is int64) -- e.g. 1e6 turns BTC "
                          "into micro-BTC so sub-1-BTC trades don't round to 0")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--qty-col", default="quantity")
    ap.add_argument("--ts-col", default="transact_time")
    ap.add_argument("--maker-col", default="is_buyer_maker")
    args = ap.parse_args()
    convert(args.input_csv, args.output_csv, args.tick_size, args.qty_scale,
            args.price_col, args.qty_col, args.ts_col, args.maker_col)
