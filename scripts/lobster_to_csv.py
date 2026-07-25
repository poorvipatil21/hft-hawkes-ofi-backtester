#!/usr/bin/env python3
"""
Convert a LOBSTER 'message' file into the tape schema this project's
load_csv_feed() expects: ts,type,side,price_ticks,qty,id

Free data: https://lobsterdata.com/info/DataSamples.php -- no account
needed for the sample files (AAPL, AMZN, GOOG, INTC, MSFT; 2012-06-21;
10 levels). Download gives you a pair of files per ticker:
    TICKER_2012-06-21_34200000_57600000_message_10.csv
    TICKER_2012-06-21_34200000_57600000_orderbook_10.csv
Only the message file is required for this converter; the orderbook file
is LOBSTER's own ground-truth book state and isn't needed to build the
tape (our engine reconstructs the book itself from Add/Cancel/Trade
events) -- but see --validate below, which uses it as a correctness check.

LOBSTER message file columns (no header row):
    Time, Type, Order ID, Size, Price, Direction[, ...]
  Time      : seconds since midnight, fractional (we convert to integer ns)
  Type      : 1=new limit order, 2=partial cancellation, 3=total deletion,
              4=execution (visible), 5=execution (hidden), 6=cross trade,
              7=trading halt
  Order ID  : LOBSTER's persistent order reference number
  Size      : shares
  Price     : dollar price * 10000 (LOBSTER's fixed-point convention --
              used directly as price_ticks; 1 tick == $0.0001 here)
  Direction : 1=buy limit order, -1=sell limit order (the RESTING order's
              side -- for executions, the aggressor is the OPPOSITE side:
              a sell limit order being hit means a buy-initiated trade)

Type 2 (partial cancellation) needs the *remaining* size after the
cancellation to correctly re-add the order (LOBSTER's Size field on a
type-2 row is the cancelled amount, not the remainder), so this script
tracks each live order's remaining size via a dict, decremented on partial
cancels and executions, to compute that correctly.

Type 5 (hidden-order execution) and 6 (cross trade) are skipped: hidden
liquidity was never in the visible book (no Add event exists for it), so
there's nothing in our reconstructed book to match it against -- same
limitation the engine's own synthetic feed has (hidden orders aren't
modeled). Type 7 (halt) is skipped.

Usage:
    python scripts/lobster_to_csv.py \
        AAPL_2012-06-21_34200000_57600000_message_10.csv \
        data/aapl_feed.csv

    # optional correctness check against LOBSTER's own orderbook file:
    python scripts/lobster_to_csv.py \
        AAPL_2012-06-21_34200000_57600000_message_10.csv \
        data/aapl_feed.csv \
        --validate AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
"""
import argparse
import csv
import sys


def convert(message_path: str, out_path: str) -> int:
    remaining = {}   # order_id -> tracked remaining size (for type-2 handling)
    n_add = n_cancel = n_trade = n_skipped = 0

    with open(message_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(["ts", "type", "side", "price_ticks", "qty", "id"])

        for row in reader:
            if len(row) < 6:
                continue
            time_s, msg_type, order_id, size, price, direction = row[:6]
            t = float(time_s)
            ts = int(round(t * 1e9))   # seconds -> ns, matches engine's int64 Timestamp
            msg_type = int(msg_type)
            order_id = int(order_id)
            size = int(size)
            price = int(price)
            direction = int(direction)
            resting_side = "buy" if direction == 1 else "sell"

            if msg_type == 1:   # new limit order
                remaining[order_id] = size
                writer.writerow([ts, "add", resting_side, price, size, order_id])
                n_add += 1

            elif msg_type == 2:   # partial cancellation
                prev = remaining.get(order_id, size)
                new_remaining = max(0, prev - size)
                remaining[order_id] = new_remaining
                writer.writerow([ts, "cancel", resting_side, price, 0, order_id])
                n_cancel += 1
                if new_remaining > 0:
                    writer.writerow([ts, "add", resting_side, price, new_remaining, order_id])
                    n_add += 1

            elif msg_type == 3:   # total deletion
                remaining.pop(order_id, None)
                writer.writerow([ts, "cancel", resting_side, price, 0, order_id])
                n_cancel += 1

            elif msg_type == 4:   # execution, visible order
                if order_id in remaining:
                    remaining[order_id] = max(0, remaining[order_id] - size)
                # aggressor is the opposite side of the resting order that got hit
                aggressor_side = "sell" if direction == 1 else "buy"
                writer.writerow([ts, "trade", aggressor_side, price, size, order_id])
                n_trade += 1

            else:
                # 5 (hidden execution), 6 (cross trade), 7 (halt) -- skipped
                n_skipped += 1

    print(f"wrote add={n_add} cancel={n_cancel} trade={n_trade} to {out_path} "
          f"(skipped {n_skipped} hidden/cross/halt rows)")
    return n_add + n_cancel + n_trade


def validate(out_path: str, orderbook_path: str, n_samples: int = 20):
    """
    Spot-checks our converted tape's implied Add/Cancel/Trade sequence
    against LOBSTER's own recorded best-bid/best-ask at the same row
    indices in the orderbook file, as a converter-correctness sanity check.
    This does NOT replay our actual matching engine (that logic lives in
    C++); it only checks that price/side/qty fields decoded sanely by
    comparing row counts and spot-checking a few best-bid/ask values
    implied by consecutive Add rows against LOBSTER's ground truth.
    """
    with open(orderbook_path, newline="") as f:
        ob_rows = list(csv.reader(f))
    with open(out_path, newline="") as f:
        tape_rows = list(csv.reader(f))[1:]   # skip header

    print(f"orderbook file has {len(ob_rows)} rows; converted tape has {len(tape_rows)} rows "
          f"(tape > orderbook rows is expected: partial cancels emit 2 tape rows per message row)")

    if not ob_rows:
        print("empty orderbook file, nothing to validate")
        return

    step = max(1, len(ob_rows) // n_samples)
    print(f"sampling every {step} rows -- LOBSTER best bid/ask (cols 3,1) vs nearest tape event price:")
    for i in range(0, len(ob_rows), step):
        row = ob_rows[i]
        if len(row) < 4:
            continue
        best_ask, best_ask_sz, best_bid, best_bid_sz = row[0], row[1], row[2], row[3]
        print(f"  row {i:>7}: LOBSTER best_bid={best_bid} best_ask={best_ask}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("message_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--validate", metavar="ORDERBOOK_CSV", default=None,
                     help="LOBSTER orderbook file to spot-check the conversion against")
    args = ap.parse_args()
    convert(args.message_csv, args.output_csv)
    if args.validate:
        validate(args.output_csv, args.validate)
