#!/usr/bin/env python3
"""
Converts a capture from capture_bitmex_ws.py (.jsonl of raw BitMEX
WebSocket messages) into this project's tape schema:
    ts,type,side,price_ticks,qty,id

BITMEX'S FORMAT DIFFERS FROM KRAKEN'S: BitMEX's orderBookL2_25 stream is
keyed by a unique per-level integer `id` (not by price), so this
converter tracks id -> (side, price) as levels are first seen, since
"update" and "delete" messages only reference the id, not the price:
  - action="partial" (first message) or "insert": entry includes
    {id, side, price, size} -- remember id -> price, emit Add.
  - action="update": entry includes {id, side, size} only (no price --
    price is immutable for a given id; only size changes) -- look up the
    remembered price, emit Cancel(id) then Add(id, remembered_price,
    new_size).
  - action="delete": entry includes {id, side} only -- emit Cancel(id),
    forget the id.
This is a real structural difference from Kraken (price-keyed, so we had
to synthesize our own ids there) worth being explicit about rather than
reusing kraken_ws_to_csv.py's logic unchanged.

Trade messages (table="trade", action="insert") give side/price/size
directly with side already representing the taker/aggressor side (same
convention as Kraken's trade channel -- no inversion needed).

Timestamps use each message's local receive time (recv_ts_ns, written by
the capture script), same convention as kraken_ws_to_csv.py, for the
same reason: reflects the actual arrival-time process a live strategy
would see, and sidesteps BitMEX's own ISO8601 timestamp string parsing.

Usage:
    python scripts/bitmex_ws_to_csv.py \
        data/bitmex_ws_capture_xbtusd_....jsonl \
        data/bitmex_feed.csv \
        --tick-size 0.5 --qty-scale 1

    Note on --qty-scale: BitMEX XBTUSD sizes are already integer USD
    contracts (not fractional coin units like Kraken's BTC), so the
    default qty_scale=1 (no rescaling) is usually correct here, unlike
    Kraken's --qty-scale 1e6. Double-check against your actual data
    before trusting downstream PnL figures (see the qty-scale bug this
    project already found once for why this matters).
"""
import argparse
import csv
import json
import sys


def convert(in_path: str, out_path: str, tick_size: float, qty_scale: float):
    id_to_price = {}   # BitMEX id -> (side_str, price) as last seen

    n_add = n_cancel = n_trade = 0
    n_book_msgs = n_trade_msgs = n_skipped = n_malformed = 0

    with open(in_path) as fin, open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["ts", "type", "side", "price_ticks", "qty", "id"])

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                wrapper = json.loads(line)
                recv_ts_ns = wrapper["recv_ts_ns"]
                msg = wrapper["data"]
            except (json.JSONDecodeError, KeyError):
                n_malformed += 1
                continue

            table = msg.get("table")
            action = msg.get("action")

            if table == "orderBookL2_25":
                n_book_msgs += 1
                for entry in msg.get("data", []):
                    try:
                        level_id = entry["id"]
                        side_str = entry["side"]          # "Buy" or "Sell" (BitMEX capitalization)
                        tape_side = "buy" if side_str == "Buy" else "sell"
                    except (KeyError, TypeError):
                        n_malformed += 1
                        continue

                    if action in ("partial", "insert"):
                        try:
                            price = float(entry["price"])
                            size = float(entry["size"])
                        except (KeyError, TypeError, ValueError):
                            n_malformed += 1
                            continue
                        price_ticks = round(price / tick_size)
                        qty_scaled = round(size * qty_scale)
                        if qty_scaled <= 0:
                            continue
                        # If this id was already tracked (shouldn't normally
                        # happen for insert/partial, but be defensive),
                        # cancel the stale one first.
                        if level_id in id_to_price:
                            writer.writerow([recv_ts_ns, "cancel", tape_side, price_ticks, 0, level_id])
                            n_cancel += 1
                        id_to_price[level_id] = (tape_side, price)
                        writer.writerow([recv_ts_ns, "add", tape_side, price_ticks, qty_scaled, level_id])
                        n_add += 1

                    elif action == "update":
                        try:
                            size = float(entry["size"])
                        except (KeyError, TypeError, ValueError):
                            n_malformed += 1
                            continue
                        prev = id_to_price.get(level_id)
                        if prev is None:
                            # Update for a level we never saw inserted (e.g.
                            # existed before capture started) -- nothing to
                            # correctly re-add at; skip rather than guess a price.
                            continue
                        prev_side, price = prev
                        price_ticks = round(price / tick_size)
                        qty_scaled = round(size * qty_scale)
                        writer.writerow([recv_ts_ns, "cancel", prev_side, price_ticks, 0, level_id])
                        n_cancel += 1
                        if qty_scaled > 0:
                            writer.writerow([recv_ts_ns, "add", prev_side, price_ticks, qty_scaled, level_id])
                            n_add += 1
                        else:
                            del id_to_price[level_id]
                            continue
                        id_to_price[level_id] = (prev_side, price)

                    elif action == "delete":
                        prev = id_to_price.pop(level_id, None)
                        if prev is None:
                            continue  # deleting a level we never tracked -- nothing to cancel
                        prev_side, price = prev
                        price_ticks = round(price / tick_size)
                        writer.writerow([recv_ts_ns, "cancel", prev_side, price_ticks, 0, level_id])
                        n_cancel += 1

            elif table == "trade":
                n_trade_msgs += 1
                for entry in msg.get("data", []):
                    try:
                        side_str = entry["side"]
                        tape_side = "buy" if side_str == "Buy" else "sell"
                        price = float(entry["price"])
                        size = float(entry["size"])
                        trade_id = entry.get("trdMatchID", n_trade)
                    except (KeyError, TypeError, ValueError):
                        n_malformed += 1
                        continue
                    price_ticks = round(price / tick_size)
                    qty_scaled = round(size * qty_scale)
                    if qty_scaled <= 0:
                        continue
                    # trdMatchID is a UUID string, not usable as our integer
                    # order id -- use a simple incrementing counter instead
                    # (trade rows don't need to match any resting order id).
                    writer.writerow([recv_ts_ns, "trade", tape_side, price_ticks, qty_scaled, n_trade + 1])
                    n_trade += 1

            else:
                n_skipped += 1

    print(f"parsed {n_book_msgs} book messages, {n_trade_msgs} trade messages "
          f"({n_skipped} other/status messages, {n_malformed} malformed entries skipped)")
    print(f"wrote add={n_add} cancel={n_cancel} trade={n_trade} to {out_path}")
    if n_add == 0:
        print("WARNING: 0 add rows -- check the capture file actually has orderBookL2_25 "
              "partial/insert messages (run with --peek to inspect raw contents).")


def peek(in_path: str, n: int):
    with open(in_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            wrapper = json.loads(line)
            print(f"--- message {i} ---")
            print(json.dumps(wrapper, indent=2)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl")
    ap.add_argument("output_csv", nargs="?", default=None)
    ap.add_argument("--tick-size", type=float, default=0.5,
                     help="dollar price per tick (default 0.5, matches XBTUSD's typical tick size)")
    ap.add_argument("--qty-scale", type=float, default=1.0,
                     help="multiply raw size by this before rounding to an integer unit "
                          "(default 1.0 -- BitMEX XBTUSD sizes are already integer USD "
                          "contracts, unlike Kraken's fractional BTC, so no rescaling is "
                          "usually needed; verify against your actual data)")
    ap.add_argument("--peek", type=int, default=None, metavar="N")
    args = ap.parse_args()

    if args.peek is not None:
        peek(args.input_jsonl, args.peek)
        sys.exit(0)

    if args.output_csv is None:
        sys.exit("output_csv is required unless --peek is given")
    convert(args.input_jsonl, args.output_csv, args.tick_size, args.qty_scale)
