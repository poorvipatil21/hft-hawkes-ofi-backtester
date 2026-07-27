#!/usr/bin/env python3
"""
Converts a capture from capture_kraken_ws.py (.jsonl of raw Kraken
WebSocket v2 messages) into this project's tape schema:
    ts,type,side,price_ticks,qty,id

WHY "SYNTHETIC" ORDERS
Kraken's "book" channel is L2 (aggregated by price level), not L3
(individual orders) -- it never tells us "order #4821 was added/removed."
It only ever says "the bid level at $46,600.10 now has 1.77 total qty" (a
snapshot) or "...now has 0 qty" (i.e. remove that level) or "...now has
2.10 qty" (i.e. level's size changed) in an update. To feed that into
this project's per-order Add/Cancel engine, each PRICE LEVEL is treated
as a single synthetic resting "order": we assign it a stable synthetic
id the first time we see that price on that side, and whenever its qty
changes we emit Cancel(old_id) + Add(new_id_or_same_id, new_qty). This
loses individual order-level granularity (who's actually resting at that
price, in what time order) but is the right granularity for OFI, which
by definition operates on price-level depth changes, not individual
orders anyway (Cont, Kukanov & Stoikov's OFI is literally defined this
way) -- so this is not a compromise for OFI, only for anything that would
want true L3 (this project's Hawkes calibration only needs the trade
stream, not book granularity, so it's unaffected either way).

TIMESTAMPS
Book/depth rows use each message's local receive time (recv_ts_ns,
written by the capture script) -- reflects the actual arrival-time
process a live strategy would see for OFI purposes.

Trade rows use Kraken's OWN per-trade timestamp instead (see
kraken_trade_timestamp_to_ns and the comment at its call site for why):
recv_ts_ns is shared across every trade in a batched WS message, which
produces duplicate (dt=0) timestamps between consecutive Hawkes events
whenever Kraken delivers multiple fills together -- a real bug found via
a live ETH/USD capture, not a hypothetical one.

Usage:
    python scripts/kraken_ws_to_csv.py \
        data/kraken_ws_capture_btcusd_....jsonl \
        data/kraken_feed.csv \
        --tick-size 0.1 --qty-scale 1e6

    # to just inspect the first few raw messages instead of converting:
    python scripts/kraken_ws_to_csv.py capture.jsonl --peek 5
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def kraken_trade_timestamp_to_ns(ts_str: str) -> int:
    """Parses Kraken's per-trade ISO8601 timestamp (e.g.
    '2026-07-26T05:51:56.292057Z', microsecond precision) into integer
    nanoseconds since epoch, using exact integer arithmetic throughout
    (never a float multiply-by-1e9) specifically to avoid reintroducing a
    precision-loss bug of the kind this project has already hit once
    (see the compensator time-scaling bug writeup) while fixing this one."""
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    delta = dt - _EPOCH
    total_microseconds = delta.days * 86400 * 1_000_000 + delta.seconds * 1_000_000 + delta.microseconds
    return total_microseconds * 1000  # exact int -> nanoseconds


def convert(in_path: str, out_path: str, tick_size: float, qty_scale: float):
    # side -> { price_float: synthetic_id }
    price_to_id = {"bid": {}, "ask": {}}
    next_id = [1]  # mutable counter, boxed so the nested helper can bump it

    def new_id():
        i = next_id[0]
        next_id[0] += 1
        return i

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

            channel = msg.get("channel")
            msg_type = msg.get("type")

            if channel == "book":
                n_book_msgs += 1
                for entry in msg.get("data", []):
                    if msg_type == "snapshot":
                        # A snapshot is an authoritative full-book reset -- this
                        # happens once at the very start, and again after every
                        # reconnect during a long capture (capture_kraken_ws.py
                        # resubscribes and gets a fresh snapshot each time). Any
                        # price levels tracked from before this point may no
                        # longer be accurate (we don't know what happened during
                        # the gap), so cancel everything currently tracked
                        # before rebuilding from this snapshot.
                        for side_name in ("bid", "ask"):
                            tape_side = "buy" if side_name == "bid" else "sell"
                            for old_price, old_id in list(price_to_id[side_name].items()):
                                old_price_ticks = round(old_price / tick_size)
                                writer.writerow([recv_ts_ns, "cancel", tape_side,
                                                  old_price_ticks, 0, old_id])
                                n_cancel += 1
                            price_to_id[side_name].clear()

                    for side_key, side_name in (("bids", "bid"), ("asks", "ask")):
                        for level in entry.get(side_key, []):
                            try:
                                price = float(level["price"])
                                qty = float(level["qty"])
                            except (KeyError, TypeError, ValueError):
                                n_malformed += 1
                                continue

                            price_ticks = round(price / tick_size)
                            tape_side = "buy" if side_name == "bid" else "sell"
                            existing_id = price_to_id[side_name].get(price)

                            if qty <= 0:
                                if existing_id is not None:
                                    writer.writerow([recv_ts_ns, "cancel", tape_side,
                                                      price_ticks, 0, existing_id])
                                    n_cancel += 1
                                    del price_to_id[side_name][price]
                                # else: removal of a level we never saw added
                                # (e.g. it existed before capture started) -- nothing to cancel.
                                continue

                            qty_scaled = round(qty * qty_scale)
                            if qty_scaled <= 0:
                                continue  # dust level, not representable as an integer qty

                            if existing_id is not None:
                                writer.writerow([recv_ts_ns, "cancel", tape_side,
                                                  price_ticks, 0, existing_id])
                                n_cancel += 1
                            oid = new_id()
                            price_to_id[side_name][price] = oid
                            writer.writerow([recv_ts_ns, "add", tape_side,
                                              price_ticks, qty_scaled, oid])
                            n_add += 1

            elif channel == "trade":
                n_trade_msgs += 1
                for entry in msg.get("data", []):
                    try:
                        side = entry["side"]              # "buy" or "sell" -- already the aggressor/taker side
                        price = float(entry["price"])
                        qty = float(entry["qty"])
                        trade_id = entry.get("trade_id", new_id())
                    except (KeyError, TypeError, ValueError):
                        n_malformed += 1
                        continue

                    # Use Kraken's own per-trade timestamp, NOT recv_ts_ns, for
                    # trade rows specifically. Bug found via real data: when
                    # Kraken batches multiple trade fills into one WS message
                    # (e.g. one aggressive order sweeping several price
                    # levels), every trade in that batch shares the same
                    # recv_ts_ns (it's the time OUR process received the
                    # message, not when each individual trade occurred). That
                    # gives dt=0 between "consecutive" Hawkes events, so the
                    # kernel's exp(-beta*dt) term equals 1 (no decay at all)
                    # -- spuriously injecting "infinite instantaneous
                    # clustering" into the likelihood. This was invisible
                    # until a real ETH/USD capture showed 27% of its trades
                    # sharing an exact duplicate timestamp with another trade
                    # (one timestamp had 31 trades stacked on it), and the
                    # resulting fit converged to a suspicious, exactly
                    # identical degenerate branching ratio (0.3698) on two
                    # entirely different instruments (ETH and SOL) -- the
                    # tell that something mechanical, not real market
                    # structure, was driving the result. Kraken's own
                    # "timestamp" field has genuine microsecond-level
                    # per-trade granularity; we parse it with exact integer
                    # arithmetic (no float multiplication) to avoid
                    # reintroducing a precision-loss bug of our own while
                    # fixing this one.
                    ts_field = entry.get("timestamp")
                    if ts_field:
                        try:
                            ts_ns = kraken_trade_timestamp_to_ns(ts_field)
                        except (ValueError, TypeError):
                            ts_ns = recv_ts_ns  # malformed timestamp string -- fall back rather than crash
                    else:
                        ts_ns = recv_ts_ns  # older captures / missing field -- fall back, not silently wrong

                    price_ticks = round(price / tick_size)
                    qty_scaled = round(qty * qty_scale)
                    if qty_scaled <= 0:
                        continue
                    writer.writerow([ts_ns, "trade", side, price_ticks, qty_scaled, trade_id])
                    n_trade += 1

            else:
                n_skipped += 1

    print(f"parsed {n_book_msgs} book messages, {n_trade_msgs} trade messages "
          f"({n_skipped} other/status messages, {n_malformed} malformed entries skipped)")
    print(f"wrote add={n_add} cancel={n_cancel} trade={n_trade} to {out_path}")
    if n_add == 0:
        print("WARNING: 0 add rows -- check the capture file actually has book "
              "snapshot/update messages (run with --peek to inspect raw contents).")


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
    ap.add_argument("--tick-size", type=float, default=0.1,
                     help="dollar price per tick (default 0.1, reasonable for BTC/USD)")
    ap.add_argument("--qty-scale", type=float, default=1e6,
                     help="multiply raw qty by this before rounding to an integer "
                          "unit (engine's Quantity is int64) -- default 1e6 turns "
                          "BTC into micro-BTC so sub-1-BTC size doesn't round to 0")
    ap.add_argument("--peek", type=int, default=None, metavar="N",
                     help="instead of converting, just print the first N raw "
                          "messages (for inspecting the capture's actual format)")
    args = ap.parse_args()

    if args.peek is not None:
        peek(args.input_jsonl, args.peek)
        sys.exit(0)

    if args.output_csv is None:
        sys.exit("output_csv is required unless --peek is given")
    convert(args.input_jsonl, args.output_csv, args.tick_size, args.qty_scale)
