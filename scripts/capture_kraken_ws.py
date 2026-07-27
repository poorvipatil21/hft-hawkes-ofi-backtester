#!/usr/bin/env python3
"""
Captures real L2 depth updates + trades from Kraken's free, public
WebSocket API v2 -- no account, no API key, no signup, no geo-blocking.

This is the third exchange this project has tried for free live L2 data:
  - Binance's live WS/API blocks US IPs (HTTP 451) -- its *historical
    static-file archive* (data.binance.vision) is a different, non-blocked
    service, which is why the earlier aggTrades download worked fine.
  - Coinbase's "level2" channel now requires authentication (a recent
    policy change); only its "matches"/trades channel is still public.
  - Kraken's "book" (L2) channel is confirmed still public/unauthenticated
    as of this writing -- only their "level3" (raw per-order) channel
    needs a session token. This script uses Kraken's "book" + "trade"
    channels.

WHAT THIS CAPTURES
    Subscribes to Kraken WebSocket API v2's "book" (L2, aggregated by
    price level) and "trade" channels over a single connection. The book
    channel sends one "snapshot" message per symbol, then "update"
    messages with changed levels (qty=0 means the level was removed).
    Every message received is written as one JSON line to a .jsonl file,
    tagged with local receive time.

This is a TRIAL/first-pass capture: no checksum/gap-detection logic
beyond what Kraken's feed guarantees itself (v2's book channel includes a
running checksum per update for exactly this purpose -- worth validating
before trusting a long capture's OFI numbers, not implemented here yet).

USAGE
    pip install websockets --break-system-packages
    python3 scripts/capture_kraken_ws.py

    Writes to: data/kraken_ws_capture_<symbol>_<UTC-timestamp>.jsonl

CONVERTING THE CAPTURE
    Not done yet -- this script only records. Once you've got a capture
    file, the next step is a converter (like lobster_to_csv.py) that
    turns these raw snapshot/update messages into the project's tape
    schema (ts,type,side,price_ticks,qty,id), representing each price
    level as a synthetic resting "order" whose quantity gets replaced
    (cancel-old + add-new) on every update -- ask for that once you've
    got a capture you're happy with.
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    raise SystemExit(
        "Missing dependency. Run:\n"
        "    pip install websockets --break-system-packages"
    )

# ----------------------------------------------------------------------
# CONFIG -- these are defaults, all overridable via CLI args (see below),
# which is what capture_loop.py uses to run many independent windows
# without editing this file each time. Running this script directly with
# no args still works exactly as before (3-minute trial capture).
# ----------------------------------------------------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--symbol", default="BTC/USD", help='Kraken v2 symbol, e.g. "BTC/USD"')
_ap.add_argument("--depth", type=int, default=25, choices=[10, 25, 100, 500, 1000])
_ap.add_argument("--duration", type=float, default=180,
                  help="capture duration in seconds (default 180 = 3 min trial)")
_ap.add_argument("--out-dir", default="data", help="output directory (default: data)")
_args, _unknown = _ap.parse_known_args()

SYMBOL = _args.symbol
DEPTH = _args.depth
DURATION_SECONDS = _args.duration
OUT_DIR = _args.out_dir

# ----------------------------------------------------------------------

WS_URL = "wss://ws.kraken.com/v2"


async def capture(out_path: str, duration_s: int):
    n_book_snapshot = 0
    n_book_update = 0
    n_trade = 0
    t_start = time.time()

    subscribe_book = json.dumps({
        "method": "subscribe",
        "params": {"channel": "book", "symbol": [SYMBOL], "depth": DEPTH},
    })
    subscribe_trade = json.dumps({
        "method": "subscribe",
        "params": {"channel": "trade", "symbol": [SYMBOL]},
    })

    reconnect_delay = 2.0   # seconds, doubles on repeated failures, capped below
    n_reconnects = 0

    with open(out_path, "w") as f:
        while time.time() - t_start < duration_s:
            try:
                print(f"Connecting to {WS_URL} ..."
                      + (f" (reconnect #{n_reconnects})" if n_reconnects else ""))
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(subscribe_book)
                    await ws.send(subscribe_trade)
                    reconnect_delay = 2.0  # reset backoff after a successful connect
                    print(f"Subscribed to book(depth={DEPTH}) + trade for {SYMBOL}. "
                          f"Capturing for {duration_s}s ({duration_s/60:.1f} min) -> {out_path}")

                    while time.time() - t_start < duration_s:
                        remaining = duration_s - (time.time() - t_start)
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
                        except asyncio.TimeoutError:
                            continue  # just re-check the outer duration loop

                        recv_ts_ns = time.time_ns()
                        parsed = json.loads(msg)
                        channel = parsed.get("channel", "")
                        msg_type = parsed.get("type", "")

                        if channel == "book" and msg_type == "snapshot":
                            n_book_snapshot += 1
                        elif channel == "book" and msg_type == "update":
                            n_book_update += 1
                        elif channel == "trade":
                            n_trade += 1
                        elif "error" in parsed or parsed.get("success") is False:
                            print(f"  [error/rejected] {parsed}")

                        f.write(json.dumps({
                            "recv_ts_ns": recv_ts_ns,
                            "data": parsed,
                        }) + "\n")
                        f.flush()   # survive crashes/kills without losing the last few messages

                        if (n_book_update + n_trade) % 500 == 0 and (n_book_update + n_trade) > 0:
                            elapsed = time.time() - t_start
                            print(f"  ... {elapsed:6.1f}s elapsed, "
                                  f"book_updates={n_book_update}, trades={n_trade}")

            except (websockets.exceptions.ConnectionClosed, ConnectionResetError, OSError) as e:
                n_reconnects += 1
                remaining = duration_s - (time.time() - t_start)
                if remaining <= 0:
                    break
                print(f"Connection dropped ({type(e).__name__}: {e}). "
                      f"Reconnecting in {reconnect_delay:.0f}s "
                      f"({remaining:.0f}s left in capture)...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)  # exponential backoff, capped at 60s
                # NOTE: a fresh connection re-subscribes and gets a brand-new book
                # snapshot (n_book_snapshot will be >1 across the whole capture) --
                # that's expected and fine, the converter treats every snapshot as
                # a full book reset, same as the very first one.

    elapsed = time.time() - t_start
    print(f"\nDone. Captured {elapsed:.1f}s of live data to {out_path} "
          f"({n_reconnects} reconnect(s) along the way)")
    print(f"  book snapshots : {n_book_snapshot}")
    print(f"  book updates   : {n_book_update}")
    print(f"  trades         : {n_trade}")
    if n_book_snapshot == 0:
        print("WARNING: no book snapshot received -- check for an "
              "[error/rejected] line printed above (e.g. bad symbol).")


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_symbol = SYMBOL.replace("/", "").lower()
    out_path = f"{OUT_DIR}/kraken_ws_capture_{safe_symbol}_{ts_tag}.jsonl"
    print(f"Duration: {DURATION_SECONDS}s ({DURATION_SECONDS/3600:.2f} hours), symbol={SYMBOL}, depth={DEPTH}")
    asyncio.run(capture(out_path, DURATION_SECONDS))
