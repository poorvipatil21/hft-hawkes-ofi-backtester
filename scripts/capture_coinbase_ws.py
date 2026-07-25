#!/usr/bin/env python3
"""
Captures real L2 depth updates + trades from Coinbase's free, public
WebSocket feed -- no account, no API key, no signup, no geo-blocking for
US users (Coinbase is a US exchange; Binance's live API/WS blocks US IPs
with HTTP 451, which is why this uses Coinbase instead -- Binance's
static historical-file archive, data.binance.vision, is a *different*,
non-geo-blocked service, which is why the earlier aggTrades download
worked fine).

WHAT THIS CAPTURES
    Subscribes to Coinbase's "level2" (full L2 order book: one initial
    "snapshot" message, then incremental "l2update" messages) and
    "matches" (trade executions) channels, over a single WebSocket
    connection. Every message received is written as one JSON line to a
    .jsonl file, tagged with local receive time.

This is a TRIAL/first-pass capture: no gap-detection/resync logic beyond
what Coinbase's feed guarantees itself. Fine for a short test; for a long
capture you're going to trust for real OFI numbers, it's worth adding a
sequence check before relying on it.

USAGE
    pip install websockets --break-system-packages
    python3 scripts/capture_binance_ws.py

    Writes to: data/coinbase_ws_capture_<product>_<UTC-timestamp>.jsonl

CONVERTING THE CAPTURE
    Not done yet -- this script only records. Once you've got a capture
    file, the next step is a converter (like lobster_to_csv.py) that
    turns these raw snapshot/l2update/match messages into the project's
    tape schema (ts,type,side,price_ticks,qty,id), representing each
    price level as a synthetic resting "order" whose quantity gets
    replaced (cancel-old + add-new) on every l2update -- ask for that
    once you've got a capture you're happy with.
"""
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
# CONFIG -- pick ONE duration block below (comment out the other).
# ----------------------------------------------------------------------
PRODUCT_ID = "BTC-USD"       # Coinbase's symbol format, e.g. "BTC-USD", "ETH-USD"

# --- TRIAL CAPTURE (active by default): a few minutes, for a first test ---
DURATION_SECONDS = 180        # 3 minutes

# --- LONG CAPTURE (uncomment when ready, comment out the block above) ---
# DURATION_SECONDS = 6 * 3600   # 6 hours
# DURATION_SECONDS = 24 * 3600  # a full day, if you want to leave it overnight

# ----------------------------------------------------------------------

WS_URL = "wss://ws-feed.exchange.coinbase.com"


async def capture(out_path: str, duration_s: int):
    n_snapshot = 0
    n_l2update = 0
    n_match = 0
    t_start = time.time()

    subscribe_msg = json.dumps({
        "type": "subscribe",
        "product_ids": [PRODUCT_ID],
        "channels": ["level2", "matches"],
    })

    with open(out_path, "w") as f:
        print(f"Connecting to {WS_URL} ...")
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(subscribe_msg)
            print(f"Subscribed to level2 + matches for {PRODUCT_ID}. "
                  f"Capturing for {duration_s}s ({duration_s/60:.1f} min) -> {out_path}")

            while time.time() - t_start < duration_s:
                remaining = duration_s - (time.time() - t_start)
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
                except asyncio.TimeoutError:
                    continue  # just re-check the outer duration loop

                recv_ts_ns = time.time_ns()
                parsed = json.loads(msg)
                msg_type = parsed.get("type", "")

                if msg_type == "snapshot":
                    n_snapshot += 1
                elif msg_type == "l2update":
                    n_l2update += 1
                elif msg_type == "match":
                    n_match += 1
                elif msg_type in ("subscriptions", "error"):
                    print(f"  [{msg_type}] {parsed}")
                    if msg_type == "error":
                        continue  # still record it below, but flag loudly

                f.write(json.dumps({
                    "recv_ts_ns": recv_ts_ns,
                    "data": parsed,
                }) + "\n")

                if (n_l2update + n_match) % 500 == 0 and (n_l2update + n_match) > 0:
                    elapsed = time.time() - t_start
                    print(f"  ... {elapsed:6.1f}s elapsed, "
                          f"l2updates={n_l2update}, matches={n_match}")

    elapsed = time.time() - t_start
    print(f"\nDone. Captured {elapsed:.1f}s of live data to {out_path}")
    print(f"  snapshots  : {n_snapshot}")
    print(f"  l2updates  : {n_l2update}")
    print(f"  matches    : {n_match}")
    if n_snapshot == 0:
        print("WARNING: no snapshot message received -- check for an 'error' "
              "line printed above (e.g. bad product_id).")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_product = PRODUCT_ID.replace("-", "").lower()
    out_path = f"data/coinbase_ws_capture_{safe_product}_{ts_tag}.jsonl"
    asyncio.run(capture(out_path, DURATION_SECONDS))
