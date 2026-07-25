# High-Performance Event-Driven Backtesting Engine

A modular, header-only C++17 backtesting engine for limit-order-book strategies.
It ingests asynchronous quote/trade events, simulates full LOB dynamics and order
state transitions through a real price-time-priority matching engine, models
execution friction (latency + volume-weighted slippage), and tracks mark-to-market
PnL. Built for CS 617 (D.K. Han, Drexel).

**Measured on this machine** (g++ 13, `-O2`, single thread):

| Path | Throughput |
|------|-----------|
| Matching-engine core | **16.8 M events/sec** |
| Full engine + market-making strategy | **10.5 M events/sec** |

Friction-model fidelity (see [methodology](#fidelity-methodology)): a naive
frictionless backtest diverges from the ground-truth "live" execution by **~440%**
of net PnL; the calibrated latency + slippage model tracks it to **<1.5% across
seeds** (typ. ~0.7%), i.e. under the 2% target.

> Numbers are reproducible with the commands below; your hardware will differ.

---

## Architecture

Everything under `include/backtest/` is header-only; drop it on your include path.

```
data_feed ─► Backtester ─┬─► MatchingEngine ─► OrderBook (price-time priority)
                         │        ▲                 (std::map levels, list FIFO,
   latency-scheduled     │        │                  O(1) cancel via id->locator)
   strategy orders ──────┘   Fill events
                         │
                         ├─► Friction (LatencyModel, SlippageModel)
                         ├─► Portfolio (signed cash, VWAP avg px, realized/unrealized)
                         └─► Strategy (on_market_data / on_fill callbacks)
```

Design notes:
- **Integer-tick prices** (`int64`) throughout — no float compares in the hot path.
- **Event-driven core.** Market events and latency-delayed strategy orders are
  merged through a single time-ordered priority queue, so a strategy order submitted
  at time *t* only reaches the book at *t + latency*, exactly as in production.
- **Real matching.** `OrderBook::match` walks resting liquidity FIFO at the maker
  price and emits both taker and maker fills; order types `Limit`, `Market`, `IOC`,
  `FOK` are honored with correct remainder handling.
- **Header-only.** No link step for the library; apps and tests just include it.

| Header | Responsibility |
|--------|----------------|
| `types.hpp` | tick config, `Side`/`OrderType`/`OrderStatus`, aliases |
| `order.hpp` | `Order` state machine, `Fill` |
| `order_book.hpp` | L2 book, price-time priority, O(1) cancel |
| `matching_engine.hpp` | order-type semantics over the book |
| `friction.hpp` | `LatencyModel`, `SlippageModel` |
| `portfolio.hpp` | cash/position accounting, equity curve |
| `strategy.hpp` | `Strategy` base + `OrderContext` |
| `data_feed.hpp` | synthetic L2 tape + CSV loader |
| `analytics.hpp` | Sharpe, max drawdown, tracking error |
| `backtester.hpp` | the event loop tying it together |

Strategies live in `strategies/`: a two-sided inventory-skewed `MarketMaker`
(resting/maker fills) and a z-score `MeanReversion` taker (IOC, exercises
slippage/latency).

---

## Build & run

Requires a C++17 compiler. CMake **or** the bundled Makefile.

```bash
# Makefile
make            # builds apps + tests into ./bin
make check      # runs the unit tests

# or CMake
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

Run a backtest and write its equity curve:

```bash
./bin/run_backtest 500000 mm equity_curve.csv     # 500k events, market maker
./bin/run_backtest 500000 mr equity_curve.csv     # mean-reversion taker
python scripts/plot_pnl.py equity_curve.csv        # optional, needs matplotlib
```

Throughput benchmark and fidelity validation:

```bash
./bin/benchmark 5000000
./bin/validate_fidelity 200000 7
```

Generate a replayable CSV tape (schema consumed by `load_csv_feed()`):

```bash
python scripts/generate_data.py --events 500000 --out data/feed.csv --seed 7
```

---

## Fidelity methodology

"Backtest-vs-live PnL tracking error" is a statement about **execution cost**, not
strategy alpha, so `validate_fidelity` isolates it in a controlled experiment
rather than reporting a single lucky run.

A fixed, alpha-bearing order flow (a strategy fading a mean-reverting price
deviation, so it has genuine positive edge) is executed three ways against the
**same mid**, so the gross mid-to-mid PnL is identical across runs and only the
modelled friction differs:

- **NAIVE** — fills at the decision mid; zero latency, zero slippage. The
  optimistic PnL an un-modelled backtester reports.
- **LIVE** — the ground-truth execution: fat-tailed latency (occasional spikes),
  adverse selection accruing over the latency window, and slippage that walks a
  geometric book with a non-linear term plus per-fill execution noise.
- **MODEL** — the backtest's friction model: a *linear* volume-weighted slippage
  term calibrated (regression-through-origin) to LIVE's mean cost over the realised
  size mix, plus a Monte-Carlo-calibrated latency cost. It is a deliberate
  simplification — it matches the mean, not the curvature, tails, or noise.

Calibration uses only cost *distributions* and the order-size mix, never LIVE's
realised per-fill draws on the test path, so the fit is honest rather than
circular. Because the calibrated model is unbiased in the mean, its cumulative
cost converges to realised cost (law of large numbers over ~10⁵ fills), giving a
low single-path tracking error; the residual is the linear-vs-non-linear structural
gap plus irreducible per-fill noise. NAIVE, by omitting cost entirely, carries the
full friction gap.

```
Order decisions            : 200000
Final net PnL (live)       : $20,221
NAIVE vs LIVE tracking err : 439.9%   (no friction modelled)
MODEL vs LIVE tracking err : 0.74%    (latency + slippage modelled)   -> PASS (<2%)
```

Across seeds {7, 11, 23, 42, 99, 123} MODEL stays in 0.5–1.4%. This is a synthetic
ground-truth-vs-model study: "live" is a high-fidelity execution simulation, not a
live brokerage feed. It demonstrates that modelling latency and volume-weighted
slippage removes the systematic bias a frictionless backtest suffers; on a real
deployment the same calibration would be fit to recorded execution logs.

---

## Tests

`make check` runs three dependency-free suites (custom `CHECK` macros, no gtest):

- `test_order_book` — best-price/depth, cancel + level collapse, price-time priority.
- `test_matching` — `Limit` rests remainder, `IOC` cancels remainder, `FOK`
  reject/fill, `Market` sweep.
- `test_friction` — latency bounds, slippage monotonicity + thin-book penalty,
  portfolio round-trip PnL (long and short-cover).

---

## Repository layout

```
include/backtest/   header-only engine
strategies/         example strategies
apps/               run_backtest, benchmark, validate_fidelity
tests/              unit tests
scripts/            data generation + PnL plotting
CMakeLists.txt      CMake build
Makefile            Make build
```

## License

MIT — see [LICENSE](LICENSE).
