// run_backtest — end-to-end backtest of a strategy on an L2 tape.
//
//   ./run_backtest [events] [strategy] [out_csv] [tape_csv] [tick_size] [qty_scale]
//     events    : number of events for the synthetic generator (default
//                 500000). Ignored if tape_csv is given.
//     strategy  : "mm" (market maker, default), "mr" (mean reversion), or
//                 "hawkes" (OFI + Hawkes-intensity-skewed MM; reads
//                 hawkes_calibration.csv written by calibrate_hawkes if
//                 present, else falls back to a data-scaled beta with
//                 near-zero mu/alpha and a warning)
//     out_csv   : equity-curve output path (default equity_curve.csv)
//     tape_csv  : optional real tape (schema: ts,type,side,price_ticks,qty,id).
//                 If omitted, a synthetic tape is generated instead.
//     tick_size : dollars per price_ticks unit (default 0.01). MUST match
//                 whatever --tick-size the converter used to build tape_csv,
//                 or every dollar figure (PnL, equity, drawdown) is wrong by
//                 that ratio -- e.g. converter used 0.1 but this defaults to
//                 0.01 gives a silent 10x price error.
//     qty_scale : how many integer Quantity units equal one real share/coin
//                 (default 1.0, meaning Quantity units ARE whole shares).
//                 MUST match whatever --qty-scale the converter used (e.g.
//                 1e6 for "1 unit == 1 micro-BTC"), or every dollar figure
//                 is wrong by that ratio too -- this one is the more
//                 dangerous silent error since it's a much larger factor
//                 (1e6 vs a 10x tick_size mismatch) and won't look
//                 obviously wrong until churn is high enough to amplify it
//                 into an impossible number (which is exactly what happened
//                 during this project's own real-data testing: identical
//                 bug, invisible at 22 fills, glaringly a -350% return at
//                 400k fills).
//
// NOTE on trades-only real data (e.g. crypto_trades_to_csv.py output): a
// tape with only "trade" rows and no "add"/"cancel" rows never populates the
// order book, so best_bid/best_ask never exist and MM/hawkes strategies
// will simply never quote (has_bid/has_ask stay false all run) -- this
// isn't a bug, it's a fundamental requirement that resting-quote strategies
// need real depth data (LOBSTER, a Binance depth-snapshot archive, etc.),
// not just a trade tape. This app detects and warns about that case below.
#include "backtest/backtester.hpp"
#include "backtest/analytics.hpp"
#include "market_maker.hpp"
#include "mean_reversion.hpp"
#include "hawkes_ofi_mm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

using namespace bt;

struct HawkesCalibration {
    std::vector<double> betas = {1e-3, 1e-4, 1e-5};   // fast, med, slow
    std::vector<double> mu    = {1e-4, 1e-4};
    // alpha[i][j][p]
    std::vector<std::vector<std::vector<double>>> alpha =
        std::vector<std::vector<std::vector<double>>>(2, std::vector<std::vector<double>>(2, std::vector<double>(3, 1e-6)));
};

// Loads calibrate_hawkes's output CSV (schema: beta0,beta1,beta2,mu0,mu1,
// a000..a112) if it exists. Returns false (leaving defaults untouched) if
// the file isn't there or doesn't parse as expected.
bool load_hawkes_calibration(HawkesCalibration& c) {
    std::ifstream in("hawkes_calibration.csv");
    if (!in) return false;
    std::string header, line;
    std::getline(in, header);
    if (!std::getline(in, line)) return false;
    std::stringstream ss(line);
    std::string tok;
    auto next = [&]() { std::getline(ss, tok, ','); return std::stod(tok); };
    c.betas = {next(), next(), next()};
    c.mu    = {next(), next()};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t p = 0; p < 3; ++p)
                c.alpha[i][j][p] = next();
    return true;
}

int main(int argc, char** argv) {
    std::size_t n_events = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 500'000;
    std::string strat    = (argc > 2) ? argv[2] : "mm";
    std::string out_csv  = (argc > 3) ? argv[3] : "equity_curve.csv";
    std::string tape_csv = (argc > 4) ? argv[4] : "";
    double tick_size     = (argc > 5) ? std::stod(argv[5]) : 0.01;
    double qty_scale     = (argc > 6) ? std::stod(argv[6]) : 1.0;

    std::vector<MarketEvent> feed;
    if (!tape_csv.empty()) {
        std::printf("Loading real tape: %s...\n", tape_csv.c_str());
        feed = load_csv_feed(tape_csv);
        if (feed.empty()) {
            std::fprintf(stderr, "ERROR: %s loaded 0 events (missing file, or wrong schema).\n",
                         tape_csv.c_str());
            return 1;
        }
        std::size_t n_add = 0, n_cancel = 0, n_trade = 0;
        for (const auto& e : feed) {
            if (e.type == MdType::Add) ++n_add;
            else if (e.type == MdType::Cancel) ++n_cancel;
            else ++n_trade;
        }
        std::printf("Loaded %zu events (add=%zu, cancel=%zu, trade=%zu)\n",
                    feed.size(), n_add, n_cancel, n_trade);
        if (n_add == 0) {
            std::fprintf(stderr,
                "WARNING: 0 Add events in this tape -- the order book will "
                "never have depth, so best_bid/best_ask never exist and "
                "MM/hawkes strategies will NEVER quote (flat equity curve is "
                "expected, not a bug). This is typical of trades-only data "
                "(e.g. crypto_trades_to_csv.py output). You need a real L2 "
                "depth dataset (LOBSTER, Binance depth snapshots) to "
                "backtest a resting-quote strategy on real data.\n");
        }
    } else {
        std::printf("Generating synthetic L2 tape: %zu events...\n", n_events);
        feed = generate_synthetic_feed(n_events);
    }

    Backtester::Config cfg;
    cfg.enable_friction = true;
    cfg.tick_size = tick_size;
    cfg.qty_scale = qty_scale;
    if (!tape_csv.empty()) {
        std::printf("Using tick_size=%.6g, qty_scale=%.6g for dollar accounting "
                    "-- MUST match whatever the converter used, or PnL/equity will be wrong.\n",
                    tick_size, qty_scale);
    }
    Backtester bt(cfg);

    std::unique_ptr<Strategy> strategy;
    if (strat == "mr") {
        strategy = std::make_unique<MeanReversion>(50, 2.0, 100);
    } else if (strat == "hawkes") {
        HawkesCalibration c;
        if (load_hawkes_calibration(c)) {
            std::printf("Loaded hawkes_calibration.csv (betas=[%.4g, %.4g, %.4g])\n",
                        c.betas[0], c.betas[1], c.betas[2]);
        } else {
            std::fprintf(stderr,
                "WARNING: hawkes_calibration.csv not found -- run "
                "./bin/calibrate_hawkes first for real params. Using "
                "uninformed near-zero defaults (signal will be inert).\n");
        }
        HawkesOFIMarketMaker::Params p;
        p.quote_size = 50; p.base_half_spread = 1; p.max_inventory = 500;
        std::vector<std::vector<std::vector<double>>> beta_mat(
            2, std::vector<std::vector<double>>(2, c.betas));
        auto mm = std::make_unique<HawkesOFIMarketMaker>(p, beta_mat);
        mm->hawkes_process().mu()    = c.mu;
        mm->hawkes_process().alpha() = c.alpha;
        strategy = std::move(mm);
    } else {
        strategy = std::make_unique<MarketMaker>(50, 1, 500);
    }
    bt.set_strategy(strategy.get());

    std::printf("Running '%s' strategy with friction enabled...\n", strategy->name().c_str());
    bt.run(feed);

    BacktestReport rep = summarize(bt.portfolio(), bt.processed());

    std::printf("\n================ BACKTEST REPORT ================\n");
    std::printf("  Strategy          : %s\n",       strategy->name().c_str());
    std::printf("  Events processed  : %zu\n",      rep.events);
    std::printf("  Strategy fills    : %zu\n",      rep.fills);
    std::printf("  End position      : %lld\n",     (long long)rep.end_position);
    std::printf("  Final equity      : $%.2f\n",    rep.final_equity);
    std::printf("  Total PnL         : $%.2f\n",    rep.total_pnl);
    std::printf("  Return            : %.4f%%\n",   rep.return_pct);
    std::printf("  Sharpe (per-step) : %.3f\n",     rep.sharpe);
    std::printf("  Max drawdown      : %.4f%%\n",   rep.max_drawdown * 100.0);
    std::printf("=================================================\n");

    std::ofstream out(out_csv);
    out << "ts,equity\n";
    for (const auto& p : bt.portfolio().curve()) out << p.ts << ',' << p.equity << '\n';
    std::printf("Equity curve written to %s (%zu points)\n",
                out_csv.c_str(), bt.portfolio().curve().size());
    return 0;
}
