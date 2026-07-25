// Fits a bivariate (buy-MO, sell-MO) Hawkes process to aggressor arrivals in
// a market-data tape, reports MLE diagnostics (log-lik, branching ratio,
// spectral radius, residual GOF), and checks the correlation between OFI and
// forward mid-price returns -- the two empirical pillars of the ICAIF signal
// section.
//
// EVERY RUN IS TIMESTAMPED, NOTHING GETS OVERWRITTEN:
//   logs/calibrate_hawkes_<UTC-timestamp>.log        -- full console output
//   hawkes_calibration_<UTC-timestamp>.csv           -- dated calibration params
//   hawkes_calibration.csv                           -- ALSO updated (unchanged
//                                                        name) so run_backtest's
//                                                        default loader keeps
//                                                        working without extra steps
//
// Usage: ./bin/calibrate_hawkes [n_events] [seed] [beta] [csv_path] [max_iters] [lr] [mu_reg]
//   csv_path  : optional, load a real tape instead of the synthetic generator.
//   beta      : <=0 auto-scales from data (1/mean inter-arrival). This is the
//               *middle* of 3 fixed decay timescales (beta/10, beta, beta*10)
//               -- a sum-of-exponentials kernel, since real order-flow
//               clustering isn't well captured by one timescale.
//   max_iters : Adam iterations (default 2000). Check `converged` in the
//               output -- if 0, rerun with more iterations before trusting
//               the fitted alpha/mu.
//   lr        : Adam learning rate in log-space (default 0.05).
//   mu_reg    : ridge penalty strength anchoring mu toward its count/T
//               estimate (default 0.5, applied as decoupled post-step
//               shrinkage -- see hawkes.hpp). Set to 0 to disable.

#include "backtest/data_feed.hpp"
#include "backtest/order_book.hpp"
#include "backtest/matching_engine.hpp"
#include "backtest/hawkes.hpp"
#include "backtest/ofi.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <cmath>
#include <fstream>
#include <ctime>
#include <sys/stat.h>

using namespace bt;

// Returns a UTC timestamp string like "20260125T153045Z", safe for filenames.
std::string utc_timestamp() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc;
    gmtime_r(&now, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm_utc);
    return std::string(buf);
}

// Prints to stdout AND appends to the run's log file, so every invocation's
// full output is preserved under a unique, timestamped filename instead of
// only living in a terminal scrollback that gets lost between runs -- this
// matters once you're running many calibrations (e.g. one per Kraken
// capture window) and want to compare/aggregate them later.
static FILE* g_logfile = nullptr;
void LOG(const char* fmt, ...) {
    va_list args1, args2;
    va_start(args1, fmt);
    va_copy(args2, args1);
    std::vprintf(fmt, args1);
    if (g_logfile) std::vfprintf(g_logfile, fmt, args2);
    va_end(args1);
    va_end(args2);
}

// Replays a MarketEvent onto the book exactly as Backtester::apply_market_event
// does internally (that method is private to Backtester, which also owns
// strategy/latency/portfolio machinery we don't need here for pure signal
// calibration, so we replicate the tape->book mapping directly).
void apply_market_event(OrderBook& /*book*/, MatchingEngine& engine, const MarketEvent& me) {
    switch (me.type) {
        case MdType::Add: {
            Order o;
            o.id = me.id; o.side = me.side; o.type = OrderType::Limit;
            o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
            o.is_ours = false; o.ts_active = me.ts;
            engine.submit(o, [](const Fill&) {});
            break;
        }
        case MdType::Cancel:
            engine.cancel(me.id);
            break;
        case MdType::Trade: {
            Order o;
            o.id = me.id; o.side = me.side; o.type = OrderType::IOC;
            o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
            o.is_ours = false; o.ts_active = me.ts;
            engine.submit(o, [](const Fill&) {});
            break;
        }
    }
}

int main(int argc, char** argv) {
    std::size_t n      = argc > 1 ? std::stoull(argv[1]) : 300000;
    std::uint64_t seed = argc > 2 ? std::stoull(argv[2]) : 7;
    double beta_arg    = argc > 3 ? std::stod(argv[3]) : -1.0;  // <=0 => auto-scale from data
    std::string csv    = argc > 4 ? argv[4] : "";
    int max_iters      = argc > 5 ? std::atoi(argv[5]) : 2000;
    double lr          = argc > 6 ? std::stod(argv[6]) : 0.05;
    double mu_reg      = argc > 7 ? std::stod(argv[7]) : 0.5;

    std::string ts = utc_timestamp();
    mkdir("logs", 0755);  // no-op if it already exists
    std::string log_path = "logs/calibrate_hawkes_" + ts + ".log";
    g_logfile = std::fopen(log_path.c_str(), "w");

    LOG("=== calibrate_hawkes run %s ===\n", ts.c_str());
    LOG("args: n=%zu seed=%llu beta=%.6g csv=%s max_iters=%d lr=%.6g mu_reg=%.6g\n\n",
        n, (unsigned long long)seed, beta_arg, csv.empty() ? "(synthetic)" : csv.c_str(),
        max_iters, lr, mu_reg);

    std::vector<MarketEvent> feed = csv.empty() ? generate_synthetic_feed(n, seed)
                                                 : load_csv_feed(csv);
    if (feed.empty()) { std::fprintf(stderr, "empty feed\n"); return 1; }

    // Replay the tape through the real book so OFI is computed off actual
    // depth, and collect Trade events as the Hawkes mark process.
    OrderBook book;
    MatchingEngine engine(book);
    RollingOFI ofi(/*depth=*/5, /*window=*/500);

    std::vector<HawkesEvent> hawkes_events;   // mark 0 = buy aggressor, 1 = sell aggressor
    hawkes_events.reserve(n / 4);

    std::vector<double> ofi_signal;   // sampled OFI at each Trade event, pre-trade
    std::vector<double> fwd_return;   // mid-price return over the next `horizon` events
    struct Sample { double ofi; std::size_t idx; };
    std::vector<Sample> pending;
    std::vector<double> mids;
    mids.reserve(n);

    const int horizon = 20;

    for (const auto& ev : feed) {
        Price bb = 0, ba = 0;
        book.best_bid(bb); book.best_ask(ba);
        double mid = (bb > 0 && ba > 0) ? (static_cast<double>(bb) + ba) / 2.0
                                        : (mids.empty() ? 0.0 : mids.back());
        mids.push_back(mid);

        double ofi_inc = ofi.update(book);
        (void)ofi_inc;

        if (ev.type == MdType::Trade) {
            hawkes_events.push_back(HawkesEvent{static_cast<double>(ev.ts),
                                                 ev.side == Side::Buy ? std::size_t(0) : std::size_t(1)});
            pending.push_back(Sample{ofi.rolling_sum(), mids.size() - 1});
        }

        // Apply the event to the book/engine so state reflects reality for
        // the *next* iteration's OFI/mid computation.
        apply_market_event(book, engine, ev);
    }

    // Resolve forward returns for OFI samples now that we have the full mid series.
    for (const auto& s : pending) {
        std::size_t j = std::min(s.idx + horizon, mids.size() - 1);
        if (mids[s.idx] > 0.0 && mids[j] > 0.0) {
            ofi_signal.push_back(s.ofi);
            fwd_return.push_back(mids[j] - mids[s.idx]);
        }
    }

    double corr = 0.0;
    {
        std::size_t m = ofi_signal.size();
        if (m > 2) {
            double mo = 0, mr = 0;
            for (std::size_t k = 0; k < m; ++k) { mo += ofi_signal[k]; mr += fwd_return[k]; }
            mo /= m; mr /= m;
            double cov = 0, vo = 0, vr = 0;
            for (std::size_t k = 0; k < m; ++k) {
                double a = ofi_signal[k] - mo, b = fwd_return[k] - mr;
                cov += a * b; vo += a * a; vr += b * b;
            }
            corr = (vo > 1e-9 && vr > 1e-9) ? cov / std::sqrt(vo * vr) : 0.0;
        }
    }

    LOG("== Tape ==\n");
    LOG("events            : %zu (%zu aggressor/Hawkes marks)\n", feed.size(), hawkes_events.size());
    LOG("OFI vs fwd-return (h=%d) corr : %.4f  (n=%zu)\n\n", horizon, corr, ofi_signal.size());

    if (hawkes_events.size() < 50) {
        std::fprintf(stderr, "too few aggressor events to fit Hawkes\n");
        LOG("too few aggressor events to fit Hawkes\n");
        if (g_logfile) std::fclose(g_logfile);
        return 1;
    }

    // CRITICAL: for real data, HawkesEvent.t is the raw tape timestamp, which
    // for live-captured data is a Unix epoch nanosecond value (~1.78e18,
    // i.e. ~56 years) -- NOT elapsed time within the observation window
    // (~1e12-1e13 ns for an hours-long capture). Using it directly as T
    // inflates the compensator's mu_i*T term by a factor of ~10^5-10^6,
    // which was the actual root cause of mu collapsing to physically
    // meaningless near-identical values across marks in every real-data run
    // so far -- something this project spent real effort chasing as a
    // "mu/slow-kernel identifiability" issue before finding this. Shifting
    // all event times so the first observed event is at t=0 fixes this:
    // T then correctly represents the elapsed observation-window duration.
    if (!hawkes_events.empty()) {
        double t0 = hawkes_events.front().t;
        for (auto& e : hawkes_events) e.t -= t0;
    }

    double T = hawkes_events.back().t;

    double beta = beta_arg;
    if (beta <= 0.0) {
        // Auto-scale decay from the data: mean inter-arrival of the mark
        // process sets the natural timescale, so beta = 1/mean_dt gives a
        // half-life on the order of one typical inter-arrival -- a sane,
        // data-driven default rather than a hardcoded guess tied to one
        // tape's timestamp units.
        double sum_dt = 0.0;
        for (std::size_t k = 1; k < hawkes_events.size(); ++k)
            sum_dt += hawkes_events[k].t - hawkes_events[k - 1].t;
        double mean_dt = hawkes_events.size() > 1 ? sum_dt / (hawkes_events.size() - 1) : 1.0;
        beta = mean_dt > 1e-12 ? 1.0 / mean_dt : 1.0;
    }

    // Sum-of-exponentials kernel: 3 fixed timescales (fast/medium/slow),
    // one decade apart, shared across all (i,j) pairs. This is the fix for
    // the GOF-variance blowup a single exponential shows on real data --
    // real clustering isn't one clean timescale.
    std::vector<double> betas = {beta * 10.0, beta, beta / 10.0};
    std::vector<std::vector<std::vector<double>>> beta_mat(
        2, std::vector<std::vector<double>>(2, betas));
    MultivariateHawkes hp(2, beta_mat);
    auto fit = hp.fit_mle(hawkes_events, T, max_iters, lr, 1e-7, mu_reg);

    LOG("== Hawkes MLE (dims: 0=buy-MO, 1=sell-MO; betas=[%.4g, %.4g, %.4g] fixed) ==\n",
                betas[0], betas[1], betas[2]);
    LOG("iterations        : %d / %d requested (converged=%d)%s\n",
                fit.iterations, max_iters, fit.converged,
                fit.converged ? "" : "  <-- rerun with larger max_iters (arg 5) before trusting alpha/mu");
    LOG("log-likelihood    : %.3f\n", fit.final_ll);
    LOG("spectral radius   : %.4f  (<1 required for stationarity)\n", fit.spectral_radius);
    LOG("mu                : [%.6e, %.6e]\n", hp.mu()[0], hp.mu()[1]);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            LOG("alpha[%zu][%zu] (fast,med,slow) : [%.6e, %.6e, %.6e]\n",
                        i, j, hp.alpha()[i][j][0], hp.alpha()[i][j][1], hp.alpha()[i][j][2]);
        }
    }
    auto n_mat = hp.branching_ratio();
    LOG("branching ratio n : [[%.4f, %.4f], [%.4f, %.4f]]\n",
                 n_mat[0][0], n_mat[0][1], n_mat[1][0], n_mat[1][1]);

    // Persist calibration params two ways:
    //   1. a timestamped, never-overwritten copy (history for Tier-1 multi-capture work)
    //   2. the fixed "hawkes_calibration.csv" name (so run_backtest's default loader
    //      keeps working without needing a filename argument every time)
    auto write_calib = [&](const std::string& path) {
        std::ofstream calib(path);
        calib << "beta0,beta1,beta2,mu0,mu1";
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                for (std::size_t p = 0; p < 3; ++p)
                    calib << ",a" << i << j << p;
        calib << "\n";
        calib << betas[0] << ',' << betas[1] << ',' << betas[2] << ','
              << hp.mu()[0] << ',' << hp.mu()[1];
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j)
                for (std::size_t p = 0; p < 3; ++p)
                    calib << ',' << hp.alpha()[i][j][p];
        calib << "\n";
    };
    std::string dated_calib_path = "hawkes_calibration_" + ts + ".csv";
    write_calib(dated_calib_path);
    write_calib("hawkes_calibration.csv");
    LOG("\nCalibration written to %s (dated) and hawkes_calibration.csv (for run_backtest)\n",
        dated_calib_path.c_str());

    // GOF: rescaled residuals should be ~Exp(1), i.e. mean~1, var~1.
    auto resid = hp.compensator_residuals(hawkes_events);
    for (std::size_t i = 0; i < resid.size(); ++i) {
        const auto& r = resid[i];
        if (r.size() < 5) continue;
        double m = 0; for (double x : r) m += x; m /= r.size();
        double v = 0; for (double x : r) v += (x - m) * (x - m); v /= (r.size() - 1);
        LOG("residual GOF mark %zu : n=%zu mean=%.4f var=%.4f  (target: 1.0, 1.0)\n",
                     i, r.size(), m, v);
    }

    LOG("\nFull log saved to %s\n", log_path.c_str());
    if (g_logfile) std::fclose(g_logfile);
    return 0;
}
