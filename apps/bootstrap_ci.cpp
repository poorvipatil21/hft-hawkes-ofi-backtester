// Block-bootstrap confidence intervals for the two headline statistics of
// this project: the OFI-vs-forward-return correlation and the Hawkes
// branching ratio. Both are computed on autocorrelated data (OFI's rolling
// window and Hawkes's self-excitation both induce serial dependence), so a
// naive i.i.d. bootstrap would understate the true sampling uncertainty --
// we use a block bootstrap in both cases, resampling contiguous chunks
// rather than individual points/events, to (approximately) preserve local
// dependency structure under resampling.
//
// Usage: ./bin/bootstrap_ci <csv_path> [n_boot] [ofi_block] [hawkes_blocks] [hawkes_iters]
//   csv_path      : real tape (schema: ts,type,side,price_ticks,qty,id).
//   n_boot        : number of bootstrap replicates (default 1000).
//   ofi_block     : block size (in samples) for the OFI correlation
//                   bootstrap (default 50).
//   hawkes_blocks : number of time-blocks to split [0,T] into for the
//                   Hawkes branching-ratio bootstrap (default 20). Each
//                   replicate resamples block *indices* with replacement,
//                   then time-shifts each sampled block's real events into
//                   consecutive synthetic slots -- this resamples chunks of
//                   real inter-event structure rather than individual
//                   events, which would destroy the very clustering we're
//                   trying to measure the uncertainty of.
//   hawkes_iters  : Adam iterations per bootstrap replicate's Hawkes refit
//                   (default 1000 -- lower than calibrate_hawkes's default
//                   since this runs many times; convergence per-replicate
//                   is secondary to getting a percentile spread across
//                   replicates that were all fit the same, consistent way).
#include "backtest/data_feed.hpp"
#include "backtest/order_book.hpp"
#include "backtest/matching_engine.hpp"
#include "backtest/hawkes.hpp"
#include "backtest/ofi.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

using namespace bt;

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

double pearson_corr(const std::vector<double>& x, const std::vector<double>& y) {
    std::size_t m = x.size();
    if (m < 3) return 0.0;
    double mx = 0, my = 0;
    for (std::size_t k = 0; k < m; ++k) { mx += x[k]; my += y[k]; }
    mx /= m; my /= m;
    double cov = 0, vx = 0, vy = 0;
    for (std::size_t k = 0; k < m; ++k) {
        double a = x[k] - mx, b = y[k] - my;
        cov += a * b; vx += a * a; vy += b * b;
    }
    return (vx > 1e-12 && vy > 1e-12) ? cov / std::sqrt(vx * vy) : 0.0;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    std::size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <csv_path> [n_boot] [ofi_block] [hawkes_blocks] [hawkes_iters]\n", argv[0]);
        return 1;
    }
    std::string csv        = argv[1];
    int n_boot             = argc > 2 ? std::atoi(argv[2]) : 1000;
    std::size_t ofi_block  = argc > 3 ? std::stoull(argv[3]) : 50;
    std::size_t n_blocks   = argc > 4 ? std::stoull(argv[4]) : 20;
    int hawkes_iters       = argc > 5 ? std::atoi(argv[5]) : 1000;

    std::vector<MarketEvent> feed = load_csv_feed(csv);
    if (feed.empty()) { std::fprintf(stderr, "empty feed: %s\n", csv.c_str()); return 1; }

    // --- Replay the tape once to extract Hawkes events + OFI/forward-return pairs ---
    // (same extraction logic as calibrate_hawkes.cpp)
    OrderBook book;
    MatchingEngine engine(book);
    RollingOFI ofi(/*depth=*/5, /*window=*/500);

    std::vector<HawkesEvent> hawkes_events;
    std::vector<double> ofi_signal, fwd_return, mids;
    struct Sample { double ofi; std::size_t idx; };
    std::vector<Sample> pending;
    const int horizon = 20;

    for (const auto& ev : feed) {
        Price bb = 0, ba = 0;
        book.best_bid(bb); book.best_ask(ba);
        double mid = (bb > 0 && ba > 0) ? (static_cast<double>(bb) + ba) / 2.0
                                        : (mids.empty() ? 0.0 : mids.back());
        mids.push_back(mid);
        ofi.update(book);

        if (ev.type == MdType::Trade) {
            hawkes_events.push_back(HawkesEvent{static_cast<double>(ev.ts),
                                                 ev.side == Side::Buy ? std::size_t(0) : std::size_t(1)});
            pending.push_back(Sample{ofi.rolling_sum(), mids.size() - 1});
        }
        apply_market_event(book, engine, ev);
    }
    for (const auto& s : pending) {
        std::size_t j = std::min(s.idx + horizon, mids.size() - 1);
        if (mids[s.idx] > 0.0 && mids[j] > 0.0) {
            ofi_signal.push_back(s.ofi);
            fwd_return.push_back(mids[j] - mids[s.idx]);
        }
    }

    std::printf("== Point estimates (full sample) ==\n");
    double point_corr = pearson_corr(ofi_signal, fwd_return);
    std::printf("OFI-vs-fwd-return corr : %.4f  (n=%zu)\n", point_corr, ofi_signal.size());

    if (hawkes_events.size() < 50) {
        std::fprintf(stderr, "too few aggressor events (%zu) to bootstrap Hawkes\n", hawkes_events.size());
        return 1;
    }
    double T = hawkes_events.back().t;
    double sum_dt = 0.0;
    for (std::size_t k = 1; k < hawkes_events.size(); ++k) sum_dt += hawkes_events[k].t - hawkes_events[k - 1].t;
    double mean_dt = sum_dt / (hawkes_events.size() - 1);
    double beta = 1.0 / mean_dt;
    std::vector<double> betas = {beta * 10.0, beta, beta / 10.0};
    std::vector<std::vector<std::vector<double>>> beta_mat(2, std::vector<std::vector<double>>(2, betas));

    MultivariateHawkes hp_point(2, beta_mat);
    auto fit_point = hp_point.fit_mle(hawkes_events, T, hawkes_iters, 0.05);
    auto n_point = hp_point.branching_ratio();
    double point_branching = (n_point[0][0] + n_point[0][1] + n_point[1][0] + n_point[1][1]) / 4.0;
    std::printf("Mean branching ratio   : %.4f  (converged=%d at %d iters, n_events=%zu)\n\n",
                point_branching, fit_point.converged, hawkes_iters, hawkes_events.size());

    std::mt19937_64 rng(42);

    // --- Bootstrap 1: OFI correlation, block-resampled over (ofi, fwd_return) pairs ---
    std::vector<double> boot_corrs;
    boot_corrs.reserve(n_boot);
    {
        std::size_t n = ofi_signal.size();
        std::size_t n_ofi_blocks = (n + ofi_block - 1) / ofi_block;
        std::uniform_int_distribution<std::size_t> block_pick(0, n_ofi_blocks - 1);
        for (int b = 0; b < n_boot; ++b) {
            std::vector<double> bx, by;
            bx.reserve(n); by.reserve(n);
            for (std::size_t blk = 0; blk < n_ofi_blocks; ++blk) {
                std::size_t src_blk = block_pick(rng);
                std::size_t start = src_blk * ofi_block;
                std::size_t end = std::min(start + ofi_block, n);
                for (std::size_t k = start; k < end; ++k) { bx.push_back(ofi_signal[k]); by.push_back(fwd_return[k]); }
            }
            boot_corrs.push_back(pearson_corr(bx, by));
        }
    }

    // --- Bootstrap 2: Hawkes branching ratio, time-block-resampled events ---
    std::vector<double> boot_branching;
    std::vector<int> boot_converged;
    boot_branching.reserve(n_boot);
    {
        double block_dur = T / static_cast<double>(n_blocks);
        // Bucket real events by which original time-block they fall in.
        std::vector<std::vector<HawkesEvent>> buckets(n_blocks);
        for (const auto& e : hawkes_events) {
            std::size_t blk = std::min(static_cast<std::size_t>(e.t / block_dur), n_blocks - 1);
            buckets[blk].push_back(e);
        }
        std::uniform_int_distribution<std::size_t> block_pick(0, n_blocks - 1);

        for (int b = 0; b < n_boot; ++b) {
            std::vector<HawkesEvent> resampled;
            resampled.reserve(hawkes_events.size());
            double slot_start = 0.0;
            for (std::size_t slot = 0; slot < n_blocks; ++slot) {
                std::size_t src_blk = block_pick(rng);
                double src_blk_start = src_blk * block_dur;
                for (const auto& e : buckets[src_blk]) {
                    double offset = e.t - src_blk_start;   // position within its original block
                    resampled.push_back(HawkesEvent{slot_start + offset, e.mark});
                }
                slot_start += block_dur;
            }
            std::sort(resampled.begin(), resampled.end(),
                      [](const HawkesEvent& a, const HawkesEvent& c) { return a.t < c.t; });
            if (resampled.size() < 20) { boot_branching.push_back(0.0); boot_converged.push_back(0); continue; }

            MultivariateHawkes hp_boot(2, beta_mat);
            auto res = hp_boot.fit_mle(resampled, slot_start, hawkes_iters, 0.05);
            auto n_boot_mat = hp_boot.branching_ratio();
            double mean_n = (n_boot_mat[0][0] + n_boot_mat[0][1] + n_boot_mat[1][0] + n_boot_mat[1][1]) / 4.0;
            boot_branching.push_back(mean_n);
            boot_converged.push_back(res.converged ? 1 : 0);

            if ((b + 1) % 100 == 0) std::printf("  ... bootstrap replicate %d/%d\n", b + 1, n_boot);
        }
    }

    int n_converged = 0;
    for (int c : boot_converged) n_converged += c;

    std::printf("\n== Bootstrap results (%d replicates) ==\n", n_boot);
    std::printf("OFI corr        95%% CI : [%.4f, %.4f]  (point estimate %.4f)\n",
                percentile(boot_corrs, 0.025), percentile(boot_corrs, 0.975), point_corr);
    std::printf("Branching ratio 95%% CI : [%.4f, %.4f]  (point estimate %.4f, %d/%d replicates converged)\n",
                percentile(boot_branching, 0.025), percentile(boot_branching, 0.975),
                point_branching, n_converged, n_boot);
    std::printf("\nNOTE: block-bootstrap CIs are an approximation -- block size/count "
                "were not tuned via a formal procedure (e.g. optimal block-length "
                "selection); treat these as indicative uncertainty bands, not exact "
                "confidence intervals.\n");
    return 0;
}
