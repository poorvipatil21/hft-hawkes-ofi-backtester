#pragma once
#include "order_book.hpp"
#include "strategy.hpp"
#include <deque>
#include <cstddef>

namespace bt {

// Order Flow Imbalance (Cont, Kukanov, Stoikov 2014).
//
// Per level m, comparing consecutive book states n-1 -> n:
//   OFI_m = I(bid_px_n >= bid_px_{n-1}) * bid_qty_n
//         - I(bid_px_n <= bid_px_{n-1}) * bid_qty_{n-1}
//         - I(ask_px_n <= ask_px_{n-1}) * ask_qty_n
//         + I(ask_px_n >= ask_px_{n-1}) * ask_qty_{n-1}
//
// Interpretation: buy-side book growth / ask-side depletion pushes OFI up,
// and empirically leads short-horizon mid-price returns. Multi-level OFI
// (summed over the top M levels) captures depth-building away from the
// touch that L1-only OFI misses.

struct LevelState {
    Price    px  = 0;
    Quantity qty = 0;
    bool     has = false;
};

// Snapshot of the top M levels of both sides, taken directly off the book
// (not the L1 MarketSnapshot the strategies see).
struct BookLevels {
    std::vector<LevelState> bids;  // best -> worst
    std::vector<LevelState> asks;
};

inline BookLevels snapshot_levels(const OrderBook& book, std::size_t depth) {
    BookLevels out;
    out.bids.reserve(depth);
    out.asks.reserve(depth);
    std::size_t i = 0;
    for (const auto& [px, lvl] : book.bids()) {
        if (i++ >= depth) break;
        out.bids.push_back(LevelState{px, lvl.total_qty, true});
    }
    i = 0;
    for (const auto& [px, lvl] : book.asks()) {
        if (i++ >= depth) break;
        out.asks.push_back(LevelState{px, lvl.total_qty, true});
    }
    return out;
}

inline double level_ofi(const LevelState& bid_n, const LevelState& bid_p,
                         const LevelState& ask_n, const LevelState& ask_p) {
    double bid_term = 0.0, ask_term = 0.0;
    if (bid_n.has && bid_p.has) {
        if (bid_n.px >= bid_p.px) bid_term += static_cast<double>(bid_n.qty);
        if (bid_n.px <= bid_p.px) bid_term -= static_cast<double>(bid_p.qty);
    } else if (bid_n.has && !bid_p.has) {
        bid_term += static_cast<double>(bid_n.qty);   // level appeared
    } else if (!bid_n.has && bid_p.has) {
        bid_term -= static_cast<double>(bid_p.qty);   // level vanished
    }
    if (ask_n.has && ask_p.has) {
        if (ask_n.px <= ask_p.px) ask_term += static_cast<double>(ask_n.qty);
        if (ask_n.px >= ask_p.px) ask_term -= static_cast<double>(ask_p.qty);
    } else if (ask_n.has && !ask_p.has) {
        ask_term += static_cast<double>(ask_n.qty);
    } else if (!ask_n.has && ask_p.has) {
        ask_term -= static_cast<double>(ask_p.qty);
    }
    return bid_term - ask_term;
}

// Multi-level OFI between two book snapshots, summed over min(depth-available).
inline double multilevel_ofi(const BookLevels& prev, const BookLevels& curr, std::size_t depth) {
    double total = 0.0;
    for (std::size_t m = 0; m < depth; ++m) {
        LevelState bp = m < prev.bids.size() ? prev.bids[m] : LevelState{};
        LevelState bn = m < curr.bids.size() ? curr.bids[m] : LevelState{};
        LevelState ap = m < prev.asks.size() ? prev.asks[m] : LevelState{};
        LevelState an = m < curr.asks.size() ? curr.asks[m] : LevelState{};
        total += level_ofi(bn, bp, an, ap);
    }
    return total;
}

// Streaming rolling OFI: maintains the previous book snapshot and a trailing
// window (by event count) of per-step OFI contributions, exposing both the
// instantaneous increment and a normalized rolling sum usable as a signal.
class RollingOFI {
public:
    explicit RollingOFI(std::size_t depth = 5, std::size_t window = 500)
        : depth_(depth), window_(window) {}

    // Call once per market-data event. Returns the instantaneous OFI increment.
    double update(const OrderBook& book) {
        BookLevels curr = snapshot_levels(book, depth_);
        double inc = 0.0;
        if (have_prev_) inc = multilevel_ofi(prev_, curr, depth_);
        prev_ = std::move(curr);
        have_prev_ = true;

        hist_.push_back(inc);
        sum_ += inc;
        if (hist_.size() > window_) { sum_ -= hist_.front(); hist_.pop_front(); }
        return inc;
    }

    double rolling_sum() const { return sum_; }
    // Normalizes by trailing average traded size so the signal is roughly
    // scale-free across symbols/regimes (a common OFI convention).
    double normalized(double avg_qty_scale) const {
        return avg_qty_scale > 1e-9 ? sum_ / avg_qty_scale : 0.0;
    }
    std::size_t count() const { return hist_.size(); }

private:
    std::size_t         depth_, window_;
    BookLevels           prev_;
    bool                 have_prev_ = false;
    std::deque<double>   hist_;
    double               sum_ = 0.0;
};

} // namespace bt
