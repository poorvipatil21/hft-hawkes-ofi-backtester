#pragma once
#include "types.hpp"
#include <random>
#include <algorithm>

namespace bt {

// Wire + processing latency between a strategy decision and the order actually
// reaching the matching engine. Modelled as a base delay plus symmetric jitter.
class LatencyModel {
public:
    LatencyModel(Timestamp base_ns = 50'000,     // 50 us default
                 Timestamp jitter_ns = 20'000,   // +/- 20 us
                 std::uint64_t seed = 42)
        : base_(base_ns), jitter_(jitter_ns), rng_(seed) {}

    Timestamp sample() {
        if (jitter_ <= 0) return base_;
        std::uniform_int_distribution<Timestamp> d(-jitter_, jitter_);
        return std::max<Timestamp>(0, base_ + d(rng_));
    }
    Timestamp base() const { return base_; }

private:
    Timestamp       base_;
    Timestamp       jitter_;
    std::mt19937_64 rng_;
};

// Volume-weighted slippage / market-impact.
//
// Sweeping several price levels already produces a natural VWAP worse than the
// touch — the matching engine handles that mechanically. This term adds the
// *residual* adverse move driven by participation rate (order size relative to
// the visible size being consumed), capturing queue position and hidden-size
// effects that a level-1 book does not show.
//
//   extra_ticks = round( impact_coeff * (qty / visible) )
class SlippageModel {
public:
    explicit SlippageModel(double impact_coeff = 0.5)
        : impact_(impact_coeff) {}

    Price extra_ticks(Quantity qty, Quantity visible) const {
        if (visible <= 0) return 1;   // hitting an empty/thin book -> 1 tick penalty
        double participation = static_cast<double>(qty) / static_cast<double>(visible);
        return static_cast<Price>(std::llround(impact_ * participation));
    }

    double impact_coeff() const { return impact_; }

private:
    double impact_;
};

} // namespace bt
