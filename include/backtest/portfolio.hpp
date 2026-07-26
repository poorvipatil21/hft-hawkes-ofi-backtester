#pragma once
#include "order.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace bt {

struct EquityPoint { Timestamp ts; double equity; };

// Tracks a single-instrument position and accumulates PnL from fills.
// Cash accounting is signed: buys reduce cash, sells increase it, and equity
// is marked to the prevailing mid.
//
// qty_scale lets Quantity represent a FRACTION of one real share/coin --
// e.g. when real-data converters scale fractional BTC into integer units
// (qty_scale=1e6 meaning "1 unit == 1 micro-BTC") to avoid the engine's
// int64 Quantity rounding sub-1-unit sizes to 0. Without this, every dollar
// computation here would silently treat "1100 units" as 1100 WHOLE coins
// instead of 1100 micro-coins, producing PnL/equity errors of exactly that
// scale factor -- position_ itself stays a raw integer (unchanged, since
// OrderBook/matching only care about relative units), only the DOLLAR
// math below divides by qty_scale to get back to real-unit terms.
class Portfolio {
public:
    explicit Portfolio(double starting_cash = 1'000'000.0, double tick_size = 0.01,
                        double qty_scale = 1.0)
        : cash_(starting_cash), start_(starting_cash), tick_(tick_size), qty_scale_(qty_scale) {}

    void on_fill(const Fill& f, double mid_at_fill = 0.0) {
        // Advance directional PnL for the OLD position up to this fill's mid
        // *before* applying the fill's position change -- see advance_pnl_clock's
        // doc comment for why this must share one timeline with mark().
        if (mid_at_fill > 0.0) advance_pnl_clock(mid_at_fill);

        const double px  = static_cast<double>(f.price) * tick_;
        const double sq  = (f.side == Side::Buy ? +1.0 : -1.0) * static_cast<double>(f.qty);
        const double pos = static_cast<double>(position_);

        // Realize PnL on the portion that reduces the existing position.
        // Divide by qty_scale here (not earlier) since avg_px_'s weighted
        // update below deliberately uses raw (unscaled) magnitudes -- the
        // weighting ratio is unaffected by a constant scale applied
        // uniformly to both terms, so only the final dollar PnL needs the
        // real-unit conversion.
        if ((pos > 0 && sq < 0) || (pos < 0 && sq > 0)) {
            double closing = std::min(std::abs(sq), std::abs(pos));
            double dir     = pos > 0 ? +1.0 : -1.0;
            realized_ += dir * (px - avg_px_) * (closing / qty_scale_);
        }

        const double new_pos = pos + sq;
        if ((pos >= 0 && sq > 0) || (pos <= 0 && sq < 0)) {
            // Increasing the position -> update the volume-weighted average price.
            double a = std::abs(pos), b = std::abs(sq);
            avg_px_ = (a + b) > 0 ? (avg_px_ * a + px * b) / (a + b) : px;
        } else if ((pos > 0 && new_pos < 0) || (pos < 0 && new_pos > 0)) {
            avg_px_ = px;  // position flipped through zero
        }

        cash_     -= (sq / qty_scale_) * px;
        position_  = static_cast<Quantity>(std::llround(new_pos));
        ++fills_;

        // PnL decomposition, part 1/2: spread-capture ("edge") PnL -- how
        // favorable the execution price was relative to the prevailing mid
        // at the moment of the fill, independent of any subsequent price
        // movement. A market maker earning the spread should show positive
        // spread_pnl_ regardless of which way the market later drifts; this
        // is what lets us separate "the strategy is good at making markets"
        // from "the strategy happened to be long/short when price moved,"
        // the confound that flipped our own strategy-comparison's sign
        // across different capture windows before this was added.
        if (mid_at_fill > 0.0) {
            double real_qty = static_cast<double>(f.qty) / qty_scale_;
            double edge = (f.side == Side::Buy) ? (mid_at_fill - px) : (px - mid_at_fill);
            spread_pnl_ += edge * real_qty;
            ++edge_tracked_count_;
            if (edge > 0.0) ++edge_positive_count_;
        }

        // Turnover: cumulative traded notional (real units x execution
        // price), independent of direction -- a standard measure of how
        // much trading activity a strategy generates relative to its
        // capital base, reported as a multiple of starting equity.
        traded_notional_ += (static_cast<double>(f.qty) / qty_scale_) * px;

        // The NEW position starts accruing directional PnL from this fill's
        // mid onward (ref_mid_ was just set to mid_at_fill by
        // advance_pnl_clock above).
        ref_position_ = position_;
    }

    // Mark to market and append a point to the equity curve.
    void mark(Timestamp ts, double mid_px) {
        advance_pnl_clock(mid_px);
        last_mark_ = mid_px;
        curve_.push_back({ts, equity(mid_px)});
    }

    double   equity(double mid_px) const {
        return cash_ + (static_cast<double>(position_) / qty_scale_) * mid_px;
    }
    double   equity()   const {
        return cash_ + (static_cast<double>(position_) / qty_scale_) * last_mark_;
    }
    double   pnl()      const { return equity() - start_; }
    double   realized() const { return realized_; }
    double   unrealized() const {
        return (static_cast<double>(position_) / qty_scale_) * (last_mark_ - avg_px_);
    }
    double   cash()     const { return cash_; }
    Quantity position() const { return position_; }
    std::size_t fills() const { return fills_; }
    const std::vector<EquityPoint>& curve() const { return curve_; }

    // See the doc comments on on_fill/mark above for what each component
    // means. The residual should now be at most floating-point-scale (both
    // components share one reference-mid timeline via advance_pnl_clock),
    // not a structural artifact of fill/mark timing.
    struct PnLDecomposition {
        double directional;   // holding/inventory PnL from price moves
        double spread;        // spread-capture/edge PnL from execution quality
        double total;         // == pnl(), for convenience
        double residual;      // total - (directional + spread); should be ~0
    };
    PnLDecomposition decomposition() const {
        double tot = pnl();
        return PnLDecomposition{directional_pnl_, spread_pnl_, tot,
                                 tot - (directional_pnl_ + spread_pnl_)};
    }

    // Turnover as a multiple of starting equity (traded notional / starting
    // capital) -- a standard activity-normalized measure, comparable across
    // strategies with different capital bases.
    double turnover() const { return start_ > 1e-9 ? traded_notional_ / start_ : 0.0; }

    // Fraction of fills whose execution price was favorable relative to the
    // prevailing mid at the moment of the fill (i.e. positive spread-capture
    // edge on that individual fill) -- an execution-quality "hit rate"
    // distinct from a directional win-rate, since it's evaluated per-fill
    // against the contemporaneous mid, not against any later price.
    double hit_rate() const {
        return edge_tracked_count_ > 0
             ? static_cast<double>(edge_positive_count_) / edge_tracked_count_ : 0.0;
    }

private:
    // Advances the shared directional-PnL reference-mid timeline that both
    // on_fill and mark() use. This MUST be the only place either component
    // updates the "clock" -- an earlier version called this logic
    // independently from on_fill and mark with their own separate mid
    // references, which (because the engine's latency model processes a
    // strategy's own resting-order fills at a different point in the event
    // loop than periodic equity marks) could reference the book at two
    // slightly different moments for what should have been the same
    // instant, producing a persistent, non-shrinking residual that we
    // initially (incorrectly) suspected was a mark-frequency artifact --
    // it wasn't; running with marks after literally every event left the
    // residual completely unchanged, which is what revealed the real cause.
    void advance_pnl_clock(double new_mid) {
        if (has_ref_) {
            directional_pnl_ += (static_cast<double>(ref_position_) / qty_scale_) * (new_mid - ref_mid_);
        }
        ref_mid_ = new_mid;
        has_ref_ = true;
    }

    double      cash_, start_, tick_, qty_scale_;
    Quantity    position_  = 0;
    double      avg_px_    = 0.0;
    double      last_mark_ = 0.0;
    double      realized_  = 0.0;
    std::size_t fills_     = 0;
    std::vector<EquityPoint> curve_;

    // PnL decomposition state (single shared reference-mid timeline)
    double      directional_pnl_ = 0.0;
    double      spread_pnl_      = 0.0;
    Quantity    ref_position_    = 0;
    double      ref_mid_         = 0.0;
    bool        has_ref_         = false;

    // Turnover and hit-rate tracking
    double      traded_notional_     = 0.0;
    std::size_t edge_tracked_count_  = 0;
    std::size_t edge_positive_count_ = 0;
};

} // namespace bt
