#pragma once
#include "backtest/strategy.hpp"

namespace bt {

// Two-sided market maker that quotes inside the market spread and skews its
// quotes against inventory to mean-revert toward flat. Demonstrates resting
// (maker) fills, order-state transitions, and adverse selection under latency.
class MarketMaker : public Strategy {
public:
    MarketMaker(Quantity size = 50, Price half_spread = 1, Quantity max_pos = 500)
        : Strategy("MarketMaker"), size_(size), half_spread_(half_spread), max_pos_(max_pos) {}

    void on_market_data(const MarketSnapshot& s, OrderContext& ctx) override {
        if (!s.has_bid || !s.has_ask) return;
        Price mid = (s.best_bid + s.best_ask) / 2;
        if (mid == last_mid_ && have_quotes_) return;   // only requote when the mid moves
        last_mid_ = mid;

        if (bid_id_) ctx.cancel(bid_id_);
        if (ask_id_) ctx.cancel(ask_id_);

        // Inventory skew: long inventory pushes both quotes down (eager to sell).
        Price skew  = static_cast<Price>(static_cast<double>(inventory_) / max_pos_ * half_spread_);
        Price bidpx = mid - half_spread_ - skew;
        Price askpx = mid + half_spread_ - skew;

        Quantity bsz = (inventory_ <  max_pos_) ? size_ : 0;
        Quantity asz = (inventory_ > -max_pos_) ? size_ : 0;

        bid_id_ = bsz ? ctx.submit(Side::Buy,  OrderType::Limit, bidpx, bsz) : 0;
        ask_id_ = asz ? ctx.submit(Side::Sell, OrderType::Limit, askpx, asz) : 0;
        have_quotes_ = true;
    }

    void on_fill(const Fill& f, OrderContext&) override {
        inventory_ += (f.side == Side::Buy ? +f.qty : -f.qty);
        if (f.order_id == bid_id_) bid_id_ = 0;
        if (f.order_id == ask_id_) ask_id_ = 0;
        have_quotes_ = false;   // force a requote after any fill
    }

private:
    Quantity size_;
    Price    half_spread_;
    Quantity max_pos_;
    Price    last_mid_   = -1;
    OrderId  bid_id_     = 0, ask_id_ = 0;
    Quantity inventory_  = 0;
    bool     have_quotes_ = false;
};

} // namespace bt
