#pragma once
#include "backtest/strategy.hpp"
#include "backtest/ofi.hpp"
#include "backtest/hawkes.hpp"
#include "backtest/data_feed.hpp"
#include <deque>
#include <algorithm>
#include <vector>

namespace bt {

// A two-sided market maker whose quote skew is driven by a fused signal:
//
//   signal = w_ofi * normalized_OFI + w_hawkes * intensity_imbalance
//   intensity_imbalance = (lambda_buy - lambda_sell) / (lambda_buy + lambda_sell)
//
// where lambda_{buy,sell} come from a Hawkes process fit offline (see
// apps/calibrate_hawkes.cpp) and updated online here via the same recursive
// R_ijp state (per source dim, target dim, and kernel component -- see
// MultivariateHawkes's sum-of-exponentials kernel), re-seeded from live
// aggressor arrivals. Quotes widen/skew away from the side the signal says
// is about to be run over, and inventory still pulls skew back toward flat
// (same risk control as the baseline MarketMaker).
//
// This is the strategy the ICAIF empirical section backtests against the
// unconditioned baseline MarketMaker to show the OFI/Hawkes signal's PnL and
// adverse-selection contribution.
class HawkesOFIMarketMaker : public Strategy {
public:
    struct Params {
        Quantity quote_size      = 100;
        Price    base_half_spread = 2;      // ticks
        Price    max_skew_ticks   = 3;
        double   inventory_gamma = 0.02;    // ticks of skew per unit inventory
        Quantity max_inventory    = 2000;
        double   w_ofi           = 0.6;
        double   w_hawkes        = 0.4;
        double   ofi_scale       = 500.0;   // divides rolling OFI to ~[-1,1]
        std::size_t ofi_depth    = 5;
        std::size_t ofi_window   = 500;
    };

    explicit HawkesOFIMarketMaker(Params p, std::vector<std::vector<std::vector<double>>> beta)
        : Strategy("HawkesOFI-MM"), p_(p), ofi_(p.ofi_depth, p.ofi_window), hawkes_(2, std::move(beta)) {
        std::size_t P = hawkes_.n_components();
        R_.assign(2, std::vector<std::vector<double>>(2, std::vector<double>(P, 0.0)));
    }

    // Allow pre-seeding with offline-calibrated mu/alpha before on_start.
    MultivariateHawkes& hawkes_process() { return hawkes_; }

    void on_market_data(const MarketSnapshot& s, OrderContext& ctx) override {
        if (!s.has_bid || !s.has_ask) return;

        double ofi_inc = ofi_.update(ctx.book());
        (void)ofi_inc;
        update_hawkes_decay(static_cast<double>(s.ts));

        std::size_t P = hawkes_.n_components();
        double lam_buy  = hawkes_.mu()[0];
        double lam_sell = hawkes_.mu()[1];
        for (std::size_t j = 0; j < 2; ++j) {
            for (std::size_t p = 0; p < P; ++p) {
                lam_buy  += hawkes_.alpha()[0][j][p] * R_[0][j][p];
                lam_sell += hawkes_.alpha()[1][j][p] * R_[1][j][p];
            }
        }
        double denom = lam_buy + lam_sell;
        double intensity_imbalance = denom > 1e-9 ? (lam_buy - lam_sell) / denom : 0.0;
        double ofi_norm = ofi_.normalized(p_.ofi_scale);
        ofi_norm = std::clamp(ofi_norm, -1.0, 1.0);

        // Positive signal => expect upward pressure => skew quotes up (raise
        // both, lean into offering less ask depth / more bid depth) by
        // widening on the side we don't want run over.
        double signal = p_.w_ofi * ofi_norm + p_.w_hawkes * intensity_imbalance;

        double inv_skew  = -p_.inventory_gamma * static_cast<double>(inventory_);
        double sig_skew  = signal * static_cast<double>(p_.max_skew_ticks);
        Price  skew      = static_cast<Price>(std::round(std::clamp(inv_skew + sig_skew,
                                    -static_cast<double>(p_.max_skew_ticks),
                                     static_cast<double>(p_.max_skew_ticks))));

        cancel_quotes(ctx);
        Price mid_px = static_cast<Price>((s.best_bid + s.best_ask) / 2);
        Price bid_px = mid_px - p_.base_half_spread + skew;
        Price ask_px = mid_px + p_.base_half_spread + skew;

        if (inventory_ >= p_.max_inventory) {
            // Over the cap on the long side: stop adding more risk, but keep
            // actively quoting the offer so the position can trade back down
            // toward flat -- freezing entirely (the old behavior) means a
            // single overshoot event permanently locks in whatever inventory
            // existed at that moment for the rest of the backtest, turning a
            // market-making strategy into a static directional bet with no
            // mechanism to unwind.
            ask_id_ = ctx.submit(Side::Sell, OrderType::Limit, ask_px, p_.quote_size);
        } else if (inventory_ <= -p_.max_inventory) {
            // Symmetric case: over the cap short, keep bidding to buy back.
            bid_id_ = ctx.submit(Side::Buy, OrderType::Limit, bid_px, p_.quote_size);
        } else {
            bid_id_ = ctx.submit(Side::Buy,  OrderType::Limit, bid_px, p_.quote_size);
            ask_id_ = ctx.submit(Side::Sell, OrderType::Limit, ask_px, p_.quote_size);
        }
    }

    void on_fill(const Fill& f, OrderContext&) override {
        inventory_ += (f.side == Side::Buy ? 1 : -1) * f.qty;
        // Our own fills are already reflected in the tape as the Trade event
        // that generated them (on_raw_event below), so nothing to mark here --
        // this just tracks inventory.
    }

    // External aggressor flow: this is the actual Hawkes mark process. Our
    // own resulting fills are a consequence of this same Trade event, not a
    // separate arrival, so on_fill (above) must not also call mark_event.
    void on_raw_event(const MarketEvent& me, OrderContext&) override {
        if (me.type != MdType::Trade) return;
        update_hawkes_decay(static_cast<double>(me.ts));
        mark_event(me.side == Side::Buy ? 0 : 1);
    }

    // Call this when a Trade MarketEvent is observed (external aggressor),
    // to keep the online Hawkes state current. Wire this from the tape if
    // the strategy is extended with a raw-event hook.
    void mark_event(std::size_t which) {
        std::size_t P = hawkes_.n_components();
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t p = 0; p < P; ++p)
                R_[i][which][p] += 1.0;
    }

    Quantity inventory() const { return inventory_; }

private:
    void cancel_quotes(OrderContext& ctx) {
        if (bid_id_) ctx.cancel(bid_id_);
        if (ask_id_) ctx.cancel(ask_id_);
        bid_id_ = ask_id_ = 0;
    }

    void update_hawkes_decay(double t) {
        if (have_prev_t_) {
            double dt = t - prev_t_;
            std::size_t P = hawkes_.n_components();
            for (std::size_t i = 0; i < 2; ++i)
                for (std::size_t j = 0; j < 2; ++j)
                    for (std::size_t p = 0; p < P; ++p)
                        R_[i][j][p] *= std::exp(-hawkes_.beta()[i][j][p] * dt);
        }
        prev_t_ = t;
        have_prev_t_ = true;
    }

    Params p_;
    RollingOFI ofi_;
    MultivariateHawkes hawkes_;
    std::vector<std::vector<std::vector<double>>> R_;   // [i][j][p]
    double prev_t_ = 0.0;
    bool   have_prev_t_ = false;

    Quantity inventory_ = 0;
    OrderId  bid_id_ = 0, ask_id_ = 0;
};

} // namespace bt
