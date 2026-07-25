#pragma once
#include "backtest/strategy.hpp"
#include <deque>
#include <cmath>

namespace bt {

// Z-score mean reversion that crosses the spread with IOC orders. Because it
// takes liquidity, it is the clearest showcase for latency + volume-weighted
// slippage: every entry pays the friction the model imposes.
class MeanReversion : public Strategy {
public:
    MeanReversion(int window = 50, double entry_z = 2.0, Quantity size = 100)
        : Strategy("MeanReversion"), window_(window), entry_z_(entry_z), size_(size) {}

    void on_market_data(const MarketSnapshot& s, OrderContext& ctx) override {
        if (!s.has_bid || !s.has_ask) return;
        double mid = s.mid();
        push(mid);
        if (static_cast<int>(buf_.size()) < window_) return;

        double m  = mean();
        double sd = stdev(m);
        if (sd < 1e-9) return;
        double z = (mid - m) / sd;

        if (z > entry_z_ && pos_ > -size_) {
            ctx.submit(Side::Sell, OrderType::IOC, s.best_bid, size_);          // fade the up-move
        } else if (z < -entry_z_ && pos_ < size_) {
            ctx.submit(Side::Buy, OrderType::IOC, s.best_ask, size_);           // fade the down-move
        } else if (std::abs(z) < 0.5 && pos_ != 0) {                            // revert to flat
            if (pos_ > 0) ctx.submit(Side::Sell, OrderType::IOC, s.best_bid,  pos_);
            else          ctx.submit(Side::Buy,  OrderType::IOC, s.best_ask, -pos_);
        }
    }

    void on_fill(const Fill& f, OrderContext&) override {
        pos_ += (f.side == Side::Buy ? +f.qty : -f.qty);
    }

private:
    void   push(double x) { buf_.push_back(x); if (static_cast<int>(buf_.size()) > window_) buf_.pop_front(); }
    double mean() const   { double s = 0; for (double v : buf_) s += v; return s / buf_.size(); }
    double stdev(double m) const {
        double s = 0; for (double v : buf_) s += (v - m) * (v - m);
        return std::sqrt(s / buf_.size());
    }

    int              window_;
    double           entry_z_;
    Quantity         size_, pos_ = 0;
    std::deque<double> buf_;
};

} // namespace bt
