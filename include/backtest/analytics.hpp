#pragma once
#include "portfolio.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace bt {

struct BacktestReport {
    double      final_equity  = 0.0;
    double      total_pnl     = 0.0;
    double      return_pct    = 0.0;
    double      sharpe        = 0.0;   // per-step, sqrt(N)-scaled
    double      max_drawdown  = 0.0;   // as a positive fraction
    std::size_t fills         = 0;
    std::size_t events        = 0;
    Quantity    end_position  = 0;
};

// Per-step Sharpe of an equity curve, scaled by sqrt(number of steps).
inline double sharpe_ratio(const std::vector<EquityPoint>& c) {
    if (c.size() < 3) return 0.0;
    std::vector<double> r;
    r.reserve(c.size());
    for (std::size_t i = 1; i < c.size(); ++i) {
        double prev = c[i - 1].equity;
        if (std::abs(prev) > 1e-9) r.push_back((c[i].equity - prev) / prev);
    }
    if (r.size() < 2) return 0.0;
    double mean = 0.0; for (double x : r) mean += x; mean /= r.size();
    double var  = 0.0; for (double x : r) var += (x - mean) * (x - mean); var /= (r.size() - 1);
    double sd   = std::sqrt(var);
    return sd > 1e-12 ? (mean / sd) * std::sqrt(static_cast<double>(r.size())) : 0.0;
}

// Largest peak-to-trough decline of the equity curve, as a positive fraction.
inline double max_drawdown(const std::vector<EquityPoint>& c) {
    double peak = -1e300, mdd = 0.0;
    for (const auto& p : c) {
        peak = std::max(peak, p.equity);
        if (peak > 1e-9) mdd = std::max(mdd, (peak - p.equity) / peak);
    }
    return mdd;
}

inline BacktestReport summarize(const Portfolio& pf, std::size_t events) {
    BacktestReport rep;
    const auto& c     = pf.curve();
    double start_eq   = c.empty() ? 0.0 : c.front().equity;
    rep.final_equity  = pf.equity();
    rep.total_pnl     = pf.pnl();
    rep.return_pct    = start_eq > 1e-9 ? 100.0 * rep.total_pnl / start_eq : 0.0;
    rep.sharpe        = sharpe_ratio(c);
    rep.max_drawdown  = max_drawdown(c);
    rep.fills         = pf.fills();
    rep.events        = events;
    rep.end_position  = pf.position();
    return rep;
}

// RMS relative tracking error between two aligned PnL/equity series, in percent.
//   sqrt(mean((a-b)^2)) / rms(b) * 100
// This is the metric used to compare a friction-aware backtest against a
// reference (e.g. live) execution series.
inline double tracking_error_pct(const std::vector<double>& a,
                                  const std::vector<double>& b) {
    std::size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    double sse = 0.0, ssb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        sse += d * d;
        ssb += b[i] * b[i];
    }
    double rms_b = std::sqrt(ssb / n);
    if (rms_b < 1e-12) return 0.0;
    return 100.0 * std::sqrt(sse / n) / rms_b;
}

// Convenience: pull the equity column out of a curve.
inline std::vector<double> equity_series(const std::vector<EquityPoint>& c) {
    std::vector<double> v; v.reserve(c.size());
    for (const auto& p : c) v.push_back(p.equity);
    return v;
}

} // namespace bt
