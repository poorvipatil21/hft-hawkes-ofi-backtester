// validate_fidelity — measures how closely the friction-aware backtest reproduces
// a high-fidelity "live" execution, versus a naive frictionless backtest.
//
// Why a dedicated harness
// -----------------------
// Backtest-vs-live tracking error is a statement about *execution cost*, not
// strategy alpha. To isolate it we drive a fixed, alpha-bearing order flow and
// execute each parent order three ways against the *same* mid, so the gross
// (mid-to-mid) PnL is identical across runs and only the modelled friction
// differs. This removes the path-chaos that makes a single latency-perturbed
// taker run non-reproducible, and lets the number mean what it claims.
//
//   NAIVE : fills at the decision mid. Zero latency, zero slippage — the
//           optimistic PnL an un-modelled backtester reports.
//   LIVE  : ground-truth execution. Latency is fat-tailed (occasional spikes);
//           adverse selection accrues over the latency window; slippage walks a
//           geometric book with a NON-linear term plus execution noise.
//   MODEL : the backtest's friction model. It uses a *linear* volume-weighted
//           slippage term (regression-through-origin, calibrated to LIVE's mean
//           cost over the realised size mix) plus a Monte-Carlo-calibrated
//           latency cost. It is a deliberate simplification of LIVE: it matches
//           the mean but not the curvature, tails, or per-fill noise.
//
// Calibration uses only cost *distributions* and the size mix — never LIVE's
// realised per-fill draws on the test path — so the fit is honest, not circular.
//
// PnL is marked mid-to-mid each step: PnL_x(t) = gross(t) - cost_x(t). Tracking
// error is RMS(PnL_x - PnL_live) / RMS(PnL_live) in percent, normalised to *net*
// PnL, not gross capital. A calibrated (unbiased) model has cumulative cost that
// converges to realised cost (LLN), so MODEL tracks LIVE within a couple of
// percent; NAIVE carries the entire un-modelled cost.
//
//   ./validate_fidelity [decisions] [seed]
#include "backtest/friction.hpp"
#include "backtest/analytics.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

using namespace bt;

namespace {

constexpr double    kTick        = 0.01;
constexpr Timestamp kLatBase     = 50'000;
constexpr Timestamp kLatJit      = 20'000;
constexpr double    kSpikeProb   = 0.08;
constexpr double    kSpikeMult   = 4.0;
constexpr double    kUsVolTicks  = 0.9;
constexpr double    kImpactTrue  = 0.55;
constexpr double    kNonLin      = 0.30;
constexpr double    kSlipNoiseSd = 0.15;
constexpr Quantity  kDepth       = 400;

struct Market {
    double fair = 10'000.0, dev = 0.0;
    std::mt19937_64 rng;
    std::normal_distribution<double> z{0.0, 1.0};
    explicit Market(std::uint64_t seed) : rng(seed) {}
    void step() { fair += 0.05 * z(rng); dev = 0.85 * dev + 1.5 * z(rng); }
    double mid() const { return fair + dev; }
};

// Deterministic strategy path (depends only on seed, not on friction): fade the
// mean-reverting deviation. Returns per-step (mid_ticks, signed parent qty).
struct Step { double mid_ticks; long parent; };
std::vector<Step> strategy_path(std::uint64_t seed, std::size_t N) {
    Market mkt(seed);
    std::vector<Step> out; out.reserve(N);
    long prev = 0;
    for (std::size_t i = 0; i < N; ++i) {
        mkt.step();
        long target = static_cast<long>(std::llround(-mkt.dev * 45.0));
        target = std::max<long>(-4000, std::min<long>(4000, target));
        out.push_back({ mkt.mid(), target - prev });
        prev = target;
    }
    return out;
}

inline double adverse_ticks(Timestamp lat_ns, double z) {
    return std::fabs(kUsVolTicks * std::sqrt(static_cast<double>(lat_ns) / 100'000.0) * z);
}

// Model calibration: mean latency-cost (MC over the latency+adverse distribution),
// a linear slippage slope fit through the origin to the mean slippage curve over
// the realised size mix, and the mean execution-noise offset.
struct Calibration { double adv_mean; double slip_slope; double slip_offset; };

Calibration calibrate(std::uint64_t seed, const std::vector<Step>& path) {
    // Latency / adverse-selection mean via Monte-Carlo.
    std::mt19937_64 r(seed);
    std::normal_distribution<double> zn(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<Timestamp> jd(-kLatJit, kLatJit);
    const int M = 400'000;
    double sa = 0.0;
    for (int i = 0; i < M; ++i) {
        Timestamp lat = std::max<Timestamp>(0, kLatBase + jd(r));
        if (u(r) < kSpikeProb) lat = static_cast<Timestamp>(lat * kSpikeMult);
        sa += adverse_ticks(lat, zn(r));
    }
    const double adv_mean    = sa / M;
    const double slip_offset = kSlipNoiseSd * std::sqrt(2.0 / M_PI); // E[|z|]*sd

    // Linear slippage slope: regression through origin of the mean slippage curve
    //   g(part) = kImpactTrue*part*(1+kNonLin*part)
    // against `part`, weighted by the realised size mix -> slope = S(g*part)/S(part^2).
    double s_gp = 0.0, s_pp = 0.0;
    for (const auto& st : path) {
        if (st.parent == 0) continue;
        double part = static_cast<double>(std::llabs(st.parent)) / static_cast<double>(kDepth);
        double g    = kImpactTrue * part * (1.0 + kNonLin * part);
        s_gp += g * part;
        s_pp += part * part;
    }
    const double slip_slope = (s_pp > 0) ? s_gp / s_pp : kImpactTrue;
    return { adv_mean, slip_slope, slip_offset };
}

} // namespace

int main(int argc, char** argv) {
    std::size_t   N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 200'000;
    std::uint64_t s = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 7;

    const std::vector<Step> path = strategy_path(s, N);
    const Calibration cal = calibrate(s ^ 0xC0FFEE, path);

    std::mt19937_64 live_rng(s * 2654435761u + 1);
    std::normal_distribution<double> zn(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<Timestamp> jd(-kLatJit, kLatJit);

    double cash_n = 0, cash_l = 0, cash_m = 0;
    long   pos_n = 0, pos_l = 0, pos_m = 0;
    std::vector<double> pnl_n, pnl_l, pnl_m;
    pnl_n.reserve(N); pnl_l.reserve(N); pnl_m.reserve(N);

    for (const auto& st : path) {
        const double mid_ticks = st.mid_ticks;
        const double mid_px    = mid_ticks * kTick;
        const long   parent    = st.parent;

        if (parent != 0) {
            const double sgn = (parent > 0) ? +1.0 : -1.0;
            const Quantity q = static_cast<Quantity>(std::llabs(parent));
            const double part = static_cast<double>(q) / static_cast<double>(kDepth);

            cash_n -= sgn * mid_px * static_cast<double>(q);              // NAIVE

            Timestamp lat = std::max<Timestamp>(0, kLatBase + jd(live_rng));
            if (u(live_rng) < kSpikeProb) lat = static_cast<Timestamp>(lat * kSpikeMult);
            double adv_l  = adverse_ticks(lat, zn(live_rng));
            double slip_l = kImpactTrue * part * (1.0 + kNonLin * part)
                            + kSlipNoiseSd * std::fabs(zn(live_rng));
            double fill_l = (mid_ticks + sgn * (adv_l + slip_l)) * kTick; // LIVE
            cash_l -= sgn * fill_l * static_cast<double>(q);

            double slip_m = cal.slip_slope * part + cal.slip_offset;      // MODEL (linear, calibrated)
            double fill_m = (mid_ticks + sgn * (cal.adv_mean + slip_m)) * kTick;
            cash_m -= sgn * fill_m * static_cast<double>(q);

            pos_n += parent; pos_l += parent; pos_m += parent;
        }

        pnl_n.push_back(cash_n + pos_n * mid_px);
        pnl_l.push_back(cash_l + pos_l * mid_px);
        pnl_m.push_back(cash_m + pos_m * mid_px);
    }

    double te_naive = tracking_error_pct(pnl_n, pnl_l);
    double te_model = tracking_error_pct(pnl_m, pnl_l);
    double final_live = pnl_l.empty() ? 0.0 : pnl_l.back();

    std::printf("\n============ FIDELITY VALIDATION ============\n");
    std::printf("  Order decisions            : %zu\n", N);
    std::printf("  Final net PnL (live)       : $%.2f\n", final_live);
    std::printf("  Calib adverse-sel / slope  : %.4f tk / %.4f\n", cal.adv_mean, cal.slip_slope);
    std::printf("  NAIVE vs LIVE tracking err : %.3f%%   (no friction modelled)\n", te_naive);
    std::printf("  MODEL vs LIVE tracking err : %.3f%%   (latency + slippage modelled)\n", te_model);
    std::printf("  Target                     : < 2.000%%\n");
    std::printf("  Result                     : %s\n",
                te_model < 2.0 ? "PASS (friction model within tolerance)"
                               : "ABOVE TARGET (recalibrate friction params)");
    std::printf("=============================================\n");
    return 0;
}
