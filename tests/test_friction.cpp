#include "backtest/friction.hpp"
#include "backtest/portfolio.hpp"
#include "test_util.hpp"

using namespace bt;

int main() {
    // Latency: non-negative, and jitter bounded around the base.
    {
        LatencyModel lat(50'000, 20'000, 123);
        for (int i = 0; i < 1000; ++i) {
            Timestamp s = lat.sample();
            CHECK(s >= 30'000 && s <= 70'000);
        }
        LatencyModel fixed(40'000, 0);
        CHECK_EQ(fixed.sample(), 40'000);
    }

    // Slippage: monotonic in participation, thin-book penalty applies.
    {
        SlippageModel sl(1.0);
        CHECK_EQ(sl.extra_ticks(100, 1000), 0);   // 10% participation -> 0.1 rounds to 0
        CHECK_EQ(sl.extra_ticks(500, 1000), 1);   // 50% -> 0.5 rounds to 1 (round-half-to-even/away)
        CHECK_EQ(sl.extra_ticks(2000, 1000), 2);  // 200% -> 2
        CHECK_EQ(sl.extra_ticks(100, 0), 1);      // empty book -> 1 tick penalty
        CHECK(sl.extra_ticks(1000, 1000) <= sl.extra_ticks(2000, 1000));  // monotone
    }

    // Portfolio: round-trip PnL and cash accounting (tick = $0.01).
    {
        Portfolio pf(1'000'000.0, 0.01);
        // buy 100 @ price 10000 ticks == $100.00
        pf.on_fill(Fill{1, Side::Buy, 10000, 100, 0, true, false});
        CHECK_NEAR(pf.cash(), 1'000'000.0 - 100 * 100.0, 1e-6);
        CHECK_EQ(pf.position(), 100);

        // mark up to $101 -> unrealized +$100
        pf.mark(1, 101.0);
        CHECK_NEAR(pf.pnl(), 100.0, 1e-6);

        // sell 100 @ 10100 ticks == $101.00 -> realize +$100
        pf.on_fill(Fill{2, Side::Sell, 10100, 100, 1, true, false});
        CHECK_EQ(pf.position(), 0);
        CHECK_NEAR(pf.realized(), 100.0, 1e-6);
        pf.mark(2, 101.0);
        CHECK_NEAR(pf.pnl(), 100.0, 1e-6);
        CHECK_EQ(pf.fills(), 2u);
    }

    // Short then cover.
    {
        Portfolio pf(1'000'000.0, 0.01);
        pf.on_fill(Fill{1, Side::Sell, 10000, 50, 0, true, false});  // short 50 @ $100
        CHECK_EQ(pf.position(), -50);
        pf.on_fill(Fill{2, Side::Buy, 9900, 50, 1, true, false});    // cover @ $99 -> +$50
        CHECK_EQ(pf.position(), 0);
        CHECK_NEAR(pf.realized(), 50.0, 1e-6);
    }

    // Regression test for the quantity-scale PnL accounting error
    // (Section on "quantity-scale PnL accounting error" in the writeup):
    // scaling both the fill quantity and qty_scale by the same factor must
    // produce identical real-unit dollar PnL, since qty_scale exists
    // precisely to represent "this many integer units equal one real
    // share/coin." Before the fix, Portfolio had no qty_scale parameter at
    // all and treated every integer unit as one whole share, so this test
    // would have failed (PnL would have scaled with the raw integer qty
    // instead of staying invariant).
    {
        // Baseline: qty_scale=1 (units ARE whole shares), buy 100 @ 99, mark @ 105.
        Portfolio pf_base(1'000'000.0, 1.0, /*qty_scale=*/1.0);
        pf_base.on_fill(Fill{1, Side::Buy, 99, 100, 0, true, false}, /*mid_at_fill=*/100.0);
        pf_base.mark(1, 105.0);

        // Same real trade (100 whole units), but represented as 100 * 1e6
        // integer "micro-units" with qty_scale=1e6 -- as real converters do
        // for fractional-crypto-quantity data.
        Portfolio pf_scaled(1'000'000.0, 1.0, /*qty_scale=*/1e6);
        pf_scaled.on_fill(Fill{1, Side::Buy, 99, 100 * 1'000'000, 0, true, false}, 100.0);
        pf_scaled.mark(1, 105.0);

        CHECK_NEAR(pf_base.pnl(), pf_scaled.pnl(), 1e-6);
        CHECK_NEAR(pf_base.decomposition().directional, pf_scaled.decomposition().directional, 1e-6);
        CHECK_NEAR(pf_base.decomposition().spread, pf_scaled.decomposition().spread, 1e-6);
    }

    test::report_and_reset("friction_portfolio");
    return test::failures() == 0 ? 0 : 1;
}
