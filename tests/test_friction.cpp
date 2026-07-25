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

    test::report_and_reset("friction_portfolio");
    return test::failures() == 0 ? 0 : 1;
}
