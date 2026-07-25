#include "backtest/order_book.hpp"
#include "test_util.hpp"
#include <vector>

using namespace bt;

static Order mk(OrderId id, Side s, Price px, Quantity q, bool ours = false) {
    Order o; o.id = id; o.side = s; o.price = px; o.qty = q; o.remaining = q; o.is_ours = ours;
    return o;
}

int main() {
    // best bid/ask and depth aggregation
    {
        OrderBook b;
        b.insert(mk(1, Side::Buy,  99, 100));
        b.insert(mk(2, Side::Buy, 100, 50));
        b.insert(mk(3, Side::Buy, 100, 25));   // same level -> aggregated
        b.insert(mk(4, Side::Sell,101, 30));
        b.insert(mk(5, Side::Sell,102, 40));

        Price p = 0;
        CHECK(b.best_bid(p)); CHECK_EQ(p, 100);
        CHECK(b.best_ask(p)); CHECK_EQ(p, 101);
        CHECK_EQ(b.bid_size(100), 75);
        CHECK_EQ(b.ask_size(101), 30);
        CHECK_EQ(b.resting_count(), 5u);
    }

    // cancel removes an order and collapses an emptied level
    {
        OrderBook b;
        b.insert(mk(1, Side::Buy, 100, 100));
        b.insert(mk(2, Side::Buy, 100, 50));
        CHECK(b.cancel(1));
        CHECK_EQ(b.bid_size(100), 50);
        CHECK(!b.cancel(999));               // unknown id
        CHECK(b.cancel(2));
        Price p = 0;
        CHECK(!b.best_bid(p));               // level gone
        CHECK(b.empty());
    }

    // aggressive buy matches asks with price-time priority
    {
        OrderBook b;
        b.insert(mk(10, Side::Sell, 101, 40));   // first in queue at 101
        b.insert(mk(11, Side::Sell, 101, 60));   // behind #10
        b.insert(mk(12, Side::Sell, 102, 100));

        Order in = mk(20, Side::Buy, 102, 70, true);
        std::vector<Fill> fills;
        b.match(in, [&](const Fill& f) { fills.push_back(f); });

        CHECK_EQ(in.remaining, 0);                    // fully filled
        // maker fills: 40 @101 (#10), then 30 @101 (#11). Taker legs mirror them.
        Quantity taker_qty = 0; Price worst = 0;
        for (auto& f : fills) if (f.is_ours) { taker_qty += f.qty; worst = std::max(worst, f.price); }
        CHECK_EQ(taker_qty, 70);
        CHECK_EQ(worst, 101);                          // never reached 102
        CHECK_EQ(b.ask_size(101), 30);                 // 30 of #11 remain
        CHECK_EQ(b.ask_size(102), 100);
    }

    test::report_and_reset("order_book");
    return test::failures() == 0 ? 0 : 1;
}
