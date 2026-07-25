#include "backtest/matching_engine.hpp"
#include "test_util.hpp"
#include <vector>

using namespace bt;

static Order mk(OrderId id, Side s, OrderType t, Price px, Quantity q) {
    Order o; o.id = id; o.side = s; o.type = t; o.price = px; o.qty = q; o.remaining = q;
    return o;
}

int main() {
    // Limit remainder rests in the book
    {
        OrderBook b; MatchingEngine e(b);
        e.submit(mk(1, Side::Sell, OrderType::Limit, 101, 30), [](const Fill&){});
        std::vector<Fill> fills;
        e.submit(mk(2, Side::Buy, OrderType::Limit, 101, 100), [&](const Fill& f){ fills.push_back(f); });
        // 30 traded, 70 rests as a bid at 101
        Price p = 0; CHECK(b.best_bid(p)); CHECK_EQ(p, 101);
        CHECK_EQ(b.bid_size(101), 70);
    }

    // IOC cancels the unfilled remainder (never rests)
    {
        OrderBook b; MatchingEngine e(b);
        e.submit(mk(1, Side::Sell, OrderType::Limit, 101, 30), [](const Fill&){});
        e.submit(mk(2, Side::Buy, OrderType::IOC, 101, 100), [](const Fill&){});
        Price p = 0;
        CHECK(!b.best_bid(p));            // nothing rested
        CHECK(!b.best_ask(p));            // the 30 got consumed
    }

    // FOK rejects when full size is not available
    {
        OrderBook b; MatchingEngine e(b);
        e.submit(mk(1, Side::Sell, OrderType::Limit, 101, 30), [](const Fill&){});
        int fills = 0;
        e.submit(mk(2, Side::Buy, OrderType::FOK, 101, 100), [&](const Fill&){ ++fills; });
        CHECK_EQ(fills, 0);              // rejected, no partial fills
        CHECK_EQ(b.ask_size(101), 30);  // resting liquidity untouched
    }

    // FOK fills fully when enough size exists
    {
        OrderBook b; MatchingEngine e(b);
        e.submit(mk(1, Side::Sell, OrderType::Limit, 101, 60), [](const Fill&){});
        e.submit(mk(2, Side::Sell, OrderType::Limit, 102, 60), [](const Fill&){});
        Quantity taker = 0;
        e.submit(mk(3, Side::Buy, OrderType::FOK, 102, 100), [&](const Fill& f){
            if (!f.is_maker) taker += f.qty;
        });
        CHECK_EQ(taker, 100);
        CHECK_EQ(b.ask_size(102), 20);  // 20 left at 102
    }

    // Market order sweeps all reachable liquidity, discards remainder
    {
        OrderBook b; MatchingEngine e(b);
        e.submit(mk(1, Side::Sell, OrderType::Limit, 101, 20), [](const Fill&){});
        e.submit(mk(2, Side::Sell, OrderType::Limit, 103, 20), [](const Fill&){});
        Quantity taker = 0;
        e.submit(mk(3, Side::Buy, OrderType::Market, 0, 100), [&](const Fill& f){
            if (!f.is_maker) taker += f.qty;
        });
        CHECK_EQ(taker, 40);            // only 40 available
        CHECK(b.empty());
    }

    test::report_and_reset("matching_engine");
    return test::failures() == 0 ? 0 : 1;
}
