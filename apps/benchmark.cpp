// benchmark — measures sustained order-event throughput.
//
// Two figures are reported:
//   1. Matching-engine core: raw Add/Cancel/Trade events driven straight
//      through the LOB with no strategy overhead (the "1M+ events/sec" claim).
//   2. Full engine: the same tape with a live market-making strategy attached,
//      i.e. end-to-end throughput including quoting, latency scheduling, and
//      portfolio accounting.
//
//   ./benchmark [events]     (default 5000000)
#include "backtest/order_book.hpp"
#include "backtest/matching_engine.hpp"
#include "backtest/backtester.hpp"
#include "market_maker.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace bt;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 5'000'000;

    std::printf("Generating %zu market events...\n", n);
    auto feed = generate_synthetic_feed(n);

    // ---- 1. Core matching engine throughput --------------------------------
    {
        OrderBook      book;
        MatchingEngine eng(book);
        std::size_t    fills = 0;
        auto sink = [&](const Fill&) { ++fills; };

        auto t0 = Clock::now();
        for (const auto& me : feed) {
            switch (me.type) {
                case MdType::Add: {
                    Order o; o.id = me.id; o.side = me.side; o.type = OrderType::Limit;
                    o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
                    eng.submit(o, sink); break;
                }
                case MdType::Cancel:
                    eng.cancel(me.id); break;
                case MdType::Trade: {
                    Order o; o.id = me.id; o.side = me.side; o.type = OrderType::IOC;
                    o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
                    eng.submit(o, sink); break;
                }
            }
        }
        auto t1 = Clock::now();
        double sec = secs(t0, t1);
        std::printf("\n---- Matching-engine core ----\n");
        std::printf("  events      : %zu\n", n);
        std::printf("  fills       : %zu\n", fills);
        std::printf("  resting     : %zu\n", book.resting_count());
        std::printf("  wall time   : %.4f s\n", sec);
        std::printf("  THROUGHPUT  : %.2f M events/sec\n", n / sec / 1e6);
    }

    // ---- 2. Full engine (strategy attached) --------------------------------
    {
        Backtester::Config cfg;
        Backtester engine(cfg);
        MarketMaker mm(50, 1, 500);
        engine.set_strategy(&mm);

        auto t0 = Clock::now();
        engine.run(feed);
        auto t1 = Clock::now();
        double sec = secs(t0, t1);
        std::printf("\n---- Full engine (market maker) ----\n");
        std::printf("  events proc : %zu\n", engine.processed());
        std::printf("  strat fills : %zu\n", engine.portfolio().fills());
        std::printf("  wall time   : %.4f s\n", sec);
        std::printf("  THROUGHPUT  : %.2f M events/sec\n", engine.processed() / sec / 1e6);
    }
    return 0;
}
