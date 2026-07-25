#pragma once
#include "order_book.hpp"
#include "matching_engine.hpp"
#include "friction.hpp"
#include "portfolio.hpp"
#include "strategy.hpp"
#include "data_feed.hpp"
#include <queue>
#include <vector>
#include <limits>

namespace bt {

// Event-driven backtester.
//
// Two time-ordered streams are merged on a single simulation clock:
//   1. the market tape (MarketEvent's, applied at their own timestamp), and
//   2. the strategy's own orders, each scheduled at submit-time + latency.
//
// Before applying each market event the engine drains every pending strategy
// order whose arrival time has passed, guaranteeing correct causal ordering:
// a decision made at time T cannot affect the book until T + latency.
class Backtester : public OrderContext {
public:
    struct Config {
        double        starting_cash   = 1'000'000.0;
        double        tick_size       = 0.01;
        double        qty_scale       = 1.0;       // see Portfolio's qty_scale doc: 1.0 = Quantity units are whole shares/coins
        Timestamp     latency_base    = 50'000;    // 50 us
        Timestamp     latency_jitter  = 20'000;    // +/- 20 us
        double        slippage_impact = 0.5;       // ticks per 100% participation
        std::uint64_t seed            = 12345;
        bool          enable_friction = true;      // toggle latency + slippage off for a naive run
        std::size_t   mark_every      = 32;        // equity-curve sampling stride (in events)
    };

    explicit Backtester(Config cfg)
        : cfg_(cfg),
          engine_(book_),
          portfolio_(cfg.starting_cash, cfg.tick_size, cfg.qty_scale),
          latency_(cfg.latency_base, cfg.latency_jitter, cfg.seed),
          slippage_(cfg.slippage_impact) {}

    void set_strategy(Strategy* s) { strat_ = s; }

    Portfolio&       portfolio()       { return portfolio_; }
    const Portfolio& portfolio() const { return portfolio_; }
    std::size_t      processed()  const { return event_count_; }
    double           tick_size()  const { return cfg_.tick_size; }

    void run(const std::vector<MarketEvent>& feed) {
        if (strat_) strat_->on_start(*this);

        for (const auto& me : feed) {
            clock_ = me.ts;
            drain_pending(me.ts);           // strategy orders that have now "arrived"
            apply_market_event(me);
            ++event_count_;
            ++md_seen_;

            if (strat_) strat_->on_raw_event(me, *this);

            if (strat_) {
                MarketSnapshot s = snapshot(me.ts);
                strat_->on_market_data(s, *this);
            }
            if (md_seen_ % cfg_.mark_every == 0) mark_now(me.ts);
        }

        drain_pending(kMaxTs);              // flush anything still in flight
        mark_now(clock_);
        if (strat_) strat_->on_stop(*this);
    }

    // ---- OrderContext ------------------------------------------------------
    OrderId submit(Side side, OrderType type, Price price, Quantity qty) override {
        OrderId id = next_our_id_++;
        Order o;
        o.id = id; o.side = side; o.type = type; o.price = price;
        o.qty = qty; o.remaining = qty; o.is_ours = true; o.ts_submit = clock_;
        o.ts_active = clock_ + (cfg_.enable_friction ? latency_.sample() : 0);
        pending_.push(Pending{o.ts_active, seq_++, o, false, 0});
        return id;
    }
    void cancel(OrderId id) override {
        Timestamp arrive = clock_ + (cfg_.enable_friction ? latency_.sample() : 0);
        pending_.push(Pending{arrive, seq_++, Order{}, true, id});
    }
    Timestamp        now()  const override { return clock_; }
    const OrderBook& book() const override { return book_; }

private:
    static constexpr Timestamp kMaxTs = std::numeric_limits<Timestamp>::max();

    struct Pending {
        Timestamp ts; SeqNum seq; Order order; bool is_cancel; OrderId cancel_id;
    };
    struct PendCmp {
        bool operator()(const Pending& a, const Pending& b) const {
            return a.ts != b.ts ? a.ts > b.ts : a.seq > b.seq;   // min-heap on (ts, seq)
        }
    };

    void drain_pending(Timestamp upto) {
        while (!pending_.empty() && pending_.top().ts <= upto) {
            Pending p = pending_.top(); pending_.pop();
            Timestamp saved = clock_;
            clock_ = p.ts;
            if (p.is_cancel) engine_.cancel(p.cancel_id);
            else             apply_our_order(p.order);
            clock_ = saved;
            ++event_count_;
        }
    }

    void apply_our_order(Order o) {
        Quantity visible = opposite_visible(o.side);   // for slippage participation
        engine_.submit(o, [&](const Fill& f) {
            Fill adj = f;
            if (cfg_.enable_friction && f.is_ours && !f.is_maker) {
                Price extra = slippage_.extra_ticks(o.qty, visible);
                adj.price  += (o.side == Side::Buy ? +extra : -extra);   // always adverse
            }
            route_fill(adj);
        });
    }

    void apply_market_event(const MarketEvent& me) {
        switch (me.type) {
            case MdType::Add: {
                Order o;
                o.id = me.id; o.side = me.side; o.type = OrderType::Limit;
                o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
                o.is_ours = false; o.ts_active = me.ts;
                engine_.submit(o, [&](const Fill& f) { route_fill(f); });
                break;
            }
            case MdType::Cancel:
                engine_.cancel(me.id);
                break;
            case MdType::Trade: {
                Order o;
                o.id = me.id; o.side = me.side; o.type = OrderType::IOC;
                o.price = me.price; o.qty = me.qty; o.remaining = me.qty;
                o.is_ours = false; o.ts_active = me.ts;
                engine_.submit(o, [&](const Fill& f) { route_fill(f); });
                break;
            }
        }
    }

    void route_fill(const Fill& f) {
        if (!f.is_ours) return;
        portfolio_.on_fill(f, current_mid());
        if (strat_) strat_->on_fill(f, *this);
    }

    double current_mid() const {
        Price bb, ba; bool hb = book_.best_bid(bb), ha = book_.best_ask(ba);
        return (hb && ha) ? (bb + ba) / 2.0 * cfg_.tick_size
             : hb        ? bb * cfg_.tick_size
             : ha        ? ba * cfg_.tick_size : 0.0;
    }

    Quantity opposite_visible(Side s) const {
        if (s == Side::Buy) { auto it = book_.asks().begin(); return it != book_.asks().end() ? it->second.total_qty : 0; }
        else                { auto it = book_.bids().begin(); return it != book_.bids().end() ? it->second.total_qty : 0; }
    }

    void mark_now(Timestamp ts) {
        double mid = current_mid();
        if (mid > 0) portfolio_.mark(ts, mid);
    }

    MarketSnapshot snapshot(Timestamp ts) const {
        MarketSnapshot s; s.ts = ts;
        Price bb, ba;
        s.has_bid = book_.best_bid(bb);
        s.has_ask = book_.best_ask(ba);
        s.best_bid = s.has_bid ? bb : 0;
        s.best_ask = s.has_ask ? ba : 0;
        s.bid_size = s.has_bid ? book_.bid_size(bb) : 0;
        s.ask_size = s.has_ask ? book_.ask_size(ba) : 0;
        return s;
    }

    Config          cfg_;
    OrderBook       book_;
    MatchingEngine  engine_;
    Portfolio       portfolio_;
    LatencyModel    latency_;
    SlippageModel   slippage_;
    Strategy*       strat_ = nullptr;

    Timestamp   clock_        = 0;
    OrderId     next_our_id_  = 1;
    SeqNum      seq_          = 0;
    std::size_t event_count_  = 0;
    std::size_t md_seen_      = 0;
    std::priority_queue<Pending, std::vector<Pending>, PendCmp> pending_;
};

} // namespace bt
