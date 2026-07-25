#pragma once
#include "order_book.hpp"
#include <functional>

namespace bt {

// Thin policy layer over OrderBook that applies order-type semantics:
//   * Limit  -> match, rest the remainder
//   * Market -> match, discard the remainder
//   * IOC    -> match, cancel the remainder
//   * FOK    -> reject unless the whole quantity can execute immediately
class MatchingEngine {
public:
    using FillCallback = std::function<void(const Fill&)>;

    explicit MatchingEngine(OrderBook& book) : book_(book) {}

    // Submit an order for immediate processing. Any executions are reported
    // through `on_fill` synchronously.
    void submit(Order o, const FillCallback& on_fill) {
        o.remaining = o.qty;
        o.on_accept();

        if (o.type == OrderType::FOK && !fully_fillable(o)) {
            o.on_reject();
            return;
        }

        book_.match(o, [&](const Fill& f) { on_fill(f); });

        if (o.remaining > 0) {
            if (o.type == OrderType::Limit) book_.insert(o);
            else                            o.on_cancel();   // Market / IOC / FOK remainder
        }
    }

    bool cancel(OrderId id) { return book_.cancel(id); }

private:
    // Walk the marketable depth to decide whether a FOK can fully execute.
    bool fully_fillable(const Order& o) const {
        Quantity avail = 0;
        if (o.side == Side::Buy) {
            for (const auto& [px, lvl] : book_.asks()) {
                if (o.type != OrderType::Market && px > o.price) break;
                avail += lvl.total_qty;
                if (avail >= o.qty) return true;
            }
        } else {
            for (const auto& [px, lvl] : book_.bids()) {
                if (o.type != OrderType::Market && px < o.price) break;
                avail += lvl.total_qty;
                if (avail >= o.qty) return true;
            }
        }
        return avail >= o.qty;
    }

    OrderBook& book_;
};

} // namespace bt
