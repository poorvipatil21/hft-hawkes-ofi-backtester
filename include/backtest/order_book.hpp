#pragma once
#include "order.hpp"
#include <map>
#include <list>
#include <unordered_map>
#include <algorithm>

namespace bt {

// A resting order living in a price level's FIFO queue.
struct RestingOrder {
    OrderId  id;
    Quantity qty;
    bool     is_ours;
};

struct PriceLevel {
    Quantity                total_qty = 0;
    std::list<RestingOrder> queue;          // front == oldest (time priority)
};

// Single-instrument limit order book with strict price-time priority.
//
//   * Bids sorted high -> low  (best bid  == begin()).
//   * Asks sorted low  -> high (best ask  == begin()).
//   * O(1) cancel via an id -> (side, price, list-iterator) index.
//   * Aggressive orders are matched in place, consuming resting liquidity FIFO.
//
// std::list nodes give stable iterators, so cancelling an order that sits
// behind others in the queue never invalidates its neighbours' handles.
class OrderBook {
public:
    using BidMap = std::map<Price, PriceLevel, std::greater<Price>>;
    using AskMap = std::map<Price, PriceLevel, std::less<Price>>;

    struct Locator {
        Side side;
        Price price;
        std::list<RestingOrder>::iterator it;
    };

    bool empty() const { return bids_.empty() && asks_.empty(); }

    bool best_bid(Price& p) const { if (bids_.empty()) return false; p = bids_.begin()->first; return true; }
    bool best_ask(Price& p) const { if (asks_.empty()) return false; p = asks_.begin()->first; return true; }

    Quantity bid_size(Price p) const { auto it = bids_.find(p); return it == bids_.end() ? 0 : it->second.total_qty; }
    Quantity ask_size(Price p) const { auto it = asks_.find(p); return it == asks_.end() ? 0 : it->second.total_qty; }

    const BidMap& bids() const { return bids_; }
    const AskMap& asks() const { return asks_; }

    std::size_t resting_count() const { return index_.size(); }

    // Rest a (non-marketable) order. Returns false if the id already exists.
    bool insert(const Order& o) {
        if (index_.count(o.id)) return false;
        if (o.side == Side::Buy) insert_into(bids_, o);
        else                     insert_into(asks_, o);
        return true;
    }

    // Cancel a resting order by id. O(1) via the index.
    bool cancel(OrderId id) {
        auto it = index_.find(id);
        if (it == index_.end()) return false;
        Locator loc = it->second;
        if (loc.side == Side::Buy) erase_locator(bids_, loc);
        else                       erase_locator(asks_, loc);
        index_.erase(it);
        return true;
    }

    // Match an aggressive order against the opposite side of the book.
    // Fills are delivered through `on_fill` (both taker and maker legs).
    // The book maintains its own invariants; the *remainder* is left in `in`.
    template <class OnFill>
    void match(Order& in, OnFill&& on_fill) {
        if (in.side == Side::Buy) match_against(in, asks_, on_fill, true);
        else                      match_against(in, bids_, on_fill, false);
    }

private:
    template <class Map>
    void insert_into(Map& m, const Order& o) {
        PriceLevel& lvl = m[o.price];
        lvl.total_qty += o.remaining;
        lvl.queue.push_back(RestingOrder{o.id, o.remaining, o.is_ours});
        index_[o.id] = Locator{o.side, o.price, std::prev(lvl.queue.end())};
    }

    template <class Map>
    void erase_locator(Map& m, Locator& loc) {
        auto mit = m.find(loc.price);
        if (mit == m.end()) return;
        mit->second.total_qty -= loc.it->qty;
        mit->second.queue.erase(loc.it);
        if (mit->second.queue.empty()) m.erase(mit);
    }

    template <class Map, class OnFill>
    void match_against(Order& in, Map& opp, OnFill& on_fill, bool incoming_is_buy) {
        while (in.remaining > 0 && !opp.empty()) {
            auto  lvl_it     = opp.begin();
            Price lvl_price  = lvl_it->first;
            bool  crosses    = (in.type == OrderType::Market) ||
                               (incoming_is_buy ? in.price >= lvl_price
                                                : in.price <= lvl_price);
            if (!crosses) break;

            auto& q = lvl_it->second.queue;
            while (in.remaining > 0 && !q.empty()) {
                RestingOrder& r      = q.front();
                Quantity      traded = std::min(in.remaining, r.qty);

                in.on_fill(traded);
                r.qty                    -= traded;
                lvl_it->second.total_qty -= traded;

                // Execution occurs at the resting (maker) price -> price-time priority.
                on_fill(Fill{in.id, in.side,           lvl_price, traded, in.ts_active, in.is_ours, false});
                on_fill(Fill{r.id,  opposite(in.side), lvl_price, traded, in.ts_active, r.is_ours,  true });

                if (r.qty <= 0) {
                    index_.erase(r.id);
                    q.pop_front();
                }
            }
            if (q.empty()) opp.erase(lvl_it);
        }
    }

    BidMap bids_;
    AskMap asks_;
    std::unordered_map<OrderId, Locator> index_;
};

} // namespace bt
