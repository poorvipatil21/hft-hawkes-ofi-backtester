#pragma once
#include "types.hpp"

namespace bt {

// An order carries an explicit lifecycle:
//   New -> Accepted -> PartiallyFilled -> Filled
//                                    \--> Canceled
//   New -> Rejected
// Transitions are funnelled through the on_*() methods so the status field
// can never drift out of sync with the remaining quantity.
struct Order {
    OrderId     id        = 0;
    Side        side      = Side::Buy;
    OrderType   type      = OrderType::Limit;
    Price       price     = 0;              // limit price in ticks (ignored for Market)
    Quantity    qty       = 0;              // original quantity
    Quantity    remaining = 0;              // unfilled quantity
    OrderStatus status    = OrderStatus::New;
    Timestamp   ts_submit = 0;              // when the strategy created it
    Timestamp   ts_active = 0;              // when it reached the matching engine (post-latency)
    bool        is_ours   = false;          // strategy order vs. background market liquidity

    Quantity filled() const { return qty - remaining; }
    bool     live()   const { return status == OrderStatus::Accepted ||
                                     status == OrderStatus::PartiallyFilled; }

    // ---- state transitions --------------------------------------------------
    void on_accept()          { status = OrderStatus::Accepted; }
    void on_fill(Quantity q)  {
        remaining -= q;
        status = (remaining <= 0) ? OrderStatus::Filled : OrderStatus::PartiallyFilled;
    }
    void on_cancel()          { if (status != OrderStatus::Filled) status = OrderStatus::Canceled; }
    void on_reject()          { status = OrderStatus::Rejected; }
};

// A single execution. Both sides of every trade emit a Fill so the portfolio
// and any resting-order owner are notified.
struct Fill {
    OrderId   order_id = 0;
    Side      side     = Side::Buy;
    Price     price    = 0;      // execution price in ticks (maker's price)
    Quantity  qty      = 0;
    Timestamp ts       = 0;
    bool      is_ours  = false;  // did this fill belong to a strategy order?
    bool      is_maker = false;  // true = our resting order was hit; false = we took liquidity
};

} // namespace bt
