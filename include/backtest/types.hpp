#pragma once
#include <cstdint>
#include <cmath>

namespace bt {

// ---- Fundamental integer types -------------------------------------------
// Prices are stored as integer *ticks* to keep matching arithmetic exact.
using Price     = std::int64_t;   // price in ticks (1 tick == TickConfig::tick_size dollars)
using Quantity  = std::int64_t;   // shares / contracts
using OrderId   = std::uint64_t;  // globally unique order identifier
using Timestamp = std::int64_t;   // nanoseconds on the simulation clock
using SeqNum    = std::uint64_t;  // monotonic tie-breaker for deterministic ordering

// ---- Enums ----------------------------------------------------------------
enum class Side : std::uint8_t { Buy = 0, Sell = 1 };

inline Side opposite(Side s) { return s == Side::Buy ? Side::Sell : Side::Buy; }
inline int  sign(Side s)     { return s == Side::Buy ? +1 : -1; }

enum class OrderType : std::uint8_t {
    Limit,   // rest remainder in the book
    Market,  // take whatever liquidity exists, discard remainder
    IOC,     // immediate-or-cancel: match now, cancel remainder
    FOK      // fill-or-kill: fill fully now or reject entirely
};

enum class OrderStatus : std::uint8_t {
    New,             // created by strategy, not yet accepted
    Accepted,        // acknowledged, resting or in-flight
    PartiallyFilled, // some quantity executed
    Filled,          // fully executed
    Canceled,        // withdrawn or IOC/Market remainder discarded
    Rejected         // FOK failure / risk reject
};

// ---- Tick <-> dollar conversion ------------------------------------------
struct TickConfig {
    double tick_size = 0.01;                     // dollar value of one tick
    Price  to_ticks(double px) const  { return static_cast<Price>(std::llround(px / tick_size)); }
    double to_dollars(Price p) const  { return static_cast<double>(p) * tick_size; }
};

// ---- Stringification (diagnostics / CSV) ----------------------------------
inline const char* to_string(OrderStatus s) {
    switch (s) {
        case OrderStatus::New:             return "NEW";
        case OrderStatus::Accepted:        return "ACCEPTED";
        case OrderStatus::PartiallyFilled: return "PARTIALLY_FILLED";
        case OrderStatus::Filled:          return "FILLED";
        case OrderStatus::Canceled:        return "CANCELED";
        case OrderStatus::Rejected:        return "REJECTED";
    }
    return "?";
}

inline const char* to_string(Side s)  { return s == Side::Buy ? "BUY" : "SELL"; }

inline const char* to_string(OrderType t) {
    switch (t) {
        case OrderType::Limit:  return "LIMIT";
        case OrderType::Market: return "MARKET";
        case OrderType::IOC:    return "IOC";
        case OrderType::FOK:    return "FOK";
    }
    return "?";
}

} // namespace bt
