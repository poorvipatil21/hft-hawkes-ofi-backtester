#pragma once
#include "types.hpp"
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <fstream>
#include <sstream>

namespace bt {

// A market data event replayed onto the book. Background liquidity and other
// participants' aggression are both expressed here:
//   Add    -> a non-strategy limit order joins the book
//   Cancel -> a previously added liquidity order is withdrawn
//   Trade  -> an external aggressor crosses the spread (may hit our quotes)
enum class MdType : std::uint8_t { Add, Cancel, Trade };

struct MarketEvent {
    Timestamp ts    = 0;
    MdType    type  = MdType::Add;
    Side      side  = Side::Buy;   // resting side (Add/Cancel) or aggressor side (Trade)
    Price     price = 0;
    Quantity  qty   = 0;
    OrderId   id    = 0;
};

// Reserved id range for background market liquidity so it never collides with
// strategy order ids (which start at 1).
inline constexpr OrderId kMarketIdBase = OrderId(1) << 40;

// Generate a self-contained, reproducible L2 tape around a random-walk mid.
//
// Design notes:
//   * Background liquidity rests two or more ticks away from the mid, leaving
//     the touch (+/-1 tick) open for a market-making strategy to quote alone.
//   * Aggressive Trade events sweep inward through the touch, so a strategy
//     resting at the inside gets realistic maker fills and adverse selection.
//   * A bounded ring of live liquidity ids is cancelled oldest-first to keep
//     book memory flat over long runs.
inline std::vector<MarketEvent> generate_synthetic_feed(std::size_t n,
                                                        std::uint64_t seed = 7,
                                                        Price mid0 = 10'000) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double>       g(0.0, 1.0);

    std::vector<MarketEvent> ev;
    ev.reserve(n + 16);

    Price     mid  = mid0;
    Timestamp t    = 0;
    OrderId   next = kMarketIdBase;
    std::deque<OrderId> live;
    const std::size_t   max_live = 2000;
    const int           levels   = 5;
    const Quantity      base_sz  = 100;

    auto emit_add = [&](Side s, Price px, Quantity q) {
        ev.push_back(MarketEvent{t, MdType::Add, s, px, q, next});
        live.push_back(next++);
        if (live.size() > max_live) {
            OrderId old = live.front(); live.pop_front();
            ev.push_back(MarketEvent{t, MdType::Cancel, Side::Buy, 0, 0, old});
        }
    };

    while (ev.size() < n) {
        t += 1'000 + static_cast<Timestamp>(u(rng) * 1'000);   // ~1-2 us spacing

        if (u(rng) < 0.10) mid += (g(rng) > 0 ? 1 : -1);       // random-walk mid
        if (mid < 100) mid = 100;

        // Replenish depth from 2 ticks out (leaving the touch for a strategy).
        for (int i = 2; i <= levels + 1 && ev.size() < n; ++i) {
            if (u(rng) < 0.6) emit_add(Side::Buy,  mid - i, base_sz + static_cast<Quantity>(u(rng) * 100));
            if (u(rng) < 0.6) emit_add(Side::Sell, mid + i, base_sz + static_cast<Quantity>(u(rng) * 100));
        }

        // Aggressor sweeping through the touch.
        if (u(rng) < 0.25 && ev.size() < n) {
            Side s     = u(rng) < 0.5 ? Side::Buy : Side::Sell;
            Price px   = (s == Side::Buy) ? mid + (levels + 1) : mid - (levels + 1);
            Quantity q = base_sz / 2 + static_cast<Quantity>(u(rng) * 150);
            ev.push_back(MarketEvent{t, MdType::Trade, s, px, q, next++});
        }
    }

    ev.resize(n);
    return ev;
}

// Load a feed from CSV. Expected header + columns:
//   ts,type,side,price_ticks,qty,id
// where type in {add,cancel,trade} and side in {buy,sell}.
inline std::vector<MarketEvent> load_csv_feed(const std::string& path) {
    std::vector<MarketEvent> ev;
    std::ifstream in(path);
    if (!in) return ev;
    std::string line;
    std::getline(in, line);                 // skip header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string ts, type, side, px, qty, id;
        std::getline(ss, ts,   ',');
        std::getline(ss, type, ',');
        std::getline(ss, side, ',');
        std::getline(ss, px,   ',');
        std::getline(ss, qty,  ',');
        std::getline(ss, id,   ',');
        MarketEvent m;
        m.ts    = std::stoll(ts);
        m.type  = (type == "trade") ? MdType::Trade : (type == "cancel") ? MdType::Cancel : MdType::Add;
        m.side  = (side == "sell") ? Side::Sell : Side::Buy;
        m.price = px.empty()  ? 0 : std::stoll(px);
        m.qty   = qty.empty() ? 0 : std::stoll(qty);
        m.id    = id.empty()  ? 0 : std::stoull(id);
        ev.push_back(m);
    }
    return ev;
}

} // namespace bt
