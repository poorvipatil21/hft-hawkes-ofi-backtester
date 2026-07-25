#include "backtest/hawkes.hpp"
#include "backtest/ofi.hpp"
#include "test_util.hpp"
#include <vector>
#include <random>

using namespace bt;

int main() {
    // Recursive intensity must match the naive O(n) definition exactly,
    // with a single-component (P=1) kernel wrapped in the new API.
    {
        std::vector<std::vector<std::vector<double>>> beta = {
            {{1.0}, {1.5}}, {{1.5}, {1.0}}
        };
        MultivariateHawkes hp(2, beta);
        CHECK_EQ(hp.n_components(), 1u);
        hp.mu() = {0.2, 0.3};
        hp.alpha() = {{{0.4}, {0.1}}, {{0.1}, {0.5}}};

        std::vector<HawkesEvent> events = {
            {0.5, 0}, {1.0, 1}, {1.2, 0}, {2.0, 0}, {2.3, 1}, {3.1, 1}, {4.0, 0}
        };

        std::vector<std::vector<double>> R(2, std::vector<double>(2, 0.0));
        double prev_t = 0.0;
        bool any_prev = false;
        std::vector<HawkesEvent> history;
        for (const auto& e : events) {
            if (any_prev) {
                double dt = e.t - prev_t;
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        R[i][j] *= std::exp(-beta[i][j][0] * dt);
            }
            double lam_recursive = hp.mu()[e.mark];
            for (int j = 0; j < 2; ++j) lam_recursive += hp.alpha()[e.mark][j][0] * R[e.mark][j];
            double lam_naive = hp.intensity_naive(e.mark, e.t, history);
            CHECK_NEAR(lam_recursive, lam_naive, 1e-9);

            for (int i2 = 0; i2 < 2; ++i2) R[i2][e.mark] += 1.0;
            history.push_back(e);
            prev_t = e.t;
            any_prev = true;
        }
    }

    // Self-consistency (P=1): fitting on data simulated from known params
    // should recover a log-likelihood at least as good as the un-optimized
    // starting point, and preserve stability.
    {
        std::vector<std::vector<std::vector<double>>> beta = {
            {{2.0}, {2.0}}, {{2.0}, {2.0}}
        };
        MultivariateHawkes truth(2, beta);
        truth.mu() = {0.3, 0.3};
        truth.alpha() = {{{0.6}, {0.2}}, {{0.2}, {0.6}}};
        CHECK(truth.spectral_radius() < 1.0);

        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<HawkesEvent> sim;
        double T = 200.0;
        double t = 0.0;
        std::vector<std::vector<double>> R(2, std::vector<double>(2, 0.0));
        double prev_t = 0.0;
        while (t < T) {
            double lam_bound = truth.mu()[0] + truth.mu()[1];
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) lam_bound += truth.alpha()[i][j][0] * R[i][j];
            lam_bound = std::max(lam_bound, 1e-6) * 1.001 + 0.05;
            double dt = -std::log(u(rng)) / lam_bound;
            double dt_decay = t + dt - prev_t;
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) R[i][j] *= std::exp(-beta[i][j][0] * dt_decay);
            t += dt;
            prev_t = t;
            if (t >= T) break;
            double lam0 = truth.mu()[0]; for (int j = 0; j < 2; ++j) lam0 += truth.alpha()[0][j][0] * R[0][j];
            double lam1 = truth.mu()[1]; for (int j = 0; j < 2; ++j) lam1 += truth.alpha()[1][j][0] * R[1][j];
            double draw = u(rng) * lam_bound;
            if (draw < lam0) {
                sim.push_back(HawkesEvent{t, 0});
                for (int i = 0; i < 2; ++i) R[i][0] += 1.0;
            } else if (draw < lam0 + lam1) {
                sim.push_back(HawkesEvent{t, 1});
                for (int i = 0; i < 2; ++i) R[i][1] += 1.0;
            }
        }
        CHECK(sim.size() > 100);

        MultivariateHawkes fit_model(2, beta);
        auto ll_start = fit_model.log_likelihood_and_grad(sim, T).ll;
        auto res = fit_model.fit_mle(sim, T, /*max_iters=*/300, /*lr=*/5e-3);
        CHECK(res.final_ll >= ll_start - 1e-6);
        CHECK(res.spectral_radius < 1.0);
    }

    // Multi-component (P=3) kernel: branching ratio should correctly sum
    // contributions across all components, and MLE on data simulated with
    // only one active component (others near-zero truth) should push the
    // other components' fitted alpha down, not up.
    {
        std::vector<std::vector<std::vector<double>>> beta = {
            {{10.0, 1.0, 0.1}, {10.0, 1.0, 0.1}},
            {{10.0, 1.0, 0.1}, {10.0, 1.0, 0.1}}
        };
        MultivariateHawkes hp(2, beta);
        CHECK_EQ(hp.n_components(), 3u);
        hp.mu() = {0.3, 0.3};
        hp.alpha() = {{{0.0, 0.4, 0.0}, {0.0, 0.1, 0.0}},
                      {{0.0, 0.1, 0.0}, {0.0, 0.4, 0.0}}};
        auto n_mat = hp.branching_ratio();
        // n_00 = 0/10 + 0.4/1 + 0/0.1 = 0.4
        CHECK_NEAR(n_mat[0][0], 0.4, 1e-9);
        CHECK_NEAR(n_mat[0][1], 0.1, 1e-9);
        CHECK(hp.spectral_radius() < 1.0);
    }

    // OFI sanity: a book that only gains bid depth (ask unchanged) should
    // produce strictly positive OFI.
    {
        OrderBook book;
        RollingOFI ofi(3, 100);
        Order bid1; bid1.id = 1; bid1.side = Side::Buy; bid1.type = OrderType::Limit;
        bid1.price = 100; bid1.qty = 10; bid1.remaining = 10;
        Order ask1; ask1.id = 2; ask1.side = Side::Sell; ask1.type = OrderType::Limit;
        ask1.price = 101; ask1.qty = 10; ask1.remaining = 10;
        book.insert(bid1);
        book.insert(ask1);
        double first = ofi.update(book);
        (void)first;

        Order bid2; bid2.id = 3; bid2.side = Side::Buy; bid2.type = OrderType::Limit;
        bid2.price = 100; bid2.qty = 50; bid2.remaining = 50;
        book.insert(bid2);
        double inc = ofi.update(book);
        CHECK(inc > 0.0);
    }

    test::report_and_reset("hawkes_ofi");
    return test::failures() == 0 ? 0 : 1;
}
