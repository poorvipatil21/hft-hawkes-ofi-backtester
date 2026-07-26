#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

namespace bt {

// Multivariate Hawkes process, sum-of-exponentials kernel (P components per
// (i,j) pair, decays fixed as a hyperparameter):
//
//   lambda_i(t) = mu_i + sum_j sum_p alpha_ijp * sum_{t_k^j < t} exp(-beta_ijp (t - t_k^j))
//
// A single exponential (P=1) assumes clustering decays at one timescale;
// real order-flow clustering typically has fast (sub-second) and slower
// components, so a mixture of exponentials at different fixed decays is the
// standard way to approximate that (Bacry & Muzy 2014-style multi-scale
// Hawkes / a discretized power-law kernel) without losing the closed-form
// compensator an exponential kernel gives you. Decays are still fixed
// hyperparameters; only mu and each component's alpha are MLE-fit.
//
// Recursive intensity per component (Ozaki 1979), independently for each p:
//   R_ijp(t_n) = exp(-beta_ijp * (t_n - t_{n-1})) * (R_ijp(t_{n-1}) + [mark_{n-1} == j])

struct HawkesEvent {
    double      t    = 0.0;
    std::size_t mark = 0;   // dimension index in [0, D)
};

// Shifts a sequence of events so the first event is at t=0, returning the
// elapsed duration (the corrected T for the compensator). This is the
// actual fix for the compensator time-scaling bug documented in the
// project's writeup: real-data timestamps are raw Unix epoch values
// (~1e18 ns), and using the last one directly as the compensator's
// integration bound inflates the background-rate penalty by orders of
// magnitude. Centralizing this here (rather than duplicating the shift
// inline in each calling app, as an earlier version did) makes it directly
// unit-testable -- see test_hawkes.cpp's regression test, which feeds
// deliberately huge-epoch-style timestamps and asserts the recovered fit
// is identical to the same events shifted to start near zero, i.e. that
// the fit is invariant to an arbitrary additive offset in the input
// timestamps, which it must be for T to be correctly computed as elapsed
// duration rather than absolute time.
inline double shift_to_relative_time(std::vector<HawkesEvent>& events) {
    if (events.empty()) return 0.0;
    double t0 = events.front().t;
    for (auto& e : events) e.t -= t0;
    return events.back().t;
}

class MultivariateHawkes {
public:
    // beta[i][j] is a vector of P fixed decay rates for that (i,j) pair; all
    // (i,j) pairs must specify the same P.
    MultivariateHawkes(std::size_t dim, std::vector<std::vector<std::vector<double>>> beta)
        : D_(dim), beta_(std::move(beta)) {
        if (beta_.size() != D_) throw std::invalid_argument("beta dim mismatch");
        P_ = beta_.empty() || beta_[0].empty() ? 1 : beta_[0][0].size();
        for (auto& row : beta_) {
            if (row.size() != D_) throw std::invalid_argument("beta dim mismatch");
            for (auto& v : row) if (v.size() != P_) throw std::invalid_argument("beta component-count mismatch");
        }
        mu_.assign(D_, 0.05);
        alpha_.assign(D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.05)));
    }

    std::size_t dim() const { return D_; }
    std::size_t n_components() const { return P_; }
    std::vector<double>&       mu()    { return mu_; }
    std::vector<std::vector<std::vector<double>>>& alpha() { return alpha_; }
    const std::vector<double>&       mu()    const { return mu_; }
    const std::vector<std::vector<std::vector<double>>>& alpha() const { return alpha_; }
    const std::vector<std::vector<std::vector<double>>>& beta()  const { return beta_; }

    // Branching ratio matrix n_ij = sum_p alpha_ijp / beta_ijp; spectral
    // radius < 1 is the stability (non-explosive) condition for the process.
    std::vector<std::vector<double>> branching_ratio() const {
        std::vector<std::vector<double>> n(D_, std::vector<double>(D_, 0.0));
        for (std::size_t i = 0; i < D_; ++i)
            for (std::size_t j = 0; j < D_; ++j)
                for (std::size_t p = 0; p < P_; ++p)
                    n[i][j] += beta_[i][j][p] > 1e-12 ? alpha_[i][j][p] / beta_[i][j][p] : 0.0;
        return n;
    }

    double spectral_radius(int iters = 100) const {
        auto n = branching_ratio();
        std::vector<double> v(D_, 1.0);
        for (int it = 0; it < iters; ++it) {
            std::vector<double> nv(D_, 0.0);
            for (std::size_t i = 0; i < D_; ++i)
                for (std::size_t j = 0; j < D_; ++j)
                    nv[i] += n[i][j] * v[j];
            double norm = 0.0; for (double x : nv) norm += x * x; norm = std::sqrt(norm);
            if (norm < 1e-12) return 0.0;
            for (auto& x : nv) x /= norm;
            v = nv;
        }
        std::vector<double> nv(D_, 0.0);
        for (std::size_t i = 0; i < D_; ++i)
            for (std::size_t j = 0; j < D_; ++j)
                nv[i] += n[i][j] * v[j];
        double num = 0.0; for (std::size_t i = 0; i < D_; ++i) num += nv[i] * v[i];
        return num;
    }

    double intensity_naive(std::size_t i, double t, const std::vector<HawkesEvent>& hist) const {
        double lam = mu_[i];
        for (const auto& e : hist) {
            if (e.t >= t) break;
            for (std::size_t p = 0; p < P_; ++p)
                lam += alpha_[i][e.mark][p] * std::exp(-beta_[i][e.mark][p] * (t - e.t));
        }
        return lam;
    }

    struct Gradients {
        std::vector<double>                           d_mu;
        std::vector<std::vector<std::vector<double>>> d_alpha;
        double                                          ll = 0.0;
    };

    Gradients log_likelihood_and_grad(const std::vector<HawkesEvent>& events, double T) const {
        Gradients g;
        g.d_mu.assign(D_, 0.0);
        g.d_alpha.assign(D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.0)));

        std::vector<std::vector<std::vector<double>>> R(
            D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.0)));
        bool any_prev = false;
        double prev_t = 0.0;

        for (const auto& e : events) {
            if (any_prev) {
                double dt = e.t - prev_t;
                for (std::size_t i = 0; i < D_; ++i)
                    for (std::size_t j = 0; j < D_; ++j)
                        for (std::size_t p = 0; p < P_; ++p)
                            R[i][j][p] *= std::exp(-beta_[i][j][p] * dt);
            }
            std::size_t i = e.mark;
            double lam_i = mu_[i];
            for (std::size_t j = 0; j < D_; ++j)
                for (std::size_t p = 0; p < P_; ++p)
                    lam_i += alpha_[i][j][p] * R[i][j][p];
            lam_i = std::max(lam_i, 1e-12);

            g.ll += std::log(lam_i);
            g.d_mu[i] += 1.0 / lam_i;
            for (std::size_t j = 0; j < D_; ++j)
                for (std::size_t p = 0; p < P_; ++p)
                    g.d_alpha[i][j][p] += R[i][j][p] / lam_i;

            for (std::size_t i2 = 0; i2 < D_; ++i2)
                for (std::size_t p = 0; p < P_; ++p)
                    R[i2][e.mark][p] += 1.0;

            prev_t = e.t;
            any_prev = true;
        }

        for (std::size_t i = 0; i < D_; ++i) {
            g.ll -= mu_[i] * T;
            g.d_mu[i] -= T;
        }
        for (const auto& e : events) {
            std::size_t j = e.mark;
            double rem = T - e.t;
            for (std::size_t i = 0; i < D_; ++i) {
                for (std::size_t p = 0; p < P_; ++p) {
                    double b = beta_[i][j][p];
                    double decay_term = b > 1e-12 ? (1.0 - std::exp(-b * rem)) / b : rem;
                    g.ll            -= alpha_[i][j][p] * decay_term;
                    g.d_alpha[i][j][p] -= decay_term;
                }
            }
        }
        return g;
    }

    struct FitResult {
        double      final_ll        = 0.0;
        int         iterations      = 0;
        double      spectral_radius = 0.0;
        bool        converged       = false;
    };

    // Adam-based projected MLE ascent in log-space (see rationale above the
    // class). max_iters/lr/tol are exposed so the CLI can iterate on
    // convergence without a rebuild.
    //
    // mu_reg_strength adds a weak ridge penalty anchoring mu toward its
    // count/T estimate (the MLE-at-alpha=0 baseline). On long sessions the
    // slowest kernel component can act almost like a constant offset over
    // the observation window, becoming nearly interchangeable with mu in
    // the likelihood -- without this anchor, the optimizer can drive mu to
    // a numerically meaningless near-zero while the slow alpha component
    // silently absorbs its role, rather than mu genuinely being near zero.
    // This is standard MAP-style regularization for exactly this kind of
    // near-collinearity; it doesn't materially bias the branching-ratio
    // estimate (mu and alpha play different roles there), just keeps mu
    // itself interpretable.
    FitResult fit_mle(const std::vector<HawkesEvent>& events, double T,
                       int max_iters = 500, double lr = 0.1, double tol = 1e-7,
                       double mu_reg_strength = 0.5) {
        FitResult res;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;

        std::vector<double> counts(D_, 0.0);
        for (const auto& e : events) counts[e.mark] += 1.0;
        std::vector<double> u(D_), m_u(D_, 0.0), v_u(D_, 0.0);
        std::vector<double> u_anchor(D_);
        for (std::size_t i = 0; i < D_; ++i) {
            u[i] = std::log(std::max(1e-12, counts[i] / std::max(T, 1e-9)));
            u_anchor[i] = u[i];   // fixed reference point for the ridge penalty
        }

        std::vector<std::vector<std::vector<double>>> w(
            D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_))),
            m_w(D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.0))),
            v_w(D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.0)));
        for (auto& row : w) for (auto& col : row) for (auto& x : col) x = std::log(1e-4);

        auto sync_params = [&]() {
            for (std::size_t i = 0; i < D_; ++i) mu_[i] = std::exp(u[i]);
            for (std::size_t i = 0; i < D_; ++i)
                for (std::size_t j = 0; j < D_; ++j)
                    for (std::size_t p = 0; p < P_; ++p)
                        alpha_[i][j][p] = std::exp(w[i][j][p]);
        };
        sync_params();

        double prev_ll = -1e300;
        for (int it = 1; it <= max_iters; ++it) {
            auto g = log_likelihood_and_grad(events, T);

            for (std::size_t i = 0; i < D_; ++i) {
                double grad_u = g.d_mu[i] * mu_[i];
                m_u[i] = b1 * m_u[i] + (1 - b1) * grad_u;
                v_u[i] = b2 * v_u[i] + (1 - b2) * grad_u * grad_u;
                double mhat = m_u[i] / (1 - std::pow(b1, it));
                double vhat = v_u[i] / (1 - std::pow(b2, it));
                u[i] += lr * mhat / (std::sqrt(vhat) + eps);
                // Decoupled weight decay (AdamW-style), applied directly to
                // the parameter rather than mixed into the likelihood
                // gradient: L2-in-gradient gets swamped by Adam's own
                // per-parameter normalization (the adaptive step size
                // rescales away a small constant penalty term almost
                // entirely), so it barely pulled mu toward its anchor in
                // practice. A direct post-step shrinkage toward u_anchor
                // actually has the intended effect regardless of the
                // likelihood gradient's scale.
                u[i] -= lr * mu_reg_strength * (u[i] - u_anchor[i]);
                u[i] = std::clamp(u[i], -60.0, 20.0);
            }
            for (std::size_t i = 0; i < D_; ++i) {
                for (std::size_t j = 0; j < D_; ++j) {
                    for (std::size_t p = 0; p < P_; ++p) {
                        double grad_w = g.d_alpha[i][j][p] * alpha_[i][j][p];
                        m_w[i][j][p] = b1 * m_w[i][j][p] + (1 - b1) * grad_w;
                        v_w[i][j][p] = b2 * v_w[i][j][p] + (1 - b2) * grad_w * grad_w;
                        double mhat = m_w[i][j][p] / (1 - std::pow(b1, it));
                        double vhat = v_w[i][j][p] / (1 - std::pow(b2, it));
                        w[i][j][p] += lr * mhat / (std::sqrt(vhat) + eps);
                        w[i][j][p] = std::clamp(w[i][j][p], -60.0, 20.0);
                    }
                }
            }
            sync_params();

            double sr = spectral_radius();
            if (sr >= 0.98) {
                double log_scale = std::log(0.95 / sr);
                for (auto& row : w) for (auto& col : row) for (auto& x : col) x += log_scale;
                sync_params();
            }

            res.iterations = it;
            if (std::abs(g.ll - prev_ll) < tol * std::max(1.0, std::abs(prev_ll))) {
                res.converged = true;
                prev_ll = g.ll;
                break;
            }
            prev_ll = g.ll;
        }
        res.final_ll = prev_ll;
        res.spectral_radius = spectral_radius();
        return res;
    }

    // Goodness-of-fit via the time-rescaling theorem: rescaled inter-event
    // "residuals" per mark should be ~Exp(1) (mean 1, var 1) if the fit is good.
    std::vector<std::vector<double>> compensator_residuals(const std::vector<HawkesEvent>& events) const {
        std::vector<std::vector<double>> resid(D_);
        std::vector<bool> seen(D_, false);

        std::vector<std::vector<std::vector<double>>> R(
            D_, std::vector<std::vector<double>>(D_, std::vector<double>(P_, 0.0)));
        double prev_t = 0.0;
        bool any_prev = false;
        std::vector<double> running_compensator(D_, 0.0);

        for (const auto& e : events) {
            double dt = any_prev ? e.t - prev_t : 0.0;
            for (std::size_t i = 0; i < D_; ++i) {
                double contrib = mu_[i] * dt;
                for (std::size_t j = 0; j < D_; ++j) {
                    for (std::size_t p = 0; p < P_; ++p) {
                        double b = beta_[i][j][p];
                        double r = R[i][j][p];
                        contrib += b > 1e-12 ? (alpha_[i][j][p] * r / b) * (1.0 - std::exp(-b * dt))
                                              : alpha_[i][j][p] * r * dt;
                    }
                }
                running_compensator[i] += contrib;
            }
            if (any_prev) {
                for (std::size_t i = 0; i < D_; ++i)
                    for (std::size_t j = 0; j < D_; ++j)
                        for (std::size_t p = 0; p < P_; ++p)
                            R[i][j][p] *= std::exp(-beta_[i][j][p] * dt);
            }
            std::size_t m = e.mark;
            if (seen[m]) resid[m].push_back(running_compensator[m]);
            running_compensator[m] = 0.0;
            seen[m] = true;

            for (std::size_t i2 = 0; i2 < D_; ++i2)
                for (std::size_t p = 0; p < P_; ++p)
                    R[i2][m][p] += 1.0;
            prev_t = e.t;
            any_prev = true;
        }
        return resid;
    }

private:
    std::size_t D_, P_;
    std::vector<double>                           mu_;
    std::vector<std::vector<std::vector<double>>> alpha_;
    std::vector<std::vector<std::vector<double>>> beta_;
};

} // namespace bt
