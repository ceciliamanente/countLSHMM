// LS-HMM: latent space hidden Markov model for longitudinal multiplex networks,
// with hurdle negative binomial (HNB) emissions.

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

static const double LP_LIM = 50.0;

// Dispersion cap: r <= R_MAX, i.e. a = r^{-1/2} >= A_MIN.
static const double R_MAX = 1000.0;
static const double A_MIN = 1.0 / std::sqrt(R_MAX);

// Keeps log NB(0 | mu, r) strictly negative.
static const double L0_MAX = -1e-300;

// Adaptation of the proposal sds during burn-in.
static const double ACC_TARGET_BLOCK  = 0.30;
static const double ACC_TARGET_SCALAR = 0.44;
static const int    ADAPT_EVERY       = 50;


// ===== Random numbers =====
// splitmix64 + Marsaglia polar method, so that the same seed gives the same
// output whatever the number of threads.

struct RNG {
  uint64_t s;
  bool     has_spare;
  double   spare;
  
  explicit RNG(uint64_t seed = 0ULL) : s(seed), has_spare(false), spare(0.0) {}
  
  uint64_t next_u64() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  
  // uniform on (0, 1)
  double unif() {
    return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0)
    + (1.0 / 18014398509481984.0);
  }
  
  double norm() {
    if (has_spare) { has_spare = false; return spare; }
    double u, v, q;
    do {
      u = 2.0 * unif() - 1.0;
      v = 2.0 * unif() - 1.0;
      q = u * u + v * v;
    } while (q <= 0.0 || q >= 1.0);
    const double f = std::sqrt(-2.0 * std::log(q) / q);
    spare = v * f; has_spare = true;
    return u * f;
  }
  
  double rnorm(double sd) { return sd * norm(); }
};

uint64_t stream_seed(uint64_t master, uint64_t id) {
  uint64_t z = master + 0x9E3779B97F4A7C15ULL * (id + 1ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Metropolis-Hastings acceptance step.
bool accept(double log_acc, RNG& rng, long long& n_nonfinite) {
  if (std::isnan(log_acc)) { ++n_nonfinite; return false; }
  return log_acc >= 0.0 || std::log(rng.unif()) < log_acc;
}

// lgamma() is not thread-safe on glibc; lgamma_r() is.
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
extern "C" double lgamma_r(double, int*);
double lgamma_ts(double x) { int sgn = 0; return ::lgamma_r(x, &sgn); }
#else
double lgamma_ts(double x) { return std::lgamma(x); }
#endif


// ===== Latent space =====

void fill_distance_matrix(const arma::mat& Z, arma::mat& D) {
  const uword n = Z.n_rows, d = Z.n_cols;
  if (D.n_rows != n || D.n_cols != n) D.set_size(n, n);
  D.zeros();
  for (uword i = 0; i < n; ++i) {
    for (uword j = i + 1; j < n; ++j) {
      double s = 0.0;
      for (uword l = 0; l < d; ++l) {
        const double diff = Z(i, l) - Z(j, l);
        s += diff * diff;
      }
      D(i, j) = D(j, i) = std::sqrt(s);
    }
  }
}

// [[Rcpp::export]]
arma::mat compute_distance_matrix(const arma::mat& Z) {
  arma::mat D;
  fill_distance_matrix(Z, D);
  return D;
}

// Zero-mean positions; applied only to the initial state and to saved draws.
void recenter_Z(arma::mat& Z) { Z.each_row() -= mean(Z, 0); }

// Exact Gibbs step for a rigid translation of all positions,
// c ~ N_d(-zbar, sigma_z^2 / n I): the likelihood depends only on distances.
void gibbs_translate_Z(arma::mat& Z, double sigma_z, RNG& rng) {
  const double sd = sigma_z / std::sqrt(static_cast<double>(Z.n_rows));
  const arma::rowvec zbar = mean(Z, 0);
  for (uword l = 0; l < Z.n_cols; ++l) Z.col(l) += rng.rnorm(sd) - zbar(l);
}


// ===== Hurdle negative binomial log-likelihood =====

// log(1 - exp(x)) for x < 0 (Maechler, 2012).
// [[Rcpp::export]]
double log1mexp_safe(double x) {
  if (x >= 0.0)  return -std::numeric_limits<double>::infinity();
  if (x < -30.0) return 0.0;
  if (x < -std::log(2.0)) return std::log1p(-std::exp(x));
  return std::log(-std::expm1(x));
}

double log1pexp(double x) {
  if (x >  35.0) return x;
  if (x < -20.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

// Observed dyads of one slice (t,k). Only positive edges are stored;
// zeros enter the likelihood only through their number n0.
struct EdgeList {
  std::vector<int>    ei, ej;   // positive edges i -> j
  std::vector<double> ey;       // their counts
  std::vector<double> uy;       // distinct counts
  std::vector<int>    eu;       // position of ey[e] in uy
  std::vector<double> lg_yp1;   // lgamma(uy + 1)
  std::vector<std::vector<std::pair<int, int>>> inc;  // node -> (other node, edge)
  double n0   = 0.0;
  double npos = 0.0;
};

EdgeList build_edge_list(const arma::mat& Y, const arma::mat& W) {
  const int n = Y.n_rows;
  EdgeList E;
  E.inc.resize(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j || W(i, j) <= 0.5) continue;
      const double y = Y(i, j);
      if (y > 0.0) {
        const int e = E.ey.size();
        E.ei.push_back(i); E.ej.push_back(j); E.ey.push_back(y);
        E.npos += 1.0;
        E.inc[i].push_back(std::make_pair(j, e));
        E.inc[j].push_back(std::make_pair(i, e));
      } else {
        E.n0 += 1.0;
      }
    }
  }
  E.uy = E.ey;
  std::sort(E.uy.begin(), E.uy.end());
  E.uy.erase(std::unique(E.uy.begin(), E.uy.end()), E.uy.end());
  E.eu.resize(E.ey.size());
  for (size_t e = 0; e < E.ey.size(); ++e)
    E.eu[e] = std::lower_bound(E.uy.begin(), E.uy.end(), E.ey[e]) - E.uy.begin();
  E.lg_yp1.resize(E.uy.size());
  for (size_t u = 0; u < E.uy.size(); ++u)
    E.lg_yp1[u] = lgamma_ts(E.uy[u] + 1.0);
  return E;
}

// Terms of the HNB that depend on beta and r only.
struct SliceCache {
  double log_pi   = 0.0;
  double log_1mpi = 0.0;
  double r        = 1.0;
  std::vector<double> Cu;   // lgamma(uy + r) - lgamma(r) - lgamma(uy + 1)
};

void set_beta(SliceCache& c, double beta) {
  c.log_pi   = -log1pexp(-beta);
  c.log_1mpi = -log1pexp( beta);
}

void set_r(SliceCache& c, double r, const EdgeList& E) {
  c.r = r;
  const double lgr = lgamma_ts(r);
  c.Cu.resize(E.uy.size());
  for (size_t u = 0; u < E.uy.size(); ++u)
    c.Cu[u] = lgamma_ts(E.uy[u] + r) - lgr - E.lg_yp1[u];
}

// log NB(y | mu, r) - log(1 - NB(0 | mu, r)), with mu = exp(lp).
// A positive edge contributes log(pi) + edge_core, a zero contributes log(1 - pi).
double edge_core(double y, double lp, double r, double Cu, long long& n_clamp) {
  if      (lp >  LP_LIM) { lp =  LP_LIM; ++n_clamp; }
  else if (lp < -LP_LIM) { lp = -LP_LIM; ++n_clamp; }
  const double mu = std::exp(lp);
  double l0 = -r * std::log1p(mu / r);
  if (l0 > L0_MAX) l0 = L0_MAX;
  const double ly = l0 + Cu + y * (lp - std::log(r + mu));
  return ly - log1mexp_safe(l0);
}

double core_loglik(const EdgeList& E, const arma::mat& D, double alpha,
                   const SliceCache& c, long long& n_clamp) {
  double s = 0.0;
  for (size_t e = 0; e < E.ey.size(); ++e)
    s += edge_core(E.ey[e], alpha - D(E.ei[e], E.ej[e]), c.r,
                   c.Cu[E.eu[e]], n_clamp);
  return s;
}

double full_loglik(double core, const EdgeList& E, const SliceCache& c) {
  return core + E.npos * c.log_pi + E.n0 * c.log_1mpi;
}

// [[Rcpp::export]]
double loglik_network(const arma::mat& Ytk, const arma::mat& Dg,
                      double alpha, double beta, double r,
                      const arma::mat& Wtk) {
  if (r <= 0.0) stop("loglik_network: r must be > 0");
  const int n = Ytk.n_rows;
  const double log_pi   = -log1pexp(-beta);
  const double log_1mpi = -log1pexp( beta);
  const double lgr      = lgamma_ts(r);
  long long dummy = 0;
  double ll = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j || Wtk(i, j) <= 0.5) continue;
      const double y = Ytk(i, j);
      if (y > 0.0) {
        const double Cu = lgamma_ts(y + r) - lgr - lgamma_ts(y + 1.0);
        ll += log_pi + edge_core(y, alpha - Dg(i, j), r, Cu, dummy);
      } else {
        ll += log_1mpi;
      }
    }
  }
  return ll;
}


// ===== Hidden states =====

// Draws a category in 0..G-1 from unnormalised log-weights.
int sample_cat_logw(const double* logw, int G, RNG& rng) {
  std::vector<double> w(G);
  double mx = -std::numeric_limits<double>::infinity();
  for (int g = 0; g < G; ++g) if (logw[g] > mx) mx = logw[g];
  double s = 0.0;
  if (std::isfinite(mx)) {
    for (int g = 0; g < G; ++g) { w[g] = std::exp(logw[g] - mx); s += w[g]; }
  }
  if (!std::isfinite(s) || s <= 0.0) {
    for (int g = 0; g < G; ++g) w[g] = 1.0;
    s = G;
  }
  const double u = rng.unif() * s;
  double cum = 0.0;
  for (int g = 0; g < G; ++g) { cum += w[g]; if (u < cum) return g; }
  return G - 1;
}

void log_softmax(const double* eta, int G, double* out) {
  double mx = eta[0];
  for (int g = 1; g < G; ++g) if (eta[g] > mx) mx = eta[g];
  double s = 0.0;
  for (int g = 0; g < G; ++g) s += std::exp(eta[g] - mx);
  const double lse = mx + std::log(s);
  for (int g = 0; g < G; ++g) out[g] = eta[g] - lse;
}

// log P(h -> g) for layer k: softmax over g of gamma(h,g) + delta(k,g).
arma::mat make_logP(const arma::mat& gamma, const arma::mat& delta, int k) {
  const int G = gamma.n_rows;
  arma::mat logP(G, G);
  arma::vec eta(G), out(G);
  for (int h = 0; h < G; ++h) {
    for (int g = 0; g < G; ++g) eta(g) = gamma(h, g) + delta(k, g);
    log_softmax(eta.memptr(), G, out.memptr());
    for (int g = 0; g < G; ++g) logP(h, g) = out(g);
  }
  return logP;
}

// N[k](h,g) = number of transitions h -> g in layer k.
void transition_counts(const arma::imat& S, std::vector<arma::mat>& N) {
  const int T = S.n_rows, K = S.n_cols;
  for (int k = 0; k < K; ++k) {
    N[k].zeros();
    for (int t = 1; t < T; ++t) N[k](S(t - 1, k), S(t, k)) += 1.0;
  }
}

// Forward filtering, backward sampling of S(., k).
// LL(t, g) = log p(Y_tk | regime g); the initial distribution is uniform.
void update_S_ffbs(int k, arma::imat& S, const arma::mat& LL,
                   const arma::mat& logP, RNG& rng) {
  const int T = LL.n_rows, G = LL.n_cols;
  arma::mat logf(T, G);
  arma::vec v(G);
  const double log_init = -std::log(static_cast<double>(G));
  
  for (int g = 0; g < G; ++g) logf(0, g) = LL(0, g) + log_init;
  for (int t = 1; t < T; ++t) {
    for (int g = 0; g < G; ++g) {
      double mx = -std::numeric_limits<double>::infinity();
      for (int h = 0; h < G; ++h) {
        v(h) = logf(t - 1, h) + logP(h, g);
        if (v(h) > mx) mx = v(h);
      }
      double s = 0.0;
      if (std::isfinite(mx)) for (int h = 0; h < G; ++h) s += std::exp(v(h) - mx);
      const double lse = (std::isfinite(mx) && s > 0.0) ? mx + std::log(s) : -1e300;
      logf(t, g) = LL(t, g) + lse;
    }
  }
  
  for (int g = 0; g < G; ++g) v(g) = logf(T - 1, g);
  S(T - 1, k) = sample_cat_logw(v.memptr(), G, rng);
  for (int t = T - 2; t >= 0; --t) {
    const int gnext = S(t + 1, k);
    for (int g = 0; g < G; ++g) v(g) = logf(t, g) + logP(g, gnext);
    S(t, k) = sample_cat_logw(v.memptr(), G, rng);
  }
}


// ===== Latent positions =====
// Random-walk MH for each node of regime g, using the slices currently in g.
// Only positive edges incident to the node matter: zeros contribute log(1 - pi),
// which does not depend on Z.

void update_Z(arma::mat& Zg, int g,
              const std::vector<int>& slices,   // indices t*K + k
              const std::vector<EdgeList>& E,
              const std::vector<SliceCache>& C,
              const arma::mat& alpha, int K,
              const arma::mat& sd_Z, double sigma_z,
              arma::umat& acc_Z, RNG& rng,
              long long& n_clamp, long long& n_nonfinite)
{
  const int n = Zg.n_rows, d = Zg.n_cols;
  const double inv2s2 = 0.5 / (sigma_z * sigma_z);
  std::vector<double> znew(d);
  
  gibbs_translate_Z(Zg, sigma_z, rng);
  
  for (int i = 0; i < n; ++i) {
    double q_new = 0.0, q_old = 0.0;
    for (int l = 0; l < d; ++l) {
      znew[l] = Zg(i, l) + rng.rnorm(sd_Z(g, i));
      q_new += znew[l] * znew[l];
      q_old += Zg(i, l) * Zg(i, l);
    }
    const double log_prior = -inv2s2 * (q_new - q_old);
    
    double ll_diff = 0.0;
    for (int idx : slices) {
      const EdgeList&   El = E[idx];
      const SliceCache& c  = C[idx];
      const double a_tk = alpha(idx / K, idx % K);
      for (const auto& p : El.inc[i]) {
        const int j = p.first, e = p.second;
        double s_old = 0.0, s_new = 0.0;
        for (int l = 0; l < d; ++l) {
          const double zo = Zg(i, l) - Zg(j, l);
          const double zn = znew[l]  - Zg(j, l);
          s_old += zo * zo;
          s_new += zn * zn;
        }
        const double Cu = c.Cu[El.eu[e]];
        ll_diff += edge_core(El.ey[e], a_tk - std::sqrt(s_new), c.r, Cu, n_clamp)
          - edge_core(El.ey[e], a_tk - std::sqrt(s_old), c.r, Cu, n_clamp);
      }
    }
    
    if (accept(log_prior + ll_diff, rng, n_nonfinite)) {
      for (int l = 0; l < d; ++l) Zg(i, l) = znew[l];
      acc_Z(g, i) += 1;
    }
  }
}


// ===== Slice parameters alpha, beta, r =====
// `core` is the current core_loglik of the slice and is kept up to date.

void update_alpha(double& a, double& core, const EdgeList& E, const arma::mat& D,
                  const SliceCache& c, double sd, double mu_alpha, double sigma_alpha,
                  unsigned& acc, RNG& rng, long long& n_clamp, long long& n_nonfinite)
{
  const double a_new = a + rng.rnorm(sd);
  const double du_n = (a_new - mu_alpha) / sigma_alpha;
  const double du_o = (a     - mu_alpha) / sigma_alpha;
  const double lpr  = -0.5 * (du_n * du_n - du_o * du_o);
  const double core_new = core_loglik(E, D, a_new, c, n_clamp);
  if (accept(lpr + (core_new - core), rng, n_nonfinite)) {
    a = a_new; core = core_new; ++acc;
  }
}

// beta enters only through npos * log(pi) + n0 * log(1 - pi).
void update_beta(double& b, SliceCache& c, const EdgeList& E,
                 double sd, double mu_beta, double sigma_beta,
                 unsigned& acc, RNG& rng, long long& n_nonfinite)
{
  const double b_new = b + rng.rnorm(sd);
  const double du_n = (b_new - mu_beta) / sigma_beta;
  const double du_o = (b     - mu_beta) / sigma_beta;
  const double lpr  = -0.5 * (du_n * du_n - du_o * du_o);
  const double lpi_new   = -log1pexp(-b_new);
  const double l1mpi_new = -log1pexp( b_new);
  const double dll = E.npos * (lpi_new   - c.log_pi)
    + E.n0   * (l1mpi_new - c.log_1mpi);
  if (accept(lpr + dll, rng, n_nonfinite)) {
    b = b_new; c.log_pi = lpi_new; c.log_1mpi = l1mpi_new; ++acc;
  }
}

// Update on a = r^{-1/2} with prior half-Normal(0, sigma_a^2).
// The proposal is reflected at 0; proposals with a < A_MIN are rejected.
void update_r(double& r, double& core, SliceCache& c,
              const EdgeList& E, const arma::mat& D, double alpha,
              double sd, double sigma_a,
              unsigned& acc, RNG& rng, long long& n_clamp, long long& n_nonfinite)
{
  const double a_old = 1.0 / std::sqrt(r);
  const double a_new = std::fabs(a_old + rng.rnorm(sd));
  if (a_new < A_MIN) return;
  const double r_new = 1.0 / (a_new * a_new);
  const double lpr = -0.5 * ((a_new * a_new) - (a_old * a_old)) / (sigma_a * sigma_a);
  
  SliceCache prop = c;
  set_r(prop, r_new, E);
  const double core_new = core_loglik(E, D, alpha, prop, n_clamp);
  if (accept(lpr + (core_new - core), rng, n_nonfinite)) {
    r = r_new; c = prop; core = core_new; ++acc;
  }
}


// ===== Transition parameters gamma, delta =====
// MH on each entry, using the transition counts N. The last column of gamma
// and delta is fixed to 0.

// Adds to dll the change in the log-likelihood of the transitions out of h in
// layer k when eta(h, g) = gamma(h, g) + delta(k, g) is replaced by eta_new.
void add_dll(double& dll, const arma::mat& gamma, const arma::mat& delta,
             const arma::mat& Nk, int h, int k, int g, double eta_new)
{
  const int G = gamma.n_rows;
  if (arma::accu(Nk.row(h)) == 0.0) return;
  arma::vec eta(G), lp_old(G), lp_new(G);
  for (int gg = 0; gg < G; ++gg) eta(gg) = gamma(h, gg) + delta(k, gg);
  log_softmax(eta.memptr(), G, lp_old.memptr());
  eta(g) = eta_new;
  log_softmax(eta.memptr(), G, lp_new.memptr());
  for (int gg = 0; gg < G; ++gg) dll += Nk(h, gg) * (lp_new(gg) - lp_old(gg));
}

void update_gamma(arma::mat& gamma, const arma::mat& delta,
                  const std::vector<arma::mat>& N,
                  const arma::mat& sd, double sigma_gamma,
                  arma::umat& acc, RNG& rng, long long& n_nonfinite)
{
  const int G = gamma.n_rows, K = N.size();
  for (int h = 0; h < G; ++h) {
    for (int g = 0; g < G - 1; ++g) {
      const double old_val = gamma(h, g);
      const double new_val = old_val + rng.rnorm(sd(h, g));
      const double lpr = -0.5 * (new_val * new_val - old_val * old_val)
        / (sigma_gamma * sigma_gamma);
      double dll = 0.0;
      for (int k = 0; k < K; ++k)
        add_dll(dll, gamma, delta, N[k], h, k, g, new_val + delta(k, g));
      if (accept(lpr + dll, rng, n_nonfinite)) { gamma(h, g) = new_val; acc(h, g) += 1; }
    }
    gamma(h, G - 1) = 0.0;
  }
}

void update_delta(arma::mat& delta, const arma::mat& gamma,
                  const std::vector<arma::mat>& N,
                  const arma::mat& sd, double sigma_delta,
                  arma::umat& acc, RNG& rng, long long& n_nonfinite)
{
  const int G = delta.n_cols, K = delta.n_rows;
  for (int k = 0; k < K; ++k) {
    for (int g = 0; g < G - 1; ++g) {
      const double old_val = delta(k, g);
      const double new_val = old_val + rng.rnorm(sd(k, g));
      const double lpr = -0.5 * (new_val * new_val - old_val * old_val)
        / (sigma_delta * sigma_delta);
      double dll = 0.0;
      for (int h = 0; h < G; ++h)
        add_dll(dll, gamma, delta, N[k], h, k, g, gamma(h, g) + new_val);
      if (accept(lpr + dll, rng, n_nonfinite)) { delta(k, g) = new_val; acc(k, g) += 1; }
    }
    delta(k, G - 1) = 0.0;
  }
}

// gamma(., g) + c and delta(., g) - c leave the transition probabilities unchanged.

// Moves the mean of delta(., g) into gamma(., g), so that sum_k delta(k, g) = 0;
// applied only to the initial state and to saved draws.
void center_delta_into_gamma(arma::mat& gamma, arma::mat& delta) {
  const int G = delta.n_cols, K = delta.n_rows;
  for (int g = 0; g < G - 1; ++g) {
    double m = 0.0;
    for (int k = 0; k < K; ++k) m += delta(k, g);
    m /= static_cast<double>(K);
    for (int k = 0; k < K; ++k) delta(k, g) -= m;
    for (int h = 0; h < G; ++h) gamma(h, g) += m;
  }
}

// Exact Gibbs step for c, column by column; only the priors depend on c.
void gibbs_ridge_gamma_delta(arma::mat& gamma, arma::mat& delta,
                             double sigma_gamma, double sigma_delta, RNG& rng) {
  const int G = gamma.n_rows, K = delta.n_rows;
  const double pg = 1.0 / (sigma_gamma * sigma_gamma);
  const double pd = 1.0 / (sigma_delta * sigma_delta);
  const double P  = G * pg + K * pd;
  const double sd = 1.0 / std::sqrt(P);
  for (int g = 0; g < G - 1; ++g) {
    const double m = (pd * arma::accu(delta.col(g)) - pg * arma::accu(gamma.col(g))) / P;
    const double c = m + rng.rnorm(sd);
    gamma.col(g) += c;
    delta.col(g) -= c;
  }
}


// ===== Input checks =====

void check_dim(const arma::mat& A, const char* nm, int nr, int nc) {
  if ((int)A.n_rows != nr || (int)A.n_cols != nc)
    stop("%s must be %d x %d", nm, nr, nc);
  if (!A.is_finite()) stop("%s contains non-finite values", nm);
}

void check_cube(const arma::cube& A, const char* nm, int k, int n, int T) {
  if ((int)A.n_rows != n || (int)A.n_cols != n || (int)A.n_slices != T)
    stop("%s[[%d]] must be %d x %d x %d", nm, k + 1, n, n, T);
  if (!A.is_finite()) stop("%s[[%d]] contains NA/NaN/Inf", nm, k + 1);
}


// ===== Main MCMC loop =====
// Note: do not put comments inside the argument list of an exported function,
// Rcpp cannot parse them.

// [[Rcpp::export]]
List run_mcmc_hmm(List Y_list,
                  List W_list,
                  int G,
                  List Z_init,
                  arma::mat S_init,
                  arma::mat alpha_init,
                  arma::mat beta_init,
                  arma::mat r_init,
                  arma::mat gamma_init,
                  arma::mat delta_init,
                  int n_iter,
                  int n_burnin,
                  int n_thin,
                  List priors,
                  List proposals,
                  double seed = 20260101,
                  bool verbose = true)
{
  // ---- Dimensions and checks ----
  if (G < 1) stop("G must be >= 1");
  if (n_thin < 1 || n_burnin < 0 || n_iter <= n_burnin)
    stop("need n_thin >= 1 and 0 <= n_burnin < n_iter");
  if (Y_list.size() == 0 || W_list.size() != Y_list.size())
    stop("Y_list and W_list must be non-empty lists of the same length");
  
  const int K = Y_list.size();
  const arma::cube Y0 = as<arma::cube>(Y_list[0]);
  const int n = Y0.n_rows, T = Y0.n_slices;
  if (Z_init.size() != G) stop("Z_init must have G = %d elements", G);
  const int d = as<arma::mat>(Z_init[0]).n_cols;
  
  check_dim(S_init,     "S_init",     T, K);
  check_dim(alpha_init, "alpha_init", T, K);
  check_dim(beta_init,  "beta_init",  T, K);
  check_dim(r_init,     "r_init",     T, K);
  check_dim(gamma_init, "gamma_init", G, G);
  check_dim(delta_init, "delta_init", K, G);
  if (S_init.min() < 0 || S_init.max() > G - 1)
    stop("S_init must contain states 0..%d (0-based)", G - 1);
  if (r_init.min() <= 0 || r_init.max() > R_MAX)
    stop("r_init must be in (0, %g]", R_MAX);
  
  // ---- Edge lists, one per slice t*K + k ----
  std::vector<EdgeList> E(T * K);
  for (int k = 0; k < K; ++k) {
    const arma::cube Yk = as<arma::cube>(Y_list[k]);
    const arma::cube Wk = as<arma::cube>(W_list[k]);
    check_cube(Yk, "Y_list", k, n, T);
    check_cube(Wk, "W_list", k, n, T);
    for (int t = 0; t < T; ++t) E[t * K + k] = build_edge_list(Yk.slice(t), Wk.slice(t));
  }
  double n_obs = 0.0, n_pos = 0.0, y_max = 0.0;
  for (const EdgeList& e : E) {
    n_obs += e.n0 + e.npos;
    n_pos += e.npos;
    if (!e.uy.empty()) y_max = std::max(y_max, e.uy.back());
  }
  if (n_pos == 0.0) stop("no positive edges observed");
  
  // ---- Initial state ----
  std::vector<arma::mat> Z(G);
  for (int g = 0; g < G; ++g) {
    Z[g] = as<arma::mat>(Z_init[g]);
    check_dim(Z[g], "Z_init[[g]]", n, d);
    recenter_Z(Z[g]);
  }
  arma::imat S = arma::conv_to<arma::imat>::from(S_init);
  arma::mat alpha = alpha_init, beta = beta_init, r = r_init;
  arma::mat gamma = gamma_init, delta = delta_init;
  gamma.col(G - 1).zeros();
  delta.col(G - 1).zeros();
  center_delta_into_gamma(gamma, delta);
  
  // ---- Priors and proposal sds ----
  const double sigma_z     = as<double>(priors["sigma_z"]);
  const double mu_alpha    = as<double>(priors["mu_alpha"]);
  const double sigma_alpha = as<double>(priors["sigma_alpha"]);
  const double mu_beta     = as<double>(priors["mu_beta"]);
  const double sigma_beta  = as<double>(priors["sigma_beta"]);
  const double sigma_a     = as<double>(priors["sigma_a"]);
  const double sigma_gamma = as<double>(priors["sigma_gamma"]);
  const double sigma_delta = as<double>(priors["sigma_delta"]);
  
  arma::mat sd_Z    (G, n, fill::value(as<double>(proposals["sd_z"])));
  arma::mat sd_alpha(T, K, fill::value(as<double>(proposals["sd_alpha"])));
  arma::mat sd_beta (T, K, fill::value(as<double>(proposals["sd_beta"])));
  arma::mat sd_r    (T, K, fill::value(as<double>(proposals["sd_r"])));
  arma::mat sd_gamma(G, G, fill::value(as<double>(proposals["sd_gamma"])));
  arma::mat sd_delta(K, G, fill::value(as<double>(proposals["sd_delta"])));
  
  // ---- Storage of the draws ----
  const int nsave = (n_iter - n_burnin + n_thin - 1) / n_thin;
  std::vector<std::vector<arma::mat>> Z_samp(nsave, std::vector<arma::mat>(G));
  arma::cube S_samp    (T, K, nsave);
  arma::cube Alpha_samp(T, K, nsave);
  arma::cube Beta_samp (T, K, nsave);
  arma::cube R_samp    (T, K, nsave);
  arma::cube Gamma_samp(G, G, nsave);
  arma::cube Delta_samp(K, G, nsave);
  arma::vec  ll_samp   (nsave);
  
  // ---- One RNG stream per layer, regime and slice, plus one for gamma/delta ----
  const uint64_t master = static_cast<uint64_t>(seed);
  std::vector<RNG> rng_layer, rng_reg, rng_slice;
  for (int k = 0; k < K; ++k)     rng_layer.emplace_back(stream_seed(master, 1000000ULL + k));
  for (int g = 0; g < G; ++g)     rng_reg.emplace_back(stream_seed(master, 2000000ULL + g));
  for (int i = 0; i < T * K; ++i) rng_slice.emplace_back(stream_seed(master, 3000000ULL + i));
  RNG rng_trans(stream_seed(master, 4000000ULL));
  
  // ---- Acceptance counters: _b for the current batch, _p after burn-in ----
  arma::umat acc_Z_b    (G, n, fill::zeros), acc_Z_p    (G, n, fill::zeros);
  arma::umat acc_alpha_b(T, K, fill::zeros), acc_alpha_p(T, K, fill::zeros);
  arma::umat acc_beta_b (T, K, fill::zeros), acc_beta_p (T, K, fill::zeros);
  arma::umat acc_r_b    (T, K, fill::zeros), acc_r_p    (T, K, fill::zeros);
  arma::umat acc_gamma_b(G, G, fill::zeros), acc_gamma_p(G, G, fill::zeros);
  arma::umat acc_delta_b(K, G, fill::zeros), acc_delta_p(K, G, fill::zeros);
  long long n_clamp = 0, n_nonfinite = 0;
  int adapt_batch = 0, sidx = 0;
  double sum_loglik = 0.0;
  
  // ---- Working objects ----
  std::vector<arma::mat> D(G);
  std::vector<SliceCache> C(T * K);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < K; ++k) {
      set_beta(C[t * K + k], beta(t, k));
      set_r   (C[t * K + k], r(t, k), E[t * K + k]);
    }
  arma::mat  ll_slice(T, K, fill::zeros);
  arma::cube LL(T, G, K);
  std::vector<arma::mat> N(K, arma::mat(G, G, fill::zeros));
  std::vector<std::vector<int>> slices(G);
  
  if (verbose) {
    Rcout << "LS-HMM MCMC | n=" << n << " T=" << T << " K=" << K
          << " G=" << G << " d=" << d << " seed=" << (long long)seed << "\n"
          << "  observed dyads=" << (long long)n_obs
          << " | positive edges=" << (long long)n_pos
          << " (" << 100.0 * n_pos / n_obs << "%)"
          << " | y_max=" << y_max << std::endl;
#ifdef _OPENMP
    Rcout << "  OpenMP: " << omp_get_max_threads() << " threads" << std::endl;
#endif
  }
  
  for (int iter = 0; iter < n_iter; ++iter) {
    
    for (int g = 0; g < G; ++g) fill_distance_matrix(Z[g], D[g]);
    
    // ---- Hidden states S ----
    {
      long long cl = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) reduction(+:cl)
#endif
      for (int k = 0; k < K; ++k)
        for (int t = 0; t < T; ++t)
          for (int g = 0; g < G; ++g) {
            const int idx = t * K + k;
            const double core = core_loglik(E[idx], D[g], alpha(t, k), C[idx], cl);
            LL(t, g, k) = full_loglik(core, E[idx], C[idx]);
          }
      n_clamp += cl;
      
      for (int k = 0; k < K; ++k)
        update_S_ffbs(k, S, LL.slice(k), make_logP(gamma, delta, k), rng_layer[k]);
    }
    
    // ---- Latent positions Z[g], in parallel over regimes ----
    {
      for (int g = 0; g < G; ++g) slices[g].clear();
      for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) slices[S(t, k)].push_back(t * K + k);
      
      long long cl = 0, nf = 0;
#ifdef _OPENMP
      // schedule(static) for compatibility with older libomp (macOS)
#pragma omp parallel for schedule(static) reduction(+:cl,nf)
#endif
      for (int g = 0; g < G; ++g)
        update_Z(Z[g], g, slices[g], E, C, alpha, K, sd_Z, sigma_z,
                 acc_Z_b, rng_reg[g], cl, nf);
      n_clamp += cl; n_nonfinite += nf;
      
      for (int g = 0; g < G; ++g) fill_distance_matrix(Z[g], D[g]);
    }
    
    // ---- alpha, beta, r, in parallel over slices ----
    {
      long long cl = 0, nf = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) reduction(+:cl,nf)
#endif
      for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
          const int idx = t * K + k;
          const int g   = S(t, k);
          unsigned aA = 0, aB = 0, aR = 0;
          double core = core_loglik(E[idx], D[g], alpha(t, k), C[idx], cl);
          
          update_alpha(alpha(t, k), core, E[idx], D[g], C[idx],
                       sd_alpha(t, k), mu_alpha, sigma_alpha, aA, rng_slice[idx], cl, nf);
          update_beta (beta(t, k), C[idx], E[idx],
                       sd_beta(t, k), mu_beta, sigma_beta, aB, rng_slice[idx], nf);
          update_r    (r(t, k), core, C[idx], E[idx], D[g], alpha(t, k),
                       sd_r(t, k), sigma_a, aR, rng_slice[idx], cl, nf);
          
          acc_alpha_b(t, k) += aA;
          acc_beta_b (t, k) += aB;
          acc_r_b    (t, k) += aR;
          ll_slice(t, k) = full_loglik(core, E[idx], C[idx]);
        }
      }
      n_clamp += cl; n_nonfinite += nf;
    }
    
    // ---- gamma and delta ----
    transition_counts(S, N);
    gibbs_ridge_gamma_delta(gamma, delta, sigma_gamma, sigma_delta, rng_trans);
    update_gamma(gamma, delta, N, sd_gamma, sigma_gamma, acc_gamma_b, rng_trans, n_nonfinite);
    update_delta(delta, gamma, N, sd_delta, sigma_delta, acc_delta_b, rng_trans, n_nonfinite);
    
    // ---- Adaptation of the proposal sds (burn-in only) ----
    if ((iter + 1) <= n_burnin && (iter + 1) % ADAPT_EVERY == 0) {
      ++adapt_batch;
      const double w  = ADAPT_EVERY;
      const double st = std::min(0.5, 1.0 / std::sqrt((double)adapt_batch));
      
      for (int g = 0; g < G; ++g)
        for (int i = 0; i < n; ++i)
          sd_Z(g, i) *= std::exp(st * (acc_Z_b(g, i) / w - ACC_TARGET_BLOCK));
      for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
          sd_alpha(t, k) *= std::exp(st * (acc_alpha_b(t, k) / w - ACC_TARGET_SCALAR));
          sd_beta (t, k) *= std::exp(st * (acc_beta_b (t, k) / w - ACC_TARGET_SCALAR));
          sd_r    (t, k) *= std::exp(st * (acc_r_b    (t, k) / w - ACC_TARGET_SCALAR));
        }
      for (int h = 0; h < G; ++h)
        for (int g = 0; g < G - 1; ++g)
          sd_gamma(h, g) *= std::exp(st * (acc_gamma_b(h, g) / w - ACC_TARGET_SCALAR));
      for (int k = 0; k < K; ++k)
        for (int g = 0; g < G - 1; ++g)
          sd_delta(k, g) *= std::exp(st * (acc_delta_b(k, g) / w - ACC_TARGET_SCALAR));
      
      sd_Z.clamp(1e-8, 1e4); sd_alpha.clamp(1e-8, 1e4); sd_beta.clamp(1e-8, 1e4);
      sd_r.clamp(1e-8, 1e4); sd_gamma.clamp(1e-8, 1e4); sd_delta.clamp(1e-8, 1e4);
      
      acc_Z_b.zeros(); acc_alpha_b.zeros(); acc_beta_b.zeros();
      acc_r_b.zeros(); acc_gamma_b.zeros(); acc_delta_b.zeros();
    }
    
    // ---- Acceptance counts after burn-in (from iteration n_burnin + 1) ----
    if (iter > n_burnin) {
      acc_Z_p     += acc_Z_b;
      acc_alpha_p += acc_alpha_b;
      acc_beta_p  += acc_beta_b;
      acc_r_p     += acc_r_b;
      acc_gamma_p += acc_gamma_b;
      acc_delta_p += acc_delta_b;
    }
    if (iter >= n_burnin) {
      acc_Z_b.zeros(); acc_alpha_b.zeros(); acc_beta_b.zeros();
      acc_r_b.zeros(); acc_gamma_b.zeros(); acc_delta_b.zeros();
    }
    
    // ---- Save the draw (centred copies; the chain itself is not changed) ----
    if (iter >= n_burnin && (iter - n_burnin) % n_thin == 0 && sidx < nsave) {
      for (int g = 0; g < G; ++g) { Z_samp[sidx][g] = Z[g]; recenter_Z(Z_samp[sidx][g]); }
      S_samp.slice(sidx)     = arma::conv_to<arma::mat>::from(S);
      Alpha_samp.slice(sidx) = alpha;
      Beta_samp.slice(sidx)  = beta;
      R_samp.slice(sidx)     = r;
      arma::mat gam_c = gamma, del_c = delta;
      center_delta_into_gamma(gam_c, del_c);
      Gamma_samp.slice(sidx) = gam_c;
      Delta_samp.slice(sidx) = del_c;
      ll_samp(sidx) = accu(ll_slice);
      sum_loglik   += ll_samp(sidx);
      ++sidx;
    }
    
    if (verbose && (iter + 1) % 200 == 0) {
      arma::vec cnt(G, fill::zeros);
      for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) cnt(S(t, k))++;
      Rcout << "Iter " << (iter + 1) << "/" << n_iter
            << " | ll=" << accu(ll_slice) << " | occ: ";
      for (int g = 0; g < G; ++g)
        Rcout << "g" << g << "=" << (int)std::round(100.0 * cnt(g) / (T * K)) << "% ";
      Rcout << "| sd_z=" << arma::mean(arma::vectorise(sd_Z))
            << " sd_a=" << arma::mean(arma::vectorise(sd_alpha))
            << " | clamp=" << n_clamp << std::endl;
    }
    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }
  
  // ---- DIC ----
  // theta_bar is built from the raw draws, so D(theta_bar) is valid only without
  // label switching; recompute it in R after relabelling.
  const double D_bar = -2.0 * sum_loglik / sidx;
  
  const arma::mat alpha_bar = arma::mean(Alpha_samp.slices(0, sidx - 1), 2);
  const arma::mat beta_bar  = arma::mean(Beta_samp.slices(0, sidx - 1),  2);
  const arma::mat r_bar     = arma::mean(R_samp.slices(0, sidx - 1),     2);
  
  std::vector<arma::mat> D_hat(G);
  for (int g = 0; g < G; ++g) {
    arma::mat Z_bar(n, d, fill::zeros);
    for (int s = 0; s < sidx; ++s) Z_bar += Z_samp[s][g];
    Z_bar /= static_cast<double>(sidx);
    D_hat[g] = compute_distance_matrix(Z_bar);
  }
  
  double ll_bar = 0.0;
  long long dummy = 0;
  SliceCache cb;
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < K; ++k) {
      // modal regime of slice (t,k)
      arma::vec cnt(G, fill::zeros);
      for (int s = 0; s < sidx; ++s) cnt((int)S_samp(t, k, s))++;
      const int g_mode = cnt.index_max();
      
      const int idx = t * K + k;
      set_beta(cb, beta_bar(t, k));
      set_r(cb, std::min(std::max(r_bar(t, k), 1e-8), R_MAX), E[idx]);
      const double core = core_loglik(E[idx], D_hat[g_mode], alpha_bar(t, k), cb, dummy);
      ll_bar += full_loglik(core, E[idx], cb);
    }
  const double D_theta_bar = -2.0 * ll_bar;
  const double pD  = D_bar - D_theta_bar;
  const double DIC = D_bar + pD;
  
  // ---- Output ----
  List Zout(sidx);
  for (int s = 0; s < sidx; ++s) {
    List tmp(G);
    for (int g = 0; g < G; ++g) tmp[g] = Z_samp[s][g];
    Zout[s] = tmp;
  }
  const double np = n_iter - n_burnin;
  
  return List::create(
    Named("Z")      = Zout,
    Named("S")      = S_samp,
    Named("alpha")  = Alpha_samp,
    Named("beta")   = Beta_samp,
    Named("r")      = R_samp,
    Named("gamma")  = Gamma_samp,
    Named("delta")  = Delta_samp,
    Named("nsave")  = sidx,
    Named("loglik") = ll_samp,
    Named("seed")   = seed,
    Named("DIC") = List::create(
      Named("DIC")         = DIC,
      Named("D_bar")       = D_bar,
      Named("D_theta_bar") = D_theta_bar,
      Named("pD")          = pD,
      Named("warning")     = "computed on raw draws: recompute in R after relabelling"
    ),
    Named("acc") = List::create(
      Named("Z")     = arma::conv_to<arma::mat>::from(acc_Z_p)     / np,
      Named("alpha") = arma::conv_to<arma::mat>::from(acc_alpha_p) / np,
      Named("beta")  = arma::conv_to<arma::mat>::from(acc_beta_p)  / np,
      Named("r")     = arma::conv_to<arma::mat>::from(acc_r_p)     / np,
      Named("gamma") = arma::conv_to<arma::mat>::from(acc_gamma_p) / np,
      Named("delta") = arma::conv_to<arma::mat>::from(acc_delta_p) / np
    ),
    Named("sd") = List::create(
      Named("Z") = sd_Z, Named("alpha") = sd_alpha, Named("beta") = sd_beta,
      Named("r") = sd_r, Named("gamma") = sd_gamma, Named("delta") = sd_delta
    ),
    Named("diag") = List::create(
      Named("n_clamp")     = static_cast<double>(n_clamp),
      Named("n_nonfinite") = static_cast<double>(n_nonfinite),
      Named("n_obs_dyads") = n_obs,
      Named("n_pos_edges") = n_pos,
      Named("y_max")       = y_max,
      Named("LP_LIM")      = LP_LIM,
      Named("R_MAX")       = R_MAX
    )
  );
}
