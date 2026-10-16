# LS-HMM.

cpp_path <- "cpp/mcmc_hmm_sr.cpp"
r_dir    <- "R"
OUT_FILE <- "lshmm_sr_fits.rds"

suppressPackageStartupMessages({ library(Rcpp); library(mclust); library(posterior) })
sourceCpp(cpp_path)                       # run_mcmc_hmm_sr(), loglik_network_sr()
source(file.path(r_dir, "init.R"))
source(file.path(r_dir, "ecr_utils.R"))
source(file.path(r_dir, "select_G.R"))

# ---- 1. Data: simulated from the SR model with G = 2 ----
simulate_small <- function(n = 20, T = 12, K = 2, G = 2, persist = 0.85, seed = 1) {
  set.seed(seed)
  centres <- 2.5 * cbind(cos(2 * pi * (1:3) / 3), sin(2 * pi * (1:3) / 3))
  Z <- lapply(seq_len(G), function(g) centres[sample(1:3, n, TRUE), ] + matrix(rnorm(2 * n, 0, .5), n))
  P <- matrix((1 - persist) / (G - 1), G, G); diag(P) <- persist
  S <- matrix(1L, T, K)
  for (k in seq_len(K)) for (t in 2:T) S[t, k] <- sample.int(G, 1, prob = P[S[t - 1, k], ])
  alpha <- matrix(runif(T * K, 2.5, 3.5), T, K); beta <- matrix(qlogis(runif(T * K, .2, .4)), T, K)
  r <- matrix(runif(T * K, 1, 3), T, K)
  u <- exp(rnorm(n, 0, .25)); v <- exp(rnorm(n, 0, .25))     # sender and receiver effects around 1
  Y_list <- W_list <- vector("list", K)
  for (k in seq_len(K)) {
    Y <- W <- array(0, c(n, n, T))
    for (t in seq_len(T)) {
      mu  <- exp(alpha[t, k] * outer(u, v, "+") / 2 - as.matrix(dist(Z[[S[t, k]]])))
      act <- matrix(rbinom(n * n, 1, plogis(beta[t, k])), n)
      p0  <- pnbinom(0, size = r[t, k], mu = mu)       # zero-truncated NB on active dyads
      y   <- qnbinom(runif(n * n, p0, 1), size = r[t, k], mu = mu)
      y   <- matrix(pmax(y, 1), n) * act; diag(y) <- 0
      Y[, , t] <- y; W[, , t] <- 1; diag(W[, , t]) <- 0
    }
    Y_list[[k]] <- Y; W_list[[k]] <- W
  }
  list(Y_list = Y_list, W_list = W_list, S = S)
}
sim <- simulate_small()
Y_list <- sim$Y_list; W_list <- sim$W_list

# ---- 2. Settings ----
G_MAX    <- 3
N_CHAINS <- 2
MCMC     <- list(n_iter = 3000L, n_burnin = 1500L, n_thin = 5L)
# Priors and initial proposal sds, as in the thesis. alpha ~ N+(mu_alpha, sigma_alpha^2)
# is now a scale factor: prior_sr_from_base() doubles the base-model values, which
# gives the same prior when u = 0 and v = 1. u_i ~ N(0, sigma_u^2), v_j ~ N(0, sigma_v^2).
PRIORS <- prior_sr_from_base(list(sigma_z = 3, mu_alpha = 2, sigma_alpha = 2, mu_beta = 0, sigma_beta = 2,
                                  sigma_a = 1, sigma_gamma = 2, sigma_delta = 1), sigma_u = 1, sigma_v = 1)
PROPS  <- list(sd_z = .15, sd_alpha = .10, sd_beta = .05, sd_r = .05, sd_gamma = .2, sd_delta = .2,
               sd_u = .05, sd_v = .05)

# ---- 3. MCMC for G = 1, ..., G_MAX ----
fits <- list()
for (G in seq_len(G_MAX)) {
  fits[[as.character(G)]] <- lapply(seq_len(N_CHAINS), function(ch) {
    ini <- make_inits_sr(Y_list, W_list, G, d = 2, seed = 1000L * G + ch)
    t0 <- proc.time()[3]
    f <- run_mcmc_hmm_sr(Y_list, W_list, G, ini$Z, ini$S, ini$alpha, ini$beta, ini$r, ini$gamma, ini$delta,
                         ini$u, ini$v, MCMC$n_iter, MCMC$n_burnin, MCMC$n_thin, PRIORS, PROPS,
                         seed = 5000L * G + 7L * ch, verbose = FALSE, sr_ident = 1L)
    cat(sprintf("G = %d, chain %d: %.0f s | clamp %g, non-finite %g\n", G, ch, proc.time()[3] - t0,
                f$diag$n_clamp, f$diag$n_nonfinite))
    f
  })
}

# ---- 4. Choice of G ----
tab <- select_G(fits, Y_list, W_list, sr = TRUE)
print(tab, digits = 4, row.names = FALSE)
Gsel <- tab$G[tab$selected]
if (length(Gsel) && exists("sim")) {          # simulated example only
  Smod <- modal_states(fits[[as.character(Gsel)]][[1]]$S, Gsel)
  cat(sprintf("\nselected G = %d | ARI(modal regimes chain 1, true regimes) = %.3f\n", Gsel,
              mclust::adjustedRandIndex(c(Smod), c(sim$S))))
  f1 <- fits[[as.character(Gsel)]][[1]]
  cat("reference nodes (u, v):", f1$diag$ref_u, f1$diag$ref_v, "\n")
}
saveRDS(list(fits = fits, selection = tab, priors = PRIORS, props = PROPS, mcmc = MCMC), OUT_FILE)