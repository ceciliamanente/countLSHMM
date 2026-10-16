# Choice of the number of regimes G from several chains for each G.
#
# Rule: among the G with pD >= 0 in every chain, mean minimum occupancy > 0.10
# and mean agreement between chains > 0.70, the one with the smallest mean DIC.
#
# Needs ecr_utils.R, the package posterior, and the compiled sampler
# (loglik_network() for the base model, loglik_network_sr() for the SR model).

# Modal regime of each slice, labels 1..G (ties go to the lower label).
modal_states <- function(S, G) apply(S + 1L, c(1, 2), function(x) which.max(tabulate(x, G)))

# Agreement between chains: the labels of every chain are aligned to the modal
# regimes of chain 1, then the modal regimes of each pair of chains are compared.
chain_agreement <- function(fits, G) {
  S_ref <- modal_states(fits[[1]]$S, G)
  Shat  <- lapply(fits, function(f) modal_states(relabel_ecr(f, S_ref, G)$S, G))
  N <- length(Shat)
  A <- matrix(1, N, N)
  for (i in seq_len(N)) for (j in seq_len(N)) if (i != j) A[i, j] <- mean(Shat[[i]] == Shat[[j]])
  list(mean = mean(A[upper.tri(A)]), Shat = Shat)
}

# Log-likelihood of one slice, base or SR model.
slice_ll <- function(Y, W, D, a, b, r, sr, u = NULL, v = NULL)
  if (sr) loglik_network_sr(Y, D, a, b, r, W, u, v) else loglik_network(Y, D, a, b, r, W)

# Posterior means of one chain, after aligning its labels to its own modal
# regimes. Dbar is the posterior mean of the distance matrix of each regime.
chain_means <- function(f, G, sr) {
  rel <- relabel_ecr(f, modal_states(f$S, G), G)
  M   <- dim(f$S)[3]
  list(Smod = modal_states(rel$S, G),
       Dbar = lapply(seq_len(G), function(g) Reduce(`+`, lapply(rel$Z, function(Zm) as.matrix(dist(Zm[[g]])))) / M),
       a = apply(f$alpha, c(1, 2), mean), b = apply(f$beta, c(1, 2), mean), r = apply(f$r, c(1, 2), mean),
       u = if (sr) colMeans(f$u), v = if (sr) colMeans(f$v))
}

# DIC of one chain. D_bar comes from fit$loglik, which does not depend on the
# labels; D(theta_bar) is computed at the posterior means of chain_means().
dic_chain <- function(f, G, Y_list, W_list, sr = FALSE) {
  m  <- chain_means(f, G, sr)
  ll <- 0
  for (k in seq_along(Y_list)) for (t in seq_len(nrow(m$Smod)))
    ll <- ll + slice_ll(Y_list[[k]][, , t], W_list[[k]][, , t], m$Dbar[[m$Smod[t, k]]],
                        m$a[t, k], m$b[t, k], m$r[t, k], sr, m$u, m$v)
  D_bar <- -2 * mean(f$loglik); pD <- D_bar + 2 * ll
  data.frame(DIC = D_bar + pD, pD = pD, D_bar = D_bar)
}

# Empirical separation between regimes:
#   Delta_gh = mean over slices with Smod = g of [l_tk(Dbar_g) - l_tk(Dbar_h)] / (n (n - 1)),
#   Delta_hat_min = min over g != h.
# Not defined if a regime is empty.
delta_hat_min_chain <- function(f, G, Y_list, W_list, sr = FALSE) {
  if (G < 2) return(NA_real_)
  m <- chain_means(f, G, sr); n <- dim(Y_list[[1]])[1]
  if (any(tabulate(m$Smod, G) == 0)) return(NA_real_)
  Mgh <- matrix(NA_real_, G, G)
  for (g in seq_len(G)) {
    sl <- which(m$Smod == g, arr.ind = TRUE)
    for (h in setdiff(seq_len(G), g))
      Mgh[g, h] <- mean(apply(sl, 1, function(s) {
        t <- s[1]; k <- s[2]; Y <- Y_list[[k]][, , t]; W <- W_list[[k]][, , t]
        slice_ll(Y, W, m$Dbar[[g]], m$a[t, k], m$b[t, k], m$r[t, k], sr, m$u, m$v) -
          slice_ll(Y, W, m$Dbar[[h]], m$a[t, k], m$b[t, k], m$r[t, k], sr, m$u, m$v) })) / (n * (n - 1))
  }
  min(Mgh, na.rm = TRUE)
}

# Chains that agree on the regimes: the largest group of an average-linkage
# clustering on 1 - agreement (best permutation), cut at 0.85. Chains with an
# empty regime are left out. Ties go to the group with the higher mean log-likelihood.
majority_chains <- function(fits, G, cut = 0.85) {
  Sm   <- lapply(fits, function(f) modal_states(f$S, G))
  full <- sapply(Sm, function(S) all(tabulate(S, G) > 0))
  if (G == 1 || sum(full) <= 1) return(which(full))
  P  <- all_perms(G)
  ag <- function(A, B) max(apply(P, 1, function(p) mean(A == matrix(p[B], nrow(B)))))
  idx <- which(full)
  AG  <- outer(idx, idx, Vectorize(function(i, j) ag(Sm[[i]], Sm[[j]])))
  cl  <- cutree(hclust(as.dist(1 - AG), method = "average"), h = 1 - cut)
  llm <- sapply(fits[idx], function(f) mean(f$loglik))
  sz  <- table(cl); mll <- tapply(llm, cl, mean)
  best <- as.integer(names(sz))[order(-as.integer(sz), -mll[names(sz)])][1]
  idx[cl == best]
}

# fits_by_G: one element per G, named "1", "2", ..., each a list of fits
# (output of run_mcmc_hmm() or run_mcmc_hmm_sr()) with that G.
select_G <- function(fits_by_G, Y_list, W_list, sr = FALSE) {
  rows <- lapply(names(fits_by_G), function(gname) {
    fits <- fits_by_G[[gname]]; G <- as.integer(gname)
    d   <- do.call(rbind, lapply(fits, dic_chain, G = G, Y_list = Y_list, W_list = W_list, sr = sr))
    ag  <- chain_agreement(fits, G)
    occ <- sapply(ag$Shat, function(S) min(tabulate(S, nbins = G)) / length(S))
    ll  <- sapply(fits, `[[`, "loglik")
    maj <- majority_chains(fits, G)
    kl  <- sapply(fits[maj], delta_hat_min_chain, G = G, Y_list = Y_list, W_list = W_list, sr = sr)
    data.frame(G = G, n_chains = length(fits), DIC_mean = mean(d$DIC), DIC_sd = sd(d$DIC),
               pD_min = min(d$pD), occ_min_mean = mean(occ),
               state_agreement_mean = if (length(fits) > 1) ag$mean else NA_real_,
               rhat_loglik = if (ncol(ll) > 1) posterior::rhat(ll) else NA_real_,
               ess_loglik_min = min(apply(ll, 2, posterior::ess_bulk)),
               n_majority = length(maj), Delta_hat_min_mean = if (length(kl)) mean(kl) else NA_real_)
  })
  tb  <- do.call(rbind, rows)
  agr <- ifelse(tb$G == 1 | is.na(tb$state_agreement_mean), 1, tb$state_agreement_mean)
  tb$eligible <- tb$pD_min >= 0 & tb$occ_min_mean > 0.10 & agr > 0.70
  tb$selected <- FALSE
  if (any(tb$eligible)) tb$selected[tb$eligible][which.min(tb$DIC_mean[tb$eligible])] <- TRUE
  tb
}