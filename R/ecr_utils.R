# Label switching: ECR relabelling of the regimes and Procrustes rotation.

# All permutations of 1..G, one per row.
all_perms <- function(G) {
  if (G == 1) return(matrix(1L, 1, 1))
  do.call(rbind, lapply(seq_len(G), function(i) {
    sub <- all_perms(G - 1L)
    rest <- setdiff(seq_len(G), i)
    cbind(i, matrix(rest[sub], nrow = nrow(sub)))
  }))
}

# For each draw, the permutation of the labels that agrees most with S_ref
# (T x K, labels 1..G). Returns S, Z, gamma and delta relabelled, and the
# permutations used.
relabel_ecr <- function(fit, S_ref, G) {
  P   <- all_perms(G)
  M   <- dim(fit$S)[3]
  out <- list(S = fit$S, Z = fit$Z, gamma = fit$gamma, delta = fit$delta,
              perm = matrix(0L, M, G))
  for (m in seq_len(M)) {
    Sm <- fit$S[, , m] + 1L
    p  <- P[which.max(apply(P, 1, function(p) sum(p[Sm] == S_ref))), ]
    inv <- order(p)
    out$perm[m, ]    <- p
    out$S[, , m]     <- matrix(p[Sm], nrow = nrow(Sm)) - 1L
    out$Z[[m]]       <- fit$Z[[m]][inv]
    out$gamma[, , m] <- fit$gamma[inv, inv, m]
    out$delta[, , m] <- fit$delta[, inv, m]
  }
  out
}

# Rotates A to match B.
procrustes <- function(A, B) {
  s <- svd(crossprod(A, B))
  A %*% (s$u %*% t(s$v))
}
