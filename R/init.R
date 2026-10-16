# Initial values for the LS-HMM samplers.
#   make_inits_pos()      base model
#   make_inits_sr()       sender-receiver (SR) model
#   prior_sr_from_base()  SR prior equivalent to a base-model prior


# Centred log-counts, one row per off-diagonal dyad and one column per slice
slice_features <- function(Y_list, W_list) {
  K <- length(Y_list); n <- dim(Y_list[[1]])[1]; T <- dim(Y_list[[1]])[3]
  off <- which(row(matrix(0, n, n)) != col(matrix(0, n, n)))
  V <- matrix(NA_real_, length(off), T * K)
  for (k in seq_len(K)) for (t in seq_len(T)) {
    y <- Y_list[[k]][, , t][off]
    w <- W_list[[k]][, , t][off]
    v <- ifelse(w > 0.5 & y > 0, log(y), NA_real_)
    V[, (t - 1) * K + k] <- v - mean(v, na.rm = TRUE)
  }
  V
}

# Correlation between slices on the dyads observed in both, and the number of such dyads.
slice_similarity <- function(V) {
  S <- suppressWarnings(cor(V, use = "pairwise.complete.obs"))
  S[!is.finite(S)] <- 0
  obs <- !is.na(V)
  list(cor = S, n_shared = crossprod(obs))
}

# Base model. 
make_inits_pos <- function(Y_list, W_list, G, d = 2, method = c("corr", "spec"), seed = 1L) {
  method <- match.arg(method); set.seed(seed)
  K <- length(Y_list); n <- dim(Y_list[[1]])[1]; T <- dim(Y_list[[1]])[3]
  off <- which(row(matrix(0, n, n)) != col(matrix(0, n, n)))
  
  V  <- slice_features(Y_list, W_list)
  sm <- slice_similarity(V)
  
  cl <- if (method == "corr") {
    cutree(hclust(as.dist(1 - sm$cor), method = "average"), k = G)
  } else {
    A  <- pmax(sm$cor, 0); diag(A) <- 0
    dg <- pmax(rowSums(A), 1e-9)
    L  <- diag(length(dg)) - diag(1 / sqrt(dg)) %*% A %*% diag(1 / sqrt(dg))
    U  <- eigen(L, symmetric = TRUE)$vectors
    U  <- U[, (ncol(U) - G + 1):ncol(U), drop = FALSE]
    U  <- U / pmax(sqrt(rowSums(U^2)), 1e-9)
    kmeans(U, centers = G, nstart = 100, iter.max = 200)$cluster
  }
  
  S <- matrix(0L, T, K)
  for (i in seq_len(T * K)) S[(i - 1) %/% K + 1, (i - 1) %% K + 1] <- cl[i] - 1L
  
  # Z. A regime with no slices (or no positive dyads) starts from the overall mean.
  global_m <- rowMeans(V, na.rm = TRUE)
  global_m[!is.finite(global_m)] <- 0
  Dbar <- Z <- vector("list", G)
  for (g in seq_len(G)) {
    sl <- which(cl == g)
    if (!length(sl)) {
      m <- global_m
    } else {
      m <- rowMeans(V[, sl, drop = FALSE], na.rm = TRUE)
      m[!is.finite(m)] <- mean(m[is.finite(m)], na.rm = TRUE)
      if (!any(is.finite(m))) m <- global_m
    }
    M  <- matrix(0, n, n); M[off] <- m
    M  <- (M + t(M)) / 2                       # D is symmetric
    Dh <- max(M) - M; diag(Dh) <- 0            # -D + const  ->  D + const
    Dh <- Dh - min(Dh[off]); diag(Dh) <- 0
    Dbar[[g]] <- Dh
    z <- cmdscale(as.dist(Dh), k = d)
    if (ncol(z) < d) z <- cbind(z, matrix(0, n, d - ncol(z)))
    Z[[g]] <- sweep(z, 2, colMeans(z))
  }
  
  # alpha and beta on the observed slices. Slices with no observed dyads (or,
  # for alpha, no positive dyads) take the mean of their regime, or the
  # overall mean if the regime has none; their value does not affect the fit.
  raw_alpha <- raw_beta <- matrix(NA_real_, T, K)
  for (k in seq_len(K)) for (t in seq_len(T)) {
    y <- Y_list[[k]][, , t][off]; w <- W_list[[k]][, , t][off]
    if (!any(w > 0.5)) next
    ok <- w > 0.5 & y > 0
    if (any(ok)) raw_alpha[t, k] <- mean(log(y[ok])) + mean(Dbar[[S[t, k] + 1L]][off])
    raw_beta[t, k] <- qlogis(min(max(mean(ok[w > 0.5]), 1e-3), 1 - 1e-3))
  }
  fill <- function(raw) {
    out <- raw
    overall <- if (any(is.finite(raw))) mean(raw, na.rm = TRUE) else 0
    for (g in seq_len(G)) {
      in_g <- S == g - 1L
      fg <- if (any(is.finite(raw[in_g]))) mean(raw[in_g], na.rm = TRUE) else overall
      out[in_g & !is.finite(raw)] <- fg
    }
    out
  }
  alpha <- fill(raw_alpha)
  beta  <- fill(raw_beta)
  
  list(Z = Z, S = S, alpha = alpha, beta = beta,
       r = matrix(1, T, K),
       gamma = matrix(0, G, G), delta = matrix(0, K, G),
       sim = sm, cluster = cl)
}

# SR model. 
ALPHA_FLOOR <- 1e-2

make_inits_sr <- function(Y_list, W_list, G, d = 2, method = c("corr", "spec"), seed = 1L) {
  ini <- make_inits_pos(Y_list, W_list, G, d = d, method = match.arg(method), seed = seed)
  n  <- dim(Y_list[[1]])[1]
  a0 <- ini$alpha
  sg <- sign(stats::median(a0)); if (sg == 0) sg <- 1
  c0 <- sg / 2
  
  alpha <- a0 / c0
  bad   <- !(alpha > ALPHA_FLOOR)
  if (any(bad)) {
    warning(sprintf("make_inits_sr: alpha_init set to %g in %d of %d slices",
                    ALPHA_FLOOR, sum(bad), length(alpha)))
    alpha[bad] <- ALPHA_FLOOR
  }
  ini$alpha <- alpha
  ini$u     <- rep(0, n)
  ini$v     <- rep(2 * c0, n)
  ini
}


prior_sr_from_base <- function(priors_base, sigma_u = 1, sigma_v = 1) {
  stopifnot(priors_base$mu_alpha > 0)
  p <- priors_base
  p$mu_alpha    <- 2 * priors_base$mu_alpha
  p$sigma_alpha <- 2 * priors_base$sigma_alpha
  p$sigma_u     <- sigma_u
  p$sigma_v     <- sigma_v
  p
}