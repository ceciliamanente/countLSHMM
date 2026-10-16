# countLSHMM

This repository contains the code for *A latent space hidden Markov model for clustering views in longitudinal multilayer networks* (Manente, C., Alfò, M., and D'Angelo, S.).

## The model

Directed count networks are observed on K layers at T time points. Each slice (t, k) belongs to one of G latent regimes, which evolve as a Markov chain within each layer. Each regime has its own latent space, and edge counts follow a hurdle negative binomial:

- **base model**: `log mu_ij = alpha_tk - ||z_i - z_j||`
- **sender-receiver model**: `log mu_ij = alpha_tk (u_i + v_j)/2 - ||z_i - z_j||`, identified by a reference node at the median degree

with `logit pi_tk = beta_tk` for the probability that an edge is active. Inference is by MCMC: forward-filtering backward-sampling for the regimes and adaptive Metropolis-Hastings for the other parameters. The number of regimes G is chosen from the realigned DIC, the minimum regime occupancy and the agreement between chains. The empirical separation between regimes (KL) is also reported.

## Repository structure

- `cpp/` — C++ source code for the MCMC samplers (interfaced with R via RcppArmadillo)
- `R/` — R scripts to initialise, fit the models and choose G

| File | Description |
|------|-------------|
| `cpp/mcmc_hmm.cpp` | Sampler, base model |
| `cpp/mcmc_hmm_sr.cpp` | Sampler, sender-receiver model |
| `R/run_base.R` | Fits the base model for G = 1, ..., G_max and chooses G |
| `R/run_sr.R` | Fits the sender-receiver model for G = 1, ..., G_max and chooses G |
| `R/init.R` | Initialisation of the chains |
| `R/ecr_utils.R` | ECR relabelling of the regimes |
| `R/select_G.R` | Realigned DIC, minimum occupancy, agreement between chains, empirical KL separation, choice of G |

## Installation

The code requires R (>= 4.3) with a C++ compiler; OpenMP is optional. Install the R packages:

```r
install.packages(c("Rcpp", "RcppArmadillo", "mclust", "posterior"))
```

The samplers are compiled at run time with `Rcpp::sourceCpp()`.

## Data availability

The data analysed in the thesis are not included in this repository.
