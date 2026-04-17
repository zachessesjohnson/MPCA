#' Two-Stage Bootstrap for CI Propagation
#'
#' Propagates both Stage-1 measurement error (perturbation stage) and
#' Stage-2 sampling uncertainty (row-resampling stage) via a two-stage
#' bootstrap (Algorithm 1 in the MPCA paper).
#'
#' @details
#' For each of the \eqn{B} replications:
#' \enumerate{
#'   \item **Perturbation**: draw \eqn{\mathbf{E}^{(b)} \sim
#'     \mathcal{N}(\mathbf{0}, (\mathbf{H}/1.96)^{\circ 2})}
#'     element-wise and clamp \eqn{\hat{\mathbf{S}} + \mathbf{E}^{(b)}} to
#'     \eqn{[0, 100]}.
#'   \item **Resampling**: draw row indices with replacement.
#'   \item **Scoring**: standardise using the original column means and
#'     signal SDs; project with \eqn{\hat{\mathbf{w}}^*}.
#'   \item **Row mapping**: for each original observation \eqn{i}, take the
#'     score from its *first* occurrence in the resample (NA if unsampled).
#' }
#' The 2.5th and 97.5th empirical quantiles of the resulting distribution
#' form the 95% CI.
#'
#' @param S_hat N×K matrix of imputed sub-index scores.
#' @param H N×K matrix of CI half-widths (0 where imputed).
#' @param col_means Length-K vector of column means (from [naive_pca()]).
#' @param w_hat_star Length-K corrected regression scoring coefficients.
#' @param sigma_sig Length-K signal standard deviations.
#' @param B Integer. Number of bootstrap replications (default 500).
#' @param seed Integer. Random seed for reproducibility (default 42).
#'
#' @return A list with:
#'   \itemize{
#'     \item `ci_lower`: length-N vector of 2.5th-percentile bootstrap
#'       scores.
#'     \item `ci_upper`: length-N vector of 97.5th-percentile bootstrap
#'       scores.
#'     \item `boot_scores`: N×B matrix of bootstrap scores (NA for
#'       unsampled observations).
#'   }
#' @importFrom stats rnorm quantile
#' @export
two_stage_bootstrap <- function(S_hat, H, col_means, w_hat_star, sigma_sig,
                                 B = 500L, seed = 42L) {
  set.seed(seed)
  N <- nrow(S_hat)
  K <- ncol(S_hat)

  sigma_e     <- H / 1.96
  boot_scores <- matrix(NA_real_, nrow = N, ncol = B)

  for (b in seq_len(B)) {
    # Stage 1: perturbation
    E      <- matrix(stats::rnorm(N * K), N, K) * sigma_e
    S_pert <- pmin(pmax(S_hat + E, 0), 100)

    # Stage 2: resample rows
    idx    <- sample.int(N, N, replace = TRUE)
    S_boot <- S_pert[idx, , drop = FALSE]

    # Standardise and score
    S_std  <- sweep(sweep(S_boot, 2, col_means, "-"), 2, sigma_sig, "/")
    f_boot <- as.vector(S_std %*% w_hat_star)

    # Map back to original observations (first-occurrence convention)
    for (i in seq_len(N)) {
      m_star <- match(i, idx)
      if (!is.na(m_star)) {
        boot_scores[i, b] <- f_boot[m_star]
      }
    }
  }

  ci_lower <- apply(boot_scores, 1, stats::quantile,
                    probs = 0.025, na.rm = TRUE)
  ci_upper <- apply(boot_scores, 1, stats::quantile,
                    probs = 0.975, na.rm = TRUE)

  list(ci_lower = ci_lower, ci_upper = ci_upper, boot_scores = boot_scores)
}
