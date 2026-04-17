#' Attenuation Correction via Disattenuation of Correlations
#'
#' Removes measurement-error variance from the observed covariance matrix,
#' constructs a corrected (signal) correlation matrix, and re-runs PCA on it.
#' This implements the Spearman (1904) disattenuation formula generalised to
#' the sub-index setting, following the methodology in the MPCA paper.
#'
#' @details
#' The error variance for sub-index \eqn{j} is estimated as
#' \deqn{\hat{\psi}_j = \frac{1}{N}\sum_i \left(\frac{h_{ij}}{1.96}\right)^2,}
#' where \eqn{h_{ij}} is the CI half-width (0 for imputed positions).
#' The signal covariance is then
#' \deqn{\hat{\boldsymbol{\Sigma}}_\text{sig} =
#'   \hat{\boldsymbol{\Sigma}}_\text{obs} - \text{diag}(\hat{\psi}_1,\ldots,\hat{\psi}_K),}
#' with diagonal elements floored at \eqn{0.01 \cdot \hat{\Sigma}_{\text{obs},jj}}
#' to ensure positive definiteness.  The corrected correlation matrix
#' \eqn{\hat{\mathbf{R}}_\text{sig}} is projected to the nearest positive
#' semidefinite matrix via eigenvalue flooring at \eqn{10^{-6}} if needed.
#'
#' @param S_hat N×K matrix of imputed sub-index scores.
#' @param H N×K matrix of CI half-widths (0 where imputed).
#' @param naive_result List returned by [naive_pca()].
#'
#' @return A named list:
#'   \itemize{
#'     \item `psi_hat`: length-K error variance estimates.
#'     \item `reliability`: length-K reliability estimates
#'       \eqn{r_j = 1 - \hat{\psi}_j / \Sigma_{\text{obs},jj}}.
#'     \item `Sigma_obs`: K×K observed covariance matrix.
#'     \item `Sigma_sig`: K×K signal covariance matrix.
#'     \item `R_sig`: K×K corrected correlation matrix.
#'     \item `lambda1_star`: leading eigenvalue of \eqn{\hat{\mathbf{R}}_\text{sig}}.
#'     \item `ell_hat_star`: length-K corrected PC1 loadings.
#'     \item `w_hat_star`: length-K corrected regression scoring coefficients.
#'     \item `f_hat_star`: length-N corrected composite scores.
#'     \item `sigma_sig`: length-K signal standard deviations.
#'     \item `var_explained_star`: PC1 variance share for
#'       \eqn{\hat{\mathbf{R}}_\text{sig}}.
#'   }
#' @importFrom Matrix nearPD
#' @importFrom stats cov
#' @export
attenuation_correction <- function(S_hat, H, naive_result) {
  N <- nrow(S_hat)
  K <- ncol(S_hat)

  col_means <- naive_result$col_means
  ell_hat   <- naive_result$ell_hat

  # ---- Error variance estimates ----
  # psi_hat_j = (1/N) * sum_i (h_ij / 1.96)^2
  sigma_e <- H / 1.96
  psi_hat <- colMeans(sigma_e ^ 2)

  # ---- Observed covariance (original scale) ----
  Sigma_obs <- stats::cov(S_hat)

  # ---- Signal covariance: subtract error diagonal ----
  Sigma_sig <- Sigma_obs
  diag(Sigma_sig) <- diag(Sigma_obs) - psi_hat

  # Regularise diagonal (floor at 1% of observed variance)
  for (j in seq_len(K)) {
    Sigma_sig[j, j] <- max(Sigma_sig[j, j], 0.01 * Sigma_obs[j, j])
  }

  # ---- Reliability ----
  reliability <- 1 - psi_hat / diag(Sigma_obs)

  # ---- Signal standard deviations ----
  sigma_sig <- sqrt(diag(Sigma_sig))

  # ---- Corrected correlation matrix ----
  R_sig <- Sigma_sig / outer(sigma_sig, sigma_sig)
  diag(R_sig) <- 1

  # ---- Project to nearest PSD if needed ----
  min_ev <- min(eigen(R_sig, symmetric = TRUE, only.values = TRUE)$values)
  if (min_ev < 1e-8) {
    R_sig <- as.matrix(
      Matrix::nearPD(R_sig, corr = TRUE, eig.tol = 1e-6)$mat
    )
  }

  # ---- PCA on R_sig ----
  ev_star      <- eigen(R_sig, symmetric = TRUE)
  lambda1_star <- ev_star$values[1]
  ell_hat_star <- ev_star$vectors[, 1]

  # Sign convention: align with naive loadings
  if (sum(ell_hat_star * ell_hat) < 0) ell_hat_star <- -ell_hat_star

  # ---- Corrected scoring coefficients ----
  w_hat_star <- as.vector(solve(R_sig) %*% ell_hat_star)

  # ---- Re-score: original col means, signal SDs ----
  S_tilde_sig <- sweep(sweep(S_hat, 2, col_means, "-"), 2, sigma_sig, "/")
  f_hat_star  <- as.vector(S_tilde_sig %*% w_hat_star)

  # ---- Variance explained ----
  var_explained_star <- lambda1_star / sum(ev_star$values)

  list(
    psi_hat          = psi_hat,
    reliability      = reliability,
    Sigma_obs        = Sigma_obs,
    Sigma_sig        = Sigma_sig,
    R_sig            = R_sig,
    lambda1_star     = lambda1_star,
    ell_hat_star     = ell_hat_star,
    w_hat_star       = w_hat_star,
    f_hat_star       = f_hat_star,
    sigma_sig        = sigma_sig,
    var_explained_star = var_explained_star
  )
}
