#' Naive PC1 Aggregation
#'
#' Standardizes the N×K score matrix, computes the sample correlation matrix,
#' extracts the leading eigenpair (PC1), derives regression scoring
#' coefficients, and scores each observation on the naive composite.
#'
#' @param S_hat N×K numeric matrix of imputed sub-index scores.
#'
#' @return A named list:
#'   \itemize{
#'     \item `col_means`: length-K vector of column means.
#'     \item `col_sds`: length-K vector of column standard deviations.
#'     \item `S_tilde`: N×K standardized score matrix.
#'     \item `R_obs`: K×K sample correlation matrix.
#'     \item `lambda1`: leading eigenvalue.
#'     \item `ell_hat`: length-K PC1 loading vector (all positive by
#'       convention).
#'     \item `w_hat`: length-K regression scoring coefficients
#'       (\eqn{\hat{\mathbf{w}} = \hat{\mathbf{R}}_\text{obs}^{-1}\hat{\boldsymbol{\ell}}}).
#'     \item `f_hat`: length-N naive composite scores.
#'     \item `var_explained`: proportion of variance explained by PC1.
#'   }
#' @importFrom stats sd cov
#' @export
naive_pca <- function(S_hat) {
  N <- nrow(S_hat)
  K <- ncol(S_hat)

  col_means <- colMeans(S_hat)
  col_sds   <- apply(S_hat, 2, sd)

  # Standardize
  S_tilde <- sweep(sweep(S_hat, 2, col_means, "-"), 2, col_sds, "/")

  # Sample correlation matrix
  R_obs <- (t(S_tilde) %*% S_tilde) / (N - 1)

  # Leading eigenpair
  ev      <- eigen(R_obs, symmetric = TRUE)
  lambda1 <- ev$values[1]
  ell_hat <- ev$vectors[, 1]

  # Sign convention: choose direction where sum of loadings is positive
  # (in practice all loadings are positive when sub-indices share a common factor)
  if (sum(ell_hat) < 0) ell_hat <- -ell_hat

  # Regression scoring coefficients: w = R_obs^{-1} * ell
  w_hat <- as.vector(solve(R_obs) %*% ell_hat)

  # Composite scores
  f_hat <- as.vector(S_tilde %*% w_hat)

  # Variance explained
  var_explained <- lambda1 / sum(ev$values)

  list(
    col_means    = col_means,
    col_sds      = col_sds,
    S_tilde      = S_tilde,
    R_obs        = R_obs,
    lambda1      = lambda1,
    ell_hat      = ell_hat,
    w_hat        = w_hat,
    f_hat        = f_hat,
    var_explained = var_explained
  )
}
