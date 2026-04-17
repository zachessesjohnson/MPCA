#' Post-Process Composite Scores
#'
#' Applies the three-step post-processing pipeline to point estimates and CI
#' bounds: (1) global z-scoring, (2) Box-Cox transformation with
#' \eqn{\lambda = 0.5}, (3) min-max rescaling to \eqn{[0, 100]}.
#'
#' @details
#' **Step 1 – Z-score**: subtract the mean and divide by SD of the point
#' estimates.  The same shift and scale are applied to CI bounds.
#'
#' **Step 2 – Box-Cox** (\eqn{\lambda = 0.5}): scores are shifted so the
#' minimum exceeds zero (\eqn{x \leftarrow x + |\min(x)| + \epsilon}),
#' then \eqn{y = (x^\lambda - 1)/\lambda} is applied.  The same affine
#' shift (from the point estimates) is used for CI bounds before applying
#' the same power transform, so that bounds stay monotonically ordered
#' relative to the point estimate.
#'
#' **Step 3 – Min-max**: \eqn{s = 100 \cdot (y - \min(y)) /
#' (\max(y) - \min(y))}, using the min and max of the point estimates.
#' CI bounds are clamped to \eqn{[0, 100]}.
#'
#' @param f_hat_star Length-N numeric vector of corrected composite scores.
#' @param ci_lower Length-N numeric vector of 2.5th-percentile bootstrap
#'   scores.
#' @param ci_upper Length-N numeric vector of 97.5th-percentile bootstrap
#'   scores.
#' @param lambda Numeric. Box-Cox power parameter (default 0.5).
#' @param eps Numeric. Small constant to ensure positivity before Box-Cox
#'   (default 1e-6).
#'
#' @return A named list with:
#'   \itemize{
#'     \item `score`: length-N rescaled point estimates in \eqn{[0, 100]}.
#'     \item `ci_lower`: length-N rescaled lower CI bounds (clamped to
#'       \eqn{[0, 100]}).
#'     \item `ci_upper`: length-N rescaled upper CI bounds (clamped to
#'       \eqn{[0, 100]}).
#'   }
#' @export
postprocess_scores <- function(f_hat_star, ci_lower, ci_upper,
                                lambda = 0.5, eps = 1e-6) {
  # ---- Step 1: Z-score ----
  mu  <- mean(f_hat_star, na.rm = TRUE)
  sig <- stats::sd(f_hat_star)

  z_score <- (f_hat_star - mu) / sig
  z_lower <- (ci_lower   - mu) / sig
  z_upper <- (ci_upper   - mu) / sig

  # ---- Step 2: Box-Cox (lambda = 0.5) ----
  # Shift to strictly positive domain using min of point estimates
  shift   <- abs(min(z_score, na.rm = TRUE)) + eps
  x_score <- z_score + shift
  x_lower <- z_lower + shift
  x_upper <- z_upper + shift

  box_cox <- function(x) (pmax(x, 0) ^ lambda - 1) / lambda

  bc_score <- box_cox(x_score)
  bc_lower <- box_cox(x_lower)
  bc_upper <- box_cox(x_upper)

  # ---- Step 3: Min-max rescale to [0, 100] ----
  bc_min <- min(bc_score, na.rm = TRUE)
  bc_max <- max(bc_score, na.rm = TRUE)
  rng    <- bc_max - bc_min

  rescale <- function(x) 100 * (x - bc_min) / rng

  score    <- rescale(bc_score)
  ci_lo    <- pmin(pmax(rescale(bc_lower), 0), 100)
  ci_up    <- pmin(pmax(rescale(bc_upper), 0), 100)

  list(score = score, ci_lower = ci_lo, ci_upper = ci_up)
}
