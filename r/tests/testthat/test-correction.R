make_test_data <- function(N = 50, K = 8, seed = 1) {
  set.seed(seed)
  S <- matrix(runif(N * K, 20, 80), N, K)
  H <- matrix(runif(N * K, 1, 10), N, K)
  list(S = S, H = H)
}

test_that("naive_pca returns correctly-shaped outputs", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)

  K <- ncol(td$S)
  N <- nrow(td$S)

  expect_length(naive$col_means, K)
  expect_length(naive$col_sds,   K)
  expect_equal(dim(naive$S_tilde), c(N, K))
  expect_equal(dim(naive$R_obs),   c(K, K))
  expect_length(naive$ell_hat,  K)
  expect_length(naive$w_hat,    K)
  expect_length(naive$f_hat,    N)
  expect_true(naive$var_explained > 0 && naive$var_explained <= 1)
})

test_that("naive_pca loadings are all positive", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)
  expect_true(all(naive$ell_hat > 0))
})

test_that("attenuation_correction returns corrected objects", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)
  corr  <- attenuation_correction(td$S, td$H, naive)

  K <- ncol(td$S)
  N <- nrow(td$S)

  expect_length(corr$psi_hat,      K)
  expect_length(corr$reliability,  K)
  expect_length(corr$sigma_sig,    K)
  expect_equal(dim(corr$Sigma_sig), c(K, K))
  expect_equal(dim(corr$R_sig),     c(K, K))
  expect_length(corr$ell_hat_star, K)
  expect_length(corr$w_hat_star,   K)
  expect_length(corr$f_hat_star,   N)
})

test_that("corrected correlation matrix is positive definite", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)
  corr  <- attenuation_correction(td$S, td$H, naive)

  ev <- eigen(corr$R_sig, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(ev > 0))
})

test_that("signal variance is smaller than observed variance", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)
  corr  <- attenuation_correction(td$S, td$H, naive)

  expect_true(all(diag(corr$Sigma_sig) <= diag(corr$Sigma_obs) + 1e-10))
})

test_that("reliability values are in (0, 1]", {
  td    <- make_test_data()
  naive <- naive_pca(td$S)
  corr  <- attenuation_correction(td$S, td$H, naive)

  expect_true(all(corr$reliability > 0))
  expect_true(all(corr$reliability <= 1 + 1e-10))
})
