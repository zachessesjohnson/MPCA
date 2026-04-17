# Changelog

## [0.1.0] — 2026-04-17

Initial release.

### Added

- `mpca_pipeline()` — end-to-end pipeline returning composite scores,
  sub-index contributions, and optional year-specific rankings.
- `option_b_filter()` — Option B minimum-coverage filter.
- `three_pass_imputation()` — three-pass missing-data hierarchy
  (within-country linear interpolation, year-mean fallback, global-mean
  fallback).
- `naive_pca()` — naive PC1 aggregation on the observed correlation matrix.
- `attenuation_correction()` — Spearman (1904) disattenuation correction:
  estimates per-sub-index error variances from CI half-widths, recovers the
  signal covariance, and re-runs PCA on the corrected correlation matrix
  with nearest positive-semidefinite projection.
- `two_stage_bootstrap()` — two-stage bootstrap propagating both
  measurement error and sampling uncertainty into 95% composite CIs.
- `postprocess_scores()` — z-score → Box-Cox (λ = 0.5) → min-max
  rescaling to [0, 100].
