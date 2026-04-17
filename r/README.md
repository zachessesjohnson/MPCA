# mpca — R Package

Multilevel PCA (MPCA) for composite indices with measurement-error correction.

## Installation

```r
# Install from local source
devtools::install("r/")
```

## Overview

MPCA corrects the attenuation bias that arises when sub-index scores used as
inputs to PCA are themselves noisy estimates with known uncertainty (e.g. from
bootstrapped confidence intervals). The package implements the full pipeline
described in the MPCA technical paper:

1. **Option Filter** – exclude observations with fewer than 5 (configurable)
   of the K sub-indices observed.
2. **Three-pass imputation** – within-country linear interpolation →
   year-mean fallback → global-mean fallback.
3. **Naive PCA** – PC1 loadings and regression scoring coefficients on the
   observed correlation matrix `R_obs`.
4. **Attenuation correction** – estimate per-column error variances from CI
   half-widths; subtract from the observed covariance to recover the signal
   covariance; form the corrected correlation matrix `R_sig`; re-run PCA.
5. **Two-stage bootstrap** – perturb sub-index scores with Gaussian noise
   scaled to their standard errors, then resample rows; propagate both
   sources of uncertainty into 95% CIs.
6. **Post-processing** – z-score → Box-Cox (λ = 0.5) → min-max rescale
   to [0, 100].

## Quick Start

```r
library(mpca)

# data: a data.frame with columns for sub-index scores, lower/upper CI bounds,
#       country, iso, and year.
result <- mpca_pipeline(
  data       = my_data,
  score_cols = paste0("s", 1:8),
  lower_cols = paste0("s", 1:8, "_lower"),
  upper_cols = paste0("s", 1:8, "_upper"),
  country_col = "country",
  iso_col     = "iso",
  year_col    = "year",
  B           = 500L,
  seed        = 42L
)

# Composite scores
head(result$scores_df)

# Sub-index loadings and weights
result$contributions_df

# 2024 rankings
result$rankings_df
```

## Key Functions

| Function | Description |
|---|---|
| `mpca_pipeline()` | End-to-end pipeline; main entry point |
| `option_b_filter()` | Flag rows below minimum sub-index coverage |
| `three_pass_imputation()` | Three-pass missing-data imputation |
| `naive_pca()` | Naive PC1 aggregation |
| `attenuation_correction()` | Disattenuation of correlations, corrected PCA |
| `two_stage_bootstrap()` | Two-stage CI propagation |
| `postprocess_scores()` | Z-score → Box-Cox → min-max rescaling |

## Running Tests

```r
devtools::test("r/")
```
