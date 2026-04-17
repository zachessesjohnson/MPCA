# mpca — Python Package

Multilevel PCA (MPCA) with measurement-error correction for composite indices.

## Installation

```bash
pip install -e python/
```

## Overview

MPCA corrects the attenuation bias that arises when sub-index scores used as
inputs to PCA are themselves noisy estimates with known uncertainty from
bootstrapped confidence intervals. The package implements the full pipeline:

1. **Option Filter** – exclude observations with fewer than 5 (configurable)
   of the K sub-indices observed.
2. **Three-pass imputation** – within-group linear interpolation →
   time-period-mean fallback → global-mean fallback.
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

```python
import pandas as pd
from mpca import mpca_pipeline

# my_data: a DataFrame with score, lower CI, upper CI, and identifier columns
result = mpca_pipeline(
    data=my_data,
    score_cols=[f"s{k}" for k in range(1, 9)],
    lower_cols=[f"s{k}_lower" for k in range(1, 9)],
    upper_cols=[f"s{k}_upper" for k in range(1, 9)],
    id_cols=["unit", "time"],   # identifier columns to carry through to output
    group_col="unit",           # for within-group interpolation (Pass 1)
    time_col="time",            # for time-mean fallback (Pass 2) and rankings
    B=500,
    seed=42,
    rankings_value=5,           # filter scores_df to time==5 for rankings table
)

# Composite scores
print(result["scores_df"].head())

# Sub-index loadings and weights
print(result["contributions_df"])

# Rankings snapshot
print(result["rankings_df"])
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

```bash
cd python
pytest tests/ -v
```
