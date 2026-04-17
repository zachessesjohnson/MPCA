"""
mpca — Multilevel PCA with measurement error correction.

Public API
----------
mpca_pipeline          End-to-end pipeline
option_b_filter        Option Filter minimum-coverage filter
three_pass_imputation  Three-pass missing-data imputation
naive_pca              Naive PC1 aggregation
attenuation_correction Disattenuation of correlations + corrected PCA
two_stage_bootstrap    Two-stage CI propagation bootstrap
postprocess_scores     Z-score → Box-Cox → min-max rescaling
"""

from .imputation import option_b_filter, three_pass_imputation
from .naive_pca import naive_pca
from .correction import attenuation_correction
from .bootstrap import two_stage_bootstrap
from .postprocess import postprocess_scores
from .pipeline import mpca_pipeline

__all__ = [
    "mpca_pipeline",
    "option_b_filter",
    "three_pass_imputation",
    "naive_pca",
    "attenuation_correction",
    "two_stage_bootstrap",
    "postprocess_scores",
]
