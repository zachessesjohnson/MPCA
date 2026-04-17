"""Naive PC1 aggregation on the observed correlation matrix."""

import numpy as np


def naive_pca(S_hat: np.ndarray) -> dict:
    """
    Standardize sub-index scores and extract the leading PC.

    Parameters
    ----------
    S_hat : np.ndarray, shape (N, K)
        Imputed sub-index score matrix.

    Returns
    -------
    dict with keys:

    ``col_means`` : np.ndarray, shape (K,)
        Column means of *S_hat*.
    ``col_sds`` : np.ndarray, shape (K,)
        Column standard deviations of *S_hat*.
    ``S_tilde`` : np.ndarray, shape (N, K)
        Standardized score matrix.
    ``R_obs`` : np.ndarray, shape (K, K)
        Sample correlation matrix :math:`(N-1)^{-1}\\tilde{S}'\\tilde{S}`.
    ``lambda1`` : float
        Leading eigenvalue.
    ``ell_hat`` : np.ndarray, shape (K,)
        PC1 loadings (all positive by sign convention).
    ``w_hat`` : np.ndarray, shape (K,)
        Regression scoring coefficients
        :math:`\\hat{\\mathbf{w}} = \\hat{\\mathbf{R}}_{\\text{obs}}^{-1}\\hat{\\boldsymbol{\\ell}}`.
    ``f_hat`` : np.ndarray, shape (N,)
        Naive composite scores :math:`\\tilde{S}\\hat{\\mathbf{w}}`.
    ``var_explained`` : float
        Proportion of variance explained by PC1.
    """
    N, K = S_hat.shape

    col_means = S_hat.mean(axis=0)
    col_sds = S_hat.std(axis=0, ddof=1)

    S_tilde = (S_hat - col_means) / col_sds

    # Sample correlation matrix
    R_obs = (S_tilde.T @ S_tilde) / (N - 1)

    # Leading eigenpair (eigh returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(R_obs)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda1 = eigenvalues[0]
    ell_hat = eigenvectors[:, 0]

    # Sign convention: sum of loadings positive
    if ell_hat.sum() < 0:
        ell_hat = -ell_hat

    # Regression scoring coefficients
    w_hat = np.linalg.solve(R_obs, ell_hat)

    # Composite scores
    f_hat = S_tilde @ w_hat

    var_explained = lambda1 / eigenvalues.sum()

    return {
        "col_means": col_means,
        "col_sds": col_sds,
        "S_tilde": S_tilde,
        "R_obs": R_obs,
        "lambda1": lambda1,
        "ell_hat": ell_hat,
        "w_hat": w_hat,
        "f_hat": f_hat,
        "var_explained": var_explained,
    }
