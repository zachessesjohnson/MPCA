"""Attenuation correction via disattenuation of correlations."""

import numpy as np


def attenuation_correction(
    S_hat: np.ndarray,
    H: np.ndarray,
    naive_result: dict,
) -> dict:
    """
    Remove measurement-error variance and rerun PCA on the corrected matrix.

    Implements the MPCA attenuation-correction methodology:

    1. Estimate per-column error variance from CI half-widths:
       :math:`\\hat{\\psi}_j = N^{-1}\\sum_i (h_{ij}/1.96)^2`.
    2. Recover signal covariance:
       :math:`\\hat{\\boldsymbol{\\Sigma}}_{\\text{sig}} =
       \\hat{\\boldsymbol{\\Sigma}}_{\\text{obs}} -
       \\operatorname{diag}(\\hat{\\psi}_1,\\ldots,\\hat{\\psi}_K)`,
       with diagonal floored at :math:`0.01\\hat{\\Sigma}_{\\text{obs},jj}`.
    3. Form corrected correlation :math:`\\hat{\\mathbf{R}}_{\\text{sig}}`;
       project to nearest positive-semidefinite matrix if needed
       (eigenvalue floor at :math:`10^{-6}`).
    4. Re-run PCA on :math:`\\hat{\\mathbf{R}}_{\\text{sig}}`; align sign
       with naive loadings.

    Parameters
    ----------
    S_hat : np.ndarray, shape (N, K)
        Imputed sub-index score matrix.
    H : np.ndarray, shape (N, K)
        CI half-width matrix (0 for imputed positions).
    naive_result : dict
        Output from :func:`naive_pca`.

    Returns
    -------
    dict with keys:

    ``psi_hat`` : np.ndarray, shape (K,)
        Error variance estimates.
    ``reliability`` : np.ndarray, shape (K,)
        :math:`r_j = 1 - \\hat{\\psi}_j / \\Sigma_{\\text{obs},jj}`.
    ``Sigma_obs`` : np.ndarray, shape (K, K)
        Observed covariance matrix.
    ``Sigma_sig`` : np.ndarray, shape (K, K)
        Signal covariance matrix.
    ``R_sig`` : np.ndarray, shape (K, K)
        Corrected correlation matrix (positive definite).
    ``lambda1_star`` : float
        Leading eigenvalue of :math:`\\hat{\\mathbf{R}}_{\\text{sig}}`.
    ``ell_hat_star`` : np.ndarray, shape (K,)
        Corrected PC1 loadings.
    ``w_hat_star`` : np.ndarray, shape (K,)
        Corrected regression scoring coefficients.
    ``f_hat_star`` : np.ndarray, shape (N,)
        Corrected composite scores.
    ``sigma_sig`` : np.ndarray, shape (K,)
        Signal standard deviations.
    ``var_explained_star`` : float
        PC1 variance share for :math:`\\hat{\\mathbf{R}}_{\\text{sig}}`.
    """
    N, K = S_hat.shape
    col_means = naive_result["col_means"]
    ell_hat = naive_result["ell_hat"]

    # ---- Error variance estimates ----
    sigma_e = H / 1.96
    psi_hat = np.mean(sigma_e ** 2, axis=0)

    # ---- Observed covariance ----
    Sigma_obs = np.cov(S_hat, rowvar=False)

    # ---- Signal covariance ----
    Sigma_sig = Sigma_obs.copy()
    np.fill_diagonal(Sigma_sig, np.diag(Sigma_obs) - psi_hat)

    # Regularise diagonal
    for j in range(K):
        Sigma_sig[j, j] = max(Sigma_sig[j, j], 0.01 * Sigma_obs[j, j])

    # ---- Reliability ----
    reliability = 1.0 - psi_hat / np.diag(Sigma_obs)

    # ---- Signal standard deviations ----
    sigma_sig = np.sqrt(np.diag(Sigma_sig))

    # ---- Corrected correlation matrix ----
    outer_sig = np.outer(sigma_sig, sigma_sig)
    R_sig = Sigma_sig / outer_sig
    np.fill_diagonal(R_sig, 1.0)

    # ---- Nearest PSD projection if needed ----
    min_ev = np.linalg.eigvalsh(R_sig).min()
    if min_ev < 1e-8:
        R_sig = _nearest_psd_corr(R_sig)

    # ---- PCA on R_sig ----
    eigenvalues, eigenvectors = np.linalg.eigh(R_sig)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda1_star = eigenvalues[0]
    ell_hat_star = eigenvectors[:, 0]

    # Sign convention: align with naive loadings
    if np.dot(ell_hat_star, ell_hat) < 0:
        ell_hat_star = -ell_hat_star

    # ---- Corrected scoring coefficients ----
    w_hat_star = np.linalg.solve(R_sig, ell_hat_star)

    # ---- Re-score: original col means, signal SDs ----
    S_tilde_sig = (S_hat - col_means) / sigma_sig
    f_hat_star = S_tilde_sig @ w_hat_star

    var_explained_star = lambda1_star / eigenvalues.sum()

    return {
        "psi_hat": psi_hat,
        "reliability": reliability,
        "Sigma_obs": Sigma_obs,
        "Sigma_sig": Sigma_sig,
        "R_sig": R_sig,
        "lambda1_star": lambda1_star,
        "ell_hat_star": ell_hat_star,
        "w_hat_star": w_hat_star,
        "f_hat_star": f_hat_star,
        "sigma_sig": sigma_sig,
        "var_explained_star": var_explained_star,
    }


def _nearest_psd_corr(R: np.ndarray, eig_tol: float = 1e-6) -> np.ndarray:
    """Project *R* to the nearest positive-semidefinite correlation matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = np.maximum(eigenvalues, eig_tol)
    R_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Rescale to enforce unit diagonal
    d = np.sqrt(np.diag(R_psd))
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd
