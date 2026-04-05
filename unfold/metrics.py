"""Evaluation metrics: Wasserstein distance and VLC divergence."""

import numpy as np
from scipy.stats import wasserstein_distance


def vlc_divergence(p_samples, q_samples, p_weights=None, q_weights=None, n_bins=100):
    """Compute Vincze-Le Cam (triangular discriminator) divergence.

    Delta(p,q) = 0.5 * integral((p-q)^2 / (p+q)) dx
    """
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    edges = np.linspace(lo, hi, n_bins + 1)

    p_hist = np.histogram(p_samples, bins=edges, weights=p_weights, density=True)[0]
    q_hist = np.histogram(q_samples, bins=edges, weights=q_weights, density=True)[0]

    denom = p_hist + q_hist
    mask = denom > 0
    vlc = 0.5 * np.sum((p_hist[mask] - q_hist[mask]) ** 2 / denom[mask]) * (edges[1] - edges[0])
    return vlc


def compute_metrics(truth_samples, test_samples, weights=None):
    """Compute Wasserstein distance and VLC divergence per observable."""
    n_vars = truth_samples.shape[1]
    wd = np.array([
        wasserstein_distance(truth_samples[:, i], test_samples[:, i],
                             v_weights=weights)
        for i in range(n_vars)
    ])
    vlc = np.array([
        vlc_divergence(truth_samples[:, i], test_samples[:, i],
                       q_weights=weights)
        for i in range(n_vars)
    ])
    return wd, vlc
