"""
Prototype learning for normal and abnormal feature distributions (Eq. 3).

Provides two K-Means variants:
  - ``spherical_kmeans``  – operates on the unit hypersphere (cosine
                            similarity, Fréchet-mean update).
  - ``euclidean_kmeans``  – standard Euclidean K-Means (used as a
                            warm-up or baseline reference).
"""

import torch
import torch.nn.functional as F


def spherical_kmeans(
    features: torch.Tensor,
    k:        int,
    n_iter:   int = 100,
) -> torch.Tensor:
    """
    Spherical K-Means on S^{d-1}  (Eq. 3).

    Assignment uses **maximum cosine similarity**; the centroid update
    is the normalised mean of assigned vectors (spherical Fréchet mean
    approximation for one iteration).

    Parameters
    ----------
    features : (N, d) unit-norm feature matrix.
    k        : number of prototype clusters.
    n_iter   : maximum number of Lloyd iterations.

    Returns
    -------
    centers : (K, d) unit-norm prototype vectors.
              If k >= N, returns a copy of *features* directly.
    """
    N = features.size(0)
    if k >= N:
        return features.clone()

    # Random initialisation (K-Means++ could replace this)
    indices = torch.randperm(N, device=features.device)[:k]
    centers = features[indices].clone()   # (K, d)

    for _ in range(n_iter):
        # --- Assignment: argmax cosine similarity ---
        sims        = torch.matmul(features, centers.T)   # (N, K)
        assignments = sims.argmax(dim=1)                   # (N,)

        # --- Update: normalised mean per cluster ---
        new_centers = torch.zeros_like(centers)
        for c in range(k):
            mask = (assignments == c)
            if mask.sum() > 0:
                new_centers[c] = F.normalize(
                    features[mask].mean(dim=0), dim=0
                )
            else:
                # Keep old centroid for empty clusters
                new_centers[c] = centers[c]

        # --- Convergence check ---
        shift = (new_centers - centers).norm(dim=1).max().item()
        centers = new_centers
        if shift < 1e-6:
            break

    return centers


def euclidean_kmeans(
    features: torch.Tensor,
    k:        int,
    n_iter:   int = 100,
) -> torch.Tensor:
    """
    Standard Euclidean K-Means.

    Used as a secondary reference or warm-up step before running
    ``spherical_kmeans``.

    Parameters
    ----------
    features : (N, d) feature matrix (need not be unit-norm).
    k        : number of clusters.
    n_iter   : maximum Lloyd iterations.

    Returns
    -------
    centers : (K, d) centroid matrix.
              If k >= N, returns a copy of *features* directly.
    """
    N = features.size(0)
    if k >= N:
        return features.clone()

    indices = torch.randperm(N, device=features.device)[:k]
    centers = features[indices].clone()   # (K, d)

    for _ in range(n_iter):
        # --- Assignment: nearest Euclidean centroid ---
        dists       = torch.cdist(features, centers)   # (N, K)
        assignments = dists.argmin(dim=1)               # (N,)

        # --- Update: arithmetic mean per cluster ---
        new_centers = torch.zeros_like(centers)
        for c in range(k):
            mask = (assignments == c)
            if mask.sum() > 0:
                new_centers[c] = features[mask].mean(dim=0)
            else:
                new_centers[c] = centers[c]

        shift = (new_centers - centers).norm(dim=1).max().item()
        centers = new_centers
        if shift < 1e-6:
            break

    return centers