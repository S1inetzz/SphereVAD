"""
Karcher / Fréchet mean on the unit hypersphere S^{d-1}.

Implements the iterative Riemannian gradient-descent algorithm
described in Appendix A.1 of the paper.

Reference
---------
Karcher, H. (1977). Riemannian center of mass and mollifier smoothing.
Communications on Pure and Applied Mathematics, 30(5), 509-541.
"""

import torch
import torch.nn.functional as F

from .utils import log_map


def frechet_mean(
    X:       torch.Tensor,
    n_iters: int = 5,
) -> torch.Tensor:
    """
    Compute the Fréchet (Karcher) mean of unit vectors on S^{d-1}.

    The algorithm alternates between:
      1. Mapping all points to the tangent space T_{mu}S^{d-1} via the
         Riemannian Log map.
      2. Taking the Euclidean mean of the tangent vectors.
      3. Mapping the mean tangent vector back to the sphere via Exp.

    Parameters
    ----------
    X       : (N, d) tensor of **unit-norm** feature vectors.
    n_iters : number of Karcher iterations (5 is usually sufficient).

    Returns
    -------
    mu : (d,) unit-norm Fréchet mean.
    """
    # Warm-start: normalised Euclidean mean
    mu = F.normalize(X.mean(dim=0), dim=0)   # (d,)

    for _ in range(n_iters):
        # --- Log map: project X onto T_{mu}S^{d-1} ---
        tangent_vecs = log_map(X, mu)          # (N, d)

        # --- Riemannian gradient step ---
        avg_tangent = tangent_vecs.mean(dim=0) # (d,)
        t_norm      = avg_tangent.norm().clamp(min=1e-8)

        # --- Exp map: walk along the geodesic ---
        mu = (torch.cos(t_norm) * mu
              + torch.sin(t_norm) * (avg_tangent / t_norm))
        mu = F.normalize(mu, dim=0)

    return mu