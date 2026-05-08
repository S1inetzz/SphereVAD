"""
Spherical centering via the Riemannian Log map  (Eq. 2).

After mapping features to the tangent space T_{mu}S^{d-1}, we L2-normalise
the resulting tangent vectors so that all features lie on a *new* unit
hypersphere centred at the origin of the tangent space.  This removes the
global distributional bias captured by the Fréchet mean mu.
"""

import torch
import torch.nn.functional as F

from .utils import log_map


def spherical_centering(
    X:  torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """
    Centre features on S^{d-1} by Log-mapping and re-normalising.

    Implements Eq. 2:
        x̃_i = Normalize( Log_{mu}(x_i) )

    Parameters
    ----------
    X  : (N, d) unit-norm feature matrix.
    mu : (d,)   unit-norm Fréchet mean (base point).

    Returns
    -------
    (N, d) unit-norm centred features in the tangent space T_{mu}S^{d-1}.
    """
    tangent = log_map(X, mu)                    # (N, d)  tangent vectors
    return F.normalize(tangent, p=2, dim=1)     # (N, d)  re-normalised