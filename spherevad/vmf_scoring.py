"""
von Mises–Fisher likelihood-ratio scoring  (Eq. 4).

Given a set of normal prototypes mu_norm and abnormal prototypes mu_abn,
each clip feature is scored by the soft probability of belonging to the
abnormal class under a two-class vMF mixture model:

    P(abn | x) = exp(-κ · d_abn) / (exp(-κ · d_norm) + exp(-κ · d_abn))

where d_{norm/abn} = min_{k} geodesic(x, mu_{norm/abn,k}).

The log-sum-exp trick is applied for numerical stability.
"""

import numpy as np
import torch

from .utils import geodesic_distance


def vmf_score(
    feat_normed: torch.Tensor,
    mu_norm:     torch.Tensor,
    mu_abn:      torch.Tensor,
    kappa:       float,
) -> np.ndarray:
    """
    Compute per-clip vMF anomaly probability  (Eq. 4).

    Parameters
    ----------
    feat_normed : (T, d) unit-norm clip features.
    mu_norm     : (K_N, d) normal prototype vectors.
    mu_abn      : (K_A, d) abnormal prototype vectors.
    kappa       : concentration parameter  (kappa = 1 / temperature).

    Returns
    -------
    prob : (T,) numpy array of anomaly probabilities in (0, 1).
    """
    # Minimum geodesic distance to any prototype of each class
    d_norm = geodesic_distance(feat_normed, mu_norm).min(dim=1).values  # (T,)
    d_abn  = geodesic_distance(feat_normed, mu_abn ).min(dim=1).values  # (T,)

    # vMF log-likelihoods (unnormalised)
    logit_norm = -kappa * d_norm   # (T,)
    logit_abn  = -kappa * d_abn   # (T,)

    # Numerically stable soft-max over the two classes
    max_logit = torch.maximum(logit_norm, logit_abn)
    exp_norm  = torch.exp(logit_norm - max_logit)
    exp_abn   = torch.exp(logit_abn  - max_logit)
    prob      = exp_abn / (exp_norm + exp_abn + 1e-8)   # (T,)

    return prob.cpu().numpy()