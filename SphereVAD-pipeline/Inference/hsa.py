"""
Holistic Scene Attention (HSA)  –  Eq. 5.

Two public APIs
---------------
build_holistic_matrix
    Construct a *per-video* soft attention matrix from holistic clip
    features (used inside SGP for intra-video neighbour aggregation).

build_global_holistic_enhanced_features
    Cross-video holistic enhancement: for every clip in the test set,
    aggregate semantically similar clips across the **entire** test
    corpus and blend them with the local clip feature.
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Per-video holistic attention matrix                                 #
# ------------------------------------------------------------------ #

def build_holistic_matrix(
    features:  torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Build a row-stochastic soft-attention matrix for a single video.

    Steps
    -----
    1. L2-normalise features.
    2. Compute cosine-similarity matrix and threshold at *threshold*.
    3. Softmax over non-zero entries (temperature = 1).

    Parameters
    ----------
    features  : (T, d) clip features for one video.
    threshold : cosine-similarity gate; pairs below this are zeroed out.

    Returns
    -------
    A_H : (T, T) row-stochastic attention matrix.
    """
    feat_norm = F.normalize(features, p=2, dim=1)             # (T, d)
    sim       = torch.matmul(feat_norm, feat_norm.T)          # (T, T)

    # Hard gate: keep only similar pairs
    sim = torch.where(sim > threshold, sim, torch.zeros_like(sim))

    # Softmax-style normalisation (log-sum-exp for numerical stability)
    row_max, _ = sim.max(dim=1, keepdim=True)
    exp_sim     = torch.exp(sim - row_max) * (sim > 0).float()
    A_H         = exp_sim / exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)

    return A_H


# ------------------------------------------------------------------ #
#  Global cross-video holistic enhancement                             #
# ------------------------------------------------------------------ #

def build_global_holistic_enhanced_features(
    all_hol_features:  torch.Tensor,
    all_main_features: torch.Tensor,
    video_boundaries:  list,
    alpha:             float,
    threshold:         float,
    top_k:             int,
    chunk_size:        int,
    device:            str = 'cuda:0',
) -> torch.Tensor:
    """
    Globally enhance clip features via cross-video holistic attention.

    For each clip i, we find the *top_k* most similar clips in the
    entire test set (measured by holistic-feature cosine similarity),
    aggregate their main features with softmax weights, and blend:

        x̃_i = (1 - α) · x_i  +  α · Σ_j A_{ij} x_j

    The result is re-normalised to the unit sphere.

    Processing is done in *chunk_size*-sized row blocks to keep GPU
    memory usage bounded.

    Parameters
    ----------
    all_hol_features  : (N, d) holistic clip features (unit-norm expected).
    all_main_features : (N, d) main clip features (centred & unit-norm).
    video_boundaries  : list of (start, end) index pairs per video.
                        (Retained for potential future intra-video masking.)
    alpha             : blending weight in [0, 1]  (0 → no enhancement).
    threshold         : cosine-similarity gate for holistic features.
    top_k             : retain at most top_k neighbours per clip (0 = all).
    chunk_size        : number of rows processed per GPU kernel call.
    device            : target device string.

    Returns
    -------
    enhanced_main : (N, d) unit-norm enhanced main features.
    """
    N        = all_hol_features.size(0)
    hol_norm = F.normalize(all_hol_features, p=2, dim=1)   # (N, d)

    enhanced_main = torch.zeros_like(all_main_features)

    for chunk_start in range(0, N, chunk_size):
        chunk_end  = min(chunk_start + chunk_size, N)
        chunk_hol  = hol_norm[chunk_start:chunk_end]        # (B, d)

        # --- Full-row cosine similarity ---
        sim_chunk = torch.matmul(chunk_hol, hol_norm.T)     # (B, N)
        sim_chunk = torch.where(
            sim_chunk > threshold, sim_chunk, torch.zeros_like(sim_chunk)
        )

        # --- Top-K sparsification ---
        if 0 < top_k < N:
            _, topk_indices = sim_chunk.topk(
                min(top_k, sim_chunk.size(1)), dim=1
            )
            mask = torch.zeros_like(sim_chunk, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            sim_chunk = sim_chunk * mask.float()

        # --- Row-wise softmax ---
        row_max, _ = sim_chunk.max(dim=1, keepdim=True)
        exp_sim     = torch.exp(sim_chunk - row_max) * (sim_chunk > 0).float()
        A_chunk     = exp_sim / exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # --- Feature aggregation + blending ---
        aggregated = torch.matmul(A_chunk, all_main_features)   # (B, d)
        original   = all_main_features[chunk_start:chunk_end]   # (B, d)
        enhanced_main[chunk_start:chunk_end] = (
            (1 - alpha) * original + alpha * aggregated
        )

    return F.normalize(enhanced_main, p=2, dim=1)