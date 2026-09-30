"""
Spherical Geodesic Pulling (SGP)  –  Algorithm 1.

Two public functions
--------------------
compute_adaptive_ambiguity
    Data-driven estimation of the ambiguity interval [ρ_low, ρ_high]
    around the decision boundary 0.5, using the Median Absolute
    Deviation (MAD) of the vMF score distribution.

vmf_guided_single_video
    Per-video SGP inference:
      1. Classify clips as normal / ambiguous / abnormal.
      2. Identify dominant prototypes for each class.
      3. Re-assign ambiguous clips via holistic neighbour aggregation.
      4. Pull (SLERP) each clip feature toward its assigned prototype
         centre, with the step-size modulated by score uncertainty.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import slerp, geodesic_distance
from .hsa   import build_holistic_matrix


# ====================================================================
#  Module-level SGP hyper-parameters (mirrors the main config)
# ====================================================================
_MIN_ABN_CLIPS         = 3
_N_DOMINANT_ABN        = 1
_N_DOMINANT_NORM       = 2
_ONLY_PULL_AMBIGUOUS   = True
_HOLISTIC_INTRA_THRESH = 0.5


# ------------------------------------------------------------------ #
#  Adaptive Ambiguity Interval                                         #
# ------------------------------------------------------------------ #

def compute_adaptive_ambiguity(
    scores: np.ndarray,
    center: float = 0.5,
    radius_min: float = 0.05,
    radius_max: float = 0.10,
) -> tuple[float, float]:
    """
    Estimate the ambiguity band [ρ_low, ρ_high] via MAD.

    A score s is *ambiguous* when it lies within *radius* of the
    decision boundary (0.5 by default).  We set the radius by
    observing that a large MAD → scores are spread far from 0.5 →
    few clips are truly uncertain → the interval can be narrow.

        radius = clip( 0.5 − MAD, radius_min, radius_max )

    Parameters
    ----------
    scores      : (N,) array of per-clip vMF anomaly probabilities.
    center      : decision boundary (default 0.5).
    radius_min  : lower bound on the interval half-width.
    radius_max  : upper bound on the interval half-width.

    Returns
    -------
    (rho_low, rho_high) : ambiguity interval endpoints.
    """
    deviations = np.abs(scores - center)
    mad        = np.median(deviations)
    radius     = np.clip(0.5 - mad, radius_min, radius_max)
    return center - radius, center + radius


# ------------------------------------------------------------------ #
#  Per-video SGP                                                        #
# ------------------------------------------------------------------ #

def vmf_guided_single_video(
    v_main_h:    torch.Tensor,
    v_hol:       torch.Tensor,
    mu_norm:     torch.Tensor,
    mu_abn:      torch.Tensor,
    init_scores: np.ndarray,
    amb_low:     float,
    amb_high:    float,
    beta_base:   float,
    device:      str = 'cuda:0',
    min_abn_clips:         int   = _MIN_ABN_CLIPS,
    n_dominant_abn:        int   = _N_DOMINANT_ABN,
    n_dominant_norm:       int   = _N_DOMINANT_NORM,
    only_pull_ambiguous:   bool  = _ONLY_PULL_AMBIGUOUS,
    holistic_intra_thresh: float = _HOLISTIC_INTRA_THRESH,
) -> tuple[torch.Tensor, bool]:
    """
    Spherical Geodesic Pulling for a single video  (Algorithm 1).

    Steps
    -----
    1. Partition clips into {normal, ambiguous, abnormal} using the
       score thresholds [amb_low, amb_high].
    2. Identify *dominant* prototypes – the K-Means centres that
       accumulate the most votes from confidently classified clips.
    3. Resolve ambiguous clips:
       - If the video is detected as abnormal, use the intra-video
         holistic attention matrix (from HSA) to aggregate neighbour
         features and compare similarity to dominant prototypes.
       - Otherwise, assign all ambiguous clips to the normal class.
    4. Pull each clip feature toward its assigned prototype centre
       via SLERP.  The SLERP step-size β is modulated by the clip's
       distance to the decision boundary (more uncertain → larger pull).

    Parameters
    ----------
    v_main_h    : (T, d) globally holistic-enhanced, unit-norm features.
    v_hol       : (T, d) centred holistic features (for intra-video HSA).
    mu_norm     : (K_N, d) normal prototype vectors.
    mu_abn      : (K_A, d) abnormal prototype vectors.
    init_scores : (T,) initial vMF anomaly probabilities.
    amb_low     : lower threshold of the ambiguity band.
    amb_high    : upper threshold of the ambiguity band.
    beta_base   : base SLERP mixing weight (0 → no pull, 1 → full pull).
    device      : target device string.
    min_abn_clips         : minimum #abnormal clips to declare video abnormal.
    n_dominant_abn        : #dominant abnormal prototypes to track.
    n_dominant_norm       : #dominant normal prototypes to track.
    only_pull_ambiguous   : if True, SLERP is applied only to ambiguous clips.
    holistic_intra_thresh : cosine-similarity gate for the intra-video
                            holistic attention matrix.

    Returns
    -------
    pulled_feat      : (T, d) unit-norm pulled features.
    video_is_abnormal: True if the video was classified as containing anomalies.
    """
    T        = v_main_h.size(0)
    scores_t = torch.from_numpy(init_scores).to(device)

    # ---------------------------------------------------------------- #
    # 1. Partition clips                                                #
    # ---------------------------------------------------------------- #
    mask_norm = scores_t < amb_low
    mask_abn  = scores_t > amb_high
    mask_amb  = (~mask_norm) & (~mask_abn)

    idx_norm = mask_norm.nonzero(as_tuple=True)[0]
    idx_abn  = mask_abn .nonzero(as_tuple=True)[0]
    idx_amb  = mask_amb .nonzero(as_tuple=True)[0]

    # ---------------------------------------------------------------- #
    # 2. Dominant prototype identification                              #
    # ---------------------------------------------------------------- #
    dominant_abn_idx: list[int] = []
    video_is_abnormal = False

    if len(idx_abn) >= min_abn_clips:
        video_is_abnormal = True
        sim_to_abn   = torch.matmul(v_main_h[idx_abn], mu_abn.T)   # (#abn, K_A)
        best_per_clip = sim_to_abn.argmax(dim=1)                     # (#abn,)
        votes        = torch.bincount(best_per_clip, minlength=mu_abn.size(0))
        _, top_idx   = votes.topk(min(n_dominant_abn, mu_abn.size(0)))
        dominant_abn_idx = top_idx.tolist()

    dominant_norm_idx: list[int] = []
    if len(idx_norm) > 0:
        sim_to_norm  = torch.matmul(v_main_h[idx_norm], mu_norm.T)
        best_per_clip = sim_to_norm.argmax(dim=1)
        votes_norm   = torch.bincount(best_per_clip, minlength=mu_norm.size(0))
        _, top_idx   = votes_norm.topk(min(n_dominant_norm, mu_norm.size(0)))
        dominant_norm_idx = top_idx.tolist()
    else:
        # Fallback: use the prototype closest to the video mean
        video_mean = v_main_h.mean(dim=0, keepdim=True)
        sim_mean   = torch.matmul(video_mean, mu_norm.T).squeeze(0)
        _, top_idx = sim_mean.topk(min(n_dominant_norm, mu_norm.size(0)))
        dominant_norm_idx = top_idx.tolist()

    # ---------------------------------------------------------------- #
    # 3. Ambiguous clip resolution                                      #
    # ---------------------------------------------------------------- #
    clip_assignment = torch.zeros(T, dtype=torch.long, device=device)
    if len(idx_abn) > 0:
        clip_assignment[idx_abn] = 1   # mark confident abnormal clips

    if len(idx_amb) > 0 and video_is_abnormal and len(dominant_abn_idx) > 0:
        # Build intra-video holistic attention matrix
        A_H          = build_holistic_matrix(v_hol, holistic_intra_thresh)
        amb_weights  = A_H[idx_amb].clone()                         # (#amb, T)
        # Zero out self-connections
        amb_weights[torch.arange(len(idx_amb), device=device), idx_amb] = 0
        neighbor_sum = amb_weights.sum(dim=1)                        # (#amb,)
        has_neighbors = neighbor_sum > 1e-8

        # --- Clips with holistic neighbours ---
        if has_neighbors.any():
            w_valid   = amb_weights[has_neighbors]
            w_valid   = w_valid / w_valid.sum(dim=1, keepdim=True)
            # Aggregated neighbour feature
            mean_nf   = F.normalize(
                torch.matmul(w_valid, v_main_h), p=2, dim=1
            )
            sim_abn_b = torch.matmul(
                mean_nf, mu_abn[dominant_abn_idx].T
            ).max(dim=1).values
            sim_norm_b = torch.matmul(
                mean_nf, mu_norm[dominant_norm_idx].T
            ).max(dim=1).values
            clip_assignment[idx_amb[has_neighbors]] = (
                sim_abn_b > sim_norm_b
            ).long()

        # --- Isolated clips (no holistic neighbours) ---
        if (~has_neighbors).any():
            self_feats = v_main_h[idx_amb[~has_neighbors]]
            sa  = torch.matmul(
                self_feats, mu_abn[dominant_abn_idx].T
            ).max(dim=1).values
            sn  = torch.matmul(
                self_feats, mu_norm[dominant_norm_idx].T
            ).max(dim=1).values
            clip_assignment[idx_amb[~has_neighbors]] = (sa > sn).long()

    elif len(idx_amb) > 0 and not video_is_abnormal:
        # Normal video: pull all ambiguous clips toward normal class
        clip_assignment[idx_amb] = 0

    # ---------------------------------------------------------------- #
    # 4. SLERP pulling                                                  #
    # ---------------------------------------------------------------- #
    pulled_feat = v_main_h.clone()

    pull_mask = mask_amb if only_pull_ambiguous else torch.ones(
        T, dtype=torch.bool, device=device
    )
    n_pull        = pull_mask.sum().item()
    has_dom_abn   = len(dominant_abn_idx) > 0
    has_dom_norm  = len(dominant_norm_idx) > 0

    if n_pull > 0 and (has_dom_abn or has_dom_norm):
        pull_idx     = pull_mask.nonzero(as_tuple=True)[0]      # (#pull,)
        pull_feats   = v_main_h[pull_idx]                        # (#pull, d)
        pull_assigns = clip_assignment[pull_idx]                  # (#pull,)
        pull_scores  = init_scores[pull_idx.cpu().numpy()]        # (#pull,)

        target_centers = torch.zeros_like(pull_feats)

        # Assign target prototype for abnormal clips
        abn_pull = (pull_assigns == 1)
        if abn_pull.any() and has_dom_abn:
            dom_abn_t  = torch.tensor(dominant_abn_idx, device=device)
            abn_feats  = pull_feats[abn_pull]
            best_local = torch.matmul(
                abn_feats, mu_abn[dom_abn_t].T
            ).argmax(dim=1)
            target_centers[abn_pull] = mu_abn[dom_abn_t[best_local]]

        # Assign target prototype for normal clips
        norm_pull = (pull_assigns == 0)
        if norm_pull.any() and has_dom_norm:
            dom_norm_t = torch.tensor(dominant_norm_idx, device=device)
            norm_feats = pull_feats[norm_pull]
            best_local = torch.matmul(
                norm_feats, mu_norm[dom_norm_t].T
            ).argmax(dim=1)
            target_centers[norm_pull] = mu_norm[dom_norm_t[best_local]]

        # Only SLERP clips that have a valid target assigned
        has_target = target_centers.norm(dim=1) > 1e-6

        if has_target.any():
            valid_idx     = pull_idx[has_target]
            valid_feats   = pull_feats[has_target]
            valid_targets = target_centers[has_target]

            # Uncertainty-modulated step-size:
            # clips closer to 0.5 get a larger SLERP step.
            valid_scores    = torch.from_numpy(
                pull_scores[has_target.cpu().numpy()]
            ).to(device)
            dist_to_center  = (valid_scores - 0.5).abs()
            half_width      = (amb_high - amb_low) / 2.0
            if half_width > 0:
                normalized_dist = (dist_to_center / half_width).clamp(max=1.0)
            else:
                normalized_dist = torch.zeros_like(dist_to_center)

            beta = beta_base * (1.0 - 0.5 * normalized_dist)  # (M,)
            beta = beta.unsqueeze(1)                            # (M, 1)

            pulled_feat[valid_idx] = slerp(valid_feats, valid_targets, beta)

    return F.normalize(pulled_feat, p=2, dim=1), video_is_abnormal