"""
Utility functions:
  - Reproducibility seeding
  - SLERP interpolation
  - Log / Exp maps on S^{d-1}
  - Geodesic distance matrix
  - Frame-score expansion + Gaussian smoothing
  - Anomaly-frame parsers (UCF / UBnormal)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ------------------------------------------------------------------ #
#  Reproducibility                                                     #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Fix all random-number generators for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
#  Spherical Geometry Primitives                                       #
# ------------------------------------------------------------------ #

def log_map(X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Riemannian Log map at base-point *mu*.

    Projects each row of X onto the tangent space T_{mu}S^{d-1}.

    Parameters
    ----------
    X  : (N, d) unit-norm feature matrix.
    mu : (d,)   unit-norm base point.

    Returns
    -------
    (N, d) tangent vectors (not unit-norm in general).
    """
    dots      = (X @ mu).clamp(-1 + 1e-7, 1 - 1e-7)   # (N,)
    angles    = torch.acos(dots)                         # (N,)
    residuals = X - dots.unsqueeze(1) * mu.unsqueeze(0) # (N, d)
    res_norms = residuals.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (residuals / res_norms) * angles.unsqueeze(1)


def exp_map(V: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Riemannian Exp map at base-point *mu*.

    Maps tangent vectors back onto S^{d-1}.

    Parameters
    ----------
    V  : (N, d) tangent vectors at mu.
    mu : (d,)   unit-norm base point.

    Returns
    -------
    (N, d) unit-norm points on the sphere.
    """
    t_norms = V.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    V_unit  = V / t_norms                                    # (N, d)
    pts = (torch.cos(t_norms) * mu.unsqueeze(0)
           + torch.sin(t_norms) * V_unit)
    return F.normalize(pts, p=2, dim=1)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise geodesic (great-circle) distance matrix.

    Parameters
    ----------
    x : (M, d) unit-norm features.
    y : (K, d) unit-norm features.

    Returns
    -------
    (M, K) distance matrix in [0, pi].
    """
    return torch.acos(torch.matmul(x, y.T).clamp(-1 + 1e-7, 1 - 1e-7))


def slerp(
    f: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP).

    Interpolates between *f* and *p* with mixing weight *beta*.
    beta=0 → returns f;  beta=1 → returns p.

    Parameters
    ----------
    f    : (N, d) source unit vectors.
    p    : (N, d) target unit vectors.
    beta : scalar or (N, 1) per-sample mixing weight in [0, 1].

    Returns
    -------
    (N, d) unit-norm interpolated vectors.
    """
    cos_theta = (f * p).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta     = torch.acos(cos_theta)                    # (N, 1)
    sin_theta = torch.sin(theta).clamp(min=1e-8)         # (N, 1)
    w_f = torch.sin((1.0 - beta) * theta) / sin_theta
    w_p = torch.sin(beta         * theta) / sin_theta
    return F.normalize(w_f * f + w_p * p, p=2, dim=1)


# ------------------------------------------------------------------ #
#  Frame-Score Post-Processing                                         #
# ------------------------------------------------------------------ #

def expand_and_smooth(
    clip_scores:   np.ndarray,
    clip_length:   int,
    sigma:         float,
    target_frames: int | None = None,
) -> np.ndarray:
    """
    Repeat clip-level scores to frame level, then apply Gaussian smoothing.

    Parameters
    ----------
    clip_scores   : (T,) per-clip anomaly scores.
    clip_length   : number of frames each clip covers.
    sigma         : std-dev of the Gaussian kernel (0 → no smoothing).
    target_frames : if given, crop or pad to this exact frame count.

    Returns
    -------
    (target_frames,) or (T*clip_length,) smoothed frame-level scores.
    """
    frame_scores = np.repeat(clip_scores, clip_length)

    if target_frames is not None:
        if len(frame_scores) >= target_frames:
            frame_scores = frame_scores[:target_frames]
        else:
            frame_scores = np.pad(
                frame_scores,
                (0, target_frames - len(frame_scores)),
                mode='edge',
            )

    if sigma > 0:
        frame_scores = gaussian_filter1d(frame_scores, sigma=sigma)

    return frame_scores


# ------------------------------------------------------------------ #
#  Anomaly-Frame Parsers                                               #
# ------------------------------------------------------------------ #

def parse_anomaly_frames_ucf(anomaly_str) -> set:
    """
    Parse UCF-Crime style annotation strings.

    Format: ``"start1-end1;start2-end2;..."``
    Returns a set of 0-based frame indices.
    """
    if pd.isna(anomaly_str) or str(anomaly_str).strip() == '':
        return set()

    anomaly_frames: set = set()
    for seg in str(anomaly_str).split(';'):
        if '-' in seg:
            try:
                start, end = map(int, seg.split('-'))
                anomaly_frames.update(range(start, end + 1))
            except ValueError:
                continue
    return anomaly_frames


def parse_anomaly_frames_ubnormal(anomaly_str) -> set:
    """
    Parse UBnormal style annotation strings.

    Format: ``"start1-end1,start2-end2,..."``
    An empty string or ``"0"`` denotes a normal video.
    Returns a set of 0-based frame indices.
    """
    if pd.isna(anomaly_str):
        return set()

    anomaly_str = str(anomaly_str).strip()
    if anomaly_str in ('', '0'):
        return set()

    anomaly_frames: set = set()
    for seg in anomaly_str.split(','):
        seg = seg.strip()
        if '-' in seg:
            try:
                start, end = map(int, seg.split('-'))
                anomaly_frames.update(range(start, end + 1))
            except ValueError:
                continue
    return anomaly_frames


def parse_anomaly_frames(anomaly_str, fmt: str = "UCF") -> set:
    """
    Dispatch to the dataset-appropriate parser.

    Parameters
    ----------
    anomaly_str : raw annotation string from the CSV.
    fmt         : ``"UCF"`` or ``"UBnormal"``.

    Returns
    -------
    Set of anomalous frame indices.
    """
    if fmt == "UBnormal":
        return parse_anomaly_frames_ubnormal(anomaly_str)
    return parse_anomaly_frames_ucf(anomaly_str)