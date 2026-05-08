"""
Spherical vMF inference pipeline for video anomaly detection.
Supports UCF-Crime, XD-Violence, and UBnormal test benchmarks.

Entry point – all heavy lifting is delegated to the spherevad package.
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings('ignore')

# ── SphereVAD modules ────────────────────────────────────────────────
from spherevad import (
    frechet_mean,
    spherical_centering,
    spherical_kmeans,
    euclidean_kmeans,
    build_global_holistic_enhanced_features,
    compute_adaptive_ambiguity,
    vmf_guided_single_video,
    vmf_score,
    set_seed,
    expand_and_smooth,
    parse_anomaly_frames,
)

# ====================================================================
#  Core Hyperparameters
# ====================================================================
KAPPA          = 1.0 / 0.3
K_NORM         = 3
K_ABN          = 25
HOLISTIC_ALPHA = 0.35
SLERP_BETA     = 0

# ====================================================================
#  Fixed Design Choices
# ====================================================================
DATASET = "UBnormal"

TRAIN_FEATURE_DIR = ""
TRAIN_LABEL_CSV   = ""

DATASET_CONFIGS = {
    "UCF": {
        "TEST_FEATURE_DIR": "",
        "TEST_LABEL_CSV":   "",
        "CSV_FORMAT":       "UCF",
    },
    "XD": {
        "TEST_FEATURE_DIR": "",
        "TEST_LABEL_CSV":   "",
        "CSV_FORMAT":       "UCF",
    },
    "UBnormal": {
        "TEST_FEATURE_DIR": "",
        "TEST_LABEL_CSV":   "",
        "CSV_FORMAT":       "UBnormal",
    },
}

assert DATASET in DATASET_CONFIGS
_cfg             = DATASET_CONFIGS[DATASET]
TEST_FEATURE_DIR = _cfg["TEST_FEATURE_DIR"]
TEST_LABEL_CSV   = _cfg["TEST_LABEL_CSV"]
CSV_FORMAT       = _cfg["CSV_FORMAT"]

TARGET_FEATURE   = 'feat_last_token'
HOLISTIC_FEATURE = 'feat_vis_last'
CLIP_LENGTH      = 24
SMOOTH_SIGMA     = 52
FRECHET_N_ITERS  = 5

HOLISTIC_THRESHOLD    = 0.45
HOLISTIC_TOPK         = 200
HOLISTIC_CHUNK        = 2048
HOLISTIC_INTRA_THRESH = 0.5

N_DOMINANT_ABN  = 1
N_DOMINANT_NORM = 2
MIN_ABN_CLIPS   = 3
ONLY_PULL_AMBIGUOUS = True

DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
MAX_WORKERS = 64


# ====================================================================
#  Test Set Loading
# ====================================================================

def load_test_videos(test_df, csv_format, test_feature_dir, clip_length):
    def _load_ucf(row):
        v_name  = os.path.splitext(str(row['video_name']))[0]
        pt_path = os.path.join(test_feature_dir, v_name + '.pt')
        if not os.path.exists(pt_path):
            return None
        pt_data   = torch.load(pt_path, map_location='cpu', weights_only=True)
        feat_main = pt_data[TARGET_FEATURE].float()
        feat_hol  = pt_data[HOLISTIC_FEATURE].float()
        n_clips   = feat_main.shape[0]
        n_frames  = n_clips * clip_length
        labels    = np.zeros(n_frames, dtype=int)
        for f in parse_anomaly_frames(row.get('anomaly_frames', ''), fmt="UCF"):
            if f < n_frames:
                labels[f] = 1
        return {
            'feat_main':     feat_main,
            'feat_holistic': feat_hol,
            'frame_labels':  labels,
            'video_label':   int(row['label']),
            'video_name':    v_name,
            'total_frames':  n_frames,
        }

    def _load_ubnormal(row):
        v_name  = str(row['video_name']).strip()
        pt_path = os.path.join(test_feature_dir, v_name + '.pt')
        if not os.path.exists(pt_path):
            return None
        pt_data      = torch.load(pt_path, map_location='cpu', weights_only=True)
        feat_main    = pt_data[TARGET_FEATURE].float()
        feat_hol     = pt_data[HOLISTIC_FEATURE].float()
        total_frames = int(row['total_frames'])
        anomaly_str  = str(row['anomaly_ranges']).strip()
        labels       = np.zeros(total_frames, dtype=int)
        for f in parse_anomaly_frames(anomaly_str, fmt="UBnormal"):
            if f < total_frames:
                labels[f] = 1
        video_label = 0 if anomaly_str in ('0', '') else 1
        return {
            'feat_main':     feat_main,
            'feat_holistic': feat_hol,
            'frame_labels':  labels,
            'video_label':   video_label,
            'video_name':    v_name,
            'total_frames':  total_frames,
        }

    loader = _load_ubnormal if csv_format == "UBnormal" else _load_ucf
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(loader, [row for _, row in test_df.iterrows()]))
    return [r for r in results if r is not None]


# ====================================================================
#  Main
# ====================================================================

def main():
    set_seed(RANDOM_SEED)
    print(f"\n{'='*90}")
    print(f"  SvMF: Spherical vMF Inference  [{DATASET}]")
    print(f"  Hyperparameters: kappa={KAPPA:.2f}, K_N={K_NORM}, K_A={K_ABN}, "
          f"alpha={HOLISTIC_ALPHA}, beta={SLERP_BETA}")
    print(f"  Ambiguity interval: adaptive (MAD)")
    print(f"{'='*90}")

    # ── Training features ────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_LABEL_CSV)
    train_feats_list, train_labels_list = [], []
    for _, row in train_df.iterrows():
        pt_path = os.path.join(TRAIN_FEATURE_DIR, row['pt_name'])
        if not os.path.exists(pt_path):
            continue
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        feat = data[TARGET_FEATURE]
        for bi, b_str in enumerate(str(row['branch_info']).split(',')):
            if ':' not in b_str:
                continue
            train_feats_list.append(feat[bi])
            train_labels_list.append(
                0 if b_str.split(':')[1] == 'Normal' else 1
            )

    train_features = torch.stack(train_feats_list).to(DEVICE)
    train_labels   = torch.tensor(train_labels_list).to(DEVICE)
    norm_feats_all = train_features[train_labels == 0]
    abn_feats_all  = train_features[train_labels == 1]
    print(f"  Training set – Normal: {norm_feats_all.size(0)}, "
          f"Abnormal: {abn_feats_all.size(0)}")

    # ── Test features ─────────────────────────────────────────────────
    test_df    = pd.read_csv(TEST_LABEL_CSV)
    test_videos = load_test_videos(
        test_df, CSV_FORMAT, TEST_FEATURE_DIR, CLIP_LENGTH
    )
    print(f"  Test videos: {len(test_videos)}")

    all_main = torch.cat(
        [v['feat_main']     for v in test_videos], dim=0
    ).to(DEVICE)
    all_hol  = torch.cat(
        [v['feat_holistic'] for v in test_videos], dim=0
    ).to(DEVICE)

    video_boundaries, offset = [], 0
    for v in test_videos:
        T = v['feat_main'].size(0)
        video_boundaries.append((offset, offset + T))
        offset += T
    print(f"  Total clips: {offset}")

    # ── Spherical normalisation ───────────────────────────────────────
    all_main_sphere   = F.normalize(all_main,       p=2, dim=1)
    all_hol_sphere    = F.normalize(all_hol,        p=2, dim=1)
    train_all_sphere  = F.normalize(train_features, p=2, dim=1)
    norm_feats_sphere = F.normalize(norm_feats_all, p=2, dim=1)
    abn_feats_sphere  = F.normalize(abn_feats_all,  p=2, dim=1)

    # ── Fréchet means ─────────────────────────────────────────────────
    print(f"\n[Computing unified Fréchet mean (train + test)...]")
    combined    = torch.cat([train_all_sphere, all_main_sphere], dim=0)
    fm_unified  = frechet_mean(combined,        n_iters=FRECHET_N_ITERS)
    fm_train    = frechet_mean(train_all_sphere, n_iters=FRECHET_N_ITERS)
    fm_test     = frechet_mean(all_main_sphere,  n_iters=FRECHET_N_ITERS)
    fm_hol      = frechet_mean(all_hol_sphere,   n_iters=FRECHET_N_ITERS)

    geo_tr  = np.degrees(np.arccos(
        np.clip((fm_train * fm_unified).sum().item(), -1, 1)))
    geo_te  = np.degrees(np.arccos(
        np.clip((fm_test  * fm_unified).sum().item(), -1, 1)))
    geo_gap = np.degrees(np.arccos(
        np.clip((fm_train * fm_test  ).sum().item(), -1, 1)))
    print(f"  Train FM <-> Unified FM : {geo_tr:.4f} deg")
    print(f"  Test  FM <-> Unified FM : {geo_te:.4f} deg")
    print(f"  Train <-> Test FM       : {geo_gap:.4f} deg")

    # ── Spherical centering ───────────────────────────────────────────
    print(f"\n[Spherical centering...]")
    all_main_centered = spherical_centering(all_main_sphere,   fm_unified)
    all_hol_centered  = spherical_centering(all_hol_sphere,    fm_hol)
    norm_centered     = spherical_centering(norm_feats_sphere, fm_unified)
    abn_centered      = spherical_centering(abn_feats_sphere,  fm_unified)

    # ── Global holistic enhancement ───────────────────────────────────
    print(f"\n[Global holistic enhancement (alpha={HOLISTIC_ALPHA})...]")
    all_main_global_h = build_global_holistic_enhanced_features(
        all_hol_centered, all_main_centered, video_boundaries,
        alpha=HOLISTIC_ALPHA, threshold=HOLISTIC_THRESHOLD,
        top_k=HOLISTIC_TOPK, chunk_size=HOLISTIC_CHUNK, device=DEVICE,
    )

    # ── Prototype learning ────────────────────────────────────────────
    print(f"\n[Spherical K-Means (K_N={K_NORM}, K_A={K_ABN})...]")
    set_seed(RANDOM_SEED)
    euclidean_kmeans(norm_centered, K_NORM, n_iter=100)
    euclidean_kmeans(abn_centered,  K_ABN,  n_iter=100)
    mu_n = spherical_kmeans(norm_centered, K_NORM, n_iter=100)
    mu_a = spherical_kmeans(abn_centered,  K_ABN,  n_iter=100)
    print(f"  Normal prototypes:   {mu_n.shape}")
    print(f"  Abnormal prototypes: {mu_a.shape}")

    # ── Initial vMF scores ────────────────────────────────────────────
    init_scores_all       = vmf_score(all_main_global_h, mu_n, mu_a, KAPPA)
    init_scores_per_video = [
        init_scores_all[s:e] for s, e in video_boundaries
    ]

    rho_low, rho_high = compute_adaptive_ambiguity(init_scores_all)
    print(f"\n  Ambiguity interval: [{rho_low:.4f}, {rho_high:.4f}]  "
          f"(radius={0.5*(rho_high-rho_low):.4f})")

    # ── vMF-guided inference (SGP) ────────────────────────────────────
    print(f"\n[vMF-guided inference (beta={SLERP_BETA}, "
          f"interval=[{rho_low:.4f}, {rho_high:.4f}])...]")
    all_main_vmf_guided = torch.zeros_like(all_main_centered)
    n_abn_detected = n_amb_total = 0

    for v_idx in range(len(test_videos)):
        start, end = video_boundaries[v_idx]
        pulled, is_abn = vmf_guided_single_video(
            v_main_h    = all_main_global_h[start:end],
            v_hol       = all_hol_centered[start:end],
            mu_norm     = mu_n,
            mu_abn      = mu_a,
            init_scores = init_scores_per_video[v_idx],
            amb_low     = rho_low,
            amb_high    = rho_high,
            beta_base   = SLERP_BETA,
            device      = DEVICE,
            min_abn_clips         = MIN_ABN_CLIPS,
            n_dominant_abn        = N_DOMINANT_ABN,
            n_dominant_norm       = N_DOMINANT_NORM,
            only_pull_ambiguous   = ONLY_PULL_AMBIGUOUS,
            holistic_intra_thresh = HOLISTIC_INTRA_THRESH,
        )
        all_main_vmf_guided[start:end] = pulled
        if is_abn:
            n_abn_detected += 1
        sv = init_scores_per_video[v_idx]
        n_amb_total += int(np.sum((sv >= rho_low) & (sv <= rho_high)))

    print(f"    Abnormal videos: {n_abn_detected}/{len(test_videos)}, "
          f"Ambiguous clips: {n_amb_total}")

    # ── Scoring & evaluation ──────────────────────────────────────────
    print(f"\n[Scoring and evaluation...]")
    method_names = [
        'M1: vMF-Baseline',
        'M2: +Holistic(Global)',
        'M3: +vMF-Guided(Single)',
    ]
    collectors = {name: ([], []) for name in method_names}
    all_labels, video_gt = [], []

    for v_idx, v in enumerate(test_videos):
        start, end   = video_boundaries[v_idx]
        total_frames = v['total_frames']

        sc_m1 = vmf_score(all_main_centered[start:end],   mu_n, mu_a, KAPPA)
        sc_m2 = vmf_score(all_main_global_h[start:end],   mu_n, mu_a, KAPPA)
        sc_m3 = vmf_score(all_main_vmf_guided[start:end], mu_n, mu_a, KAPPA)

        result_map = {
            'M1: vMF-Baseline':        sc_m1,
            'M2: +Holistic(Global)':   sc_m2,
            'M3: +vMF-Guided(Single)': sc_m3,
        }
        for name in method_names:
            sc = result_map[name]
            collectors[name][0].append(
                expand_and_smooth(sc, CLIP_LENGTH, SMOOTH_SIGMA,
                                  target_frames=total_frames)
            )
            collectors[name][1].append(
                sc_m2.max() if name == 'M3: +vMF-Guided(Single)'
                else sc.max()
            )

        all_labels.append(v['frame_labels'])
        video_gt.append(v['video_label'])

    final_labels = np.concatenate(all_labels)

    # ── Results table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  K_N={K_NORM}, K_A={K_ABN}, kappa={KAPPA:.2f}, "
          f"alpha={HOLISTIC_ALPHA}, beta={SLERP_BETA}")
    print(f"  Ambiguity interval: [{rho_low:.4f}, {rho_high:.4f}] (adaptive)")
    print(f"{'_'*90}")
    print(f"  {'Method':35s} | {'Frame AP':>10s} | "
          f"{'Frame AUC':>10s} | {'Video AUC':>10s}")
    print(f"{'_'*90}")

    results_cache = {}
    best_ap, best_name = -1, ""
    for name in method_names:
        fs      = np.concatenate(collectors[name][0])
        min_len = min(len(fs), len(final_labels))
        f_ap    = average_precision_score(final_labels[:min_len], fs[:min_len])
        f_auc   = roc_auc_score(final_labels[:min_len], fs[:min_len])
        v_auc   = roc_auc_score(video_gt, collectors[name][1])
        results_cache[name] = (f_ap, f_auc, v_auc)
        if f_ap > best_ap:
            best_ap, best_name = f_ap, name

    for name in method_names:
        f_ap, f_auc, v_auc = results_cache[name]
        marker = " *" if name == best_name else ""
        print(f"  {name:35s} | {f_ap:>10.4f} | "
              f"{f_auc:>10.4f} | {v_auc:>10.4f}{marker}")
    print(f"{'='*90}")

    print(f"\n  [Incremental analysis (delta vs M1)]")
    m1_ap = results_cache['M1: vMF-Baseline'][0]
    for name in method_names[1:]:
        delta = results_cache[name][0] - m1_ap
        print(f"    {name:35s} : {'+' if delta >= 0 else ''}{delta:.4f} Frame AP")
    print()


if __name__ == "__main__":
    main()