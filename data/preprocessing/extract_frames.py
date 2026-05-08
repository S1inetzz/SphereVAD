# extract_frames_v2.py
import cv2
import os
import csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# ============================================================
# Configuration
# ============================================================

DATASET = "UBnormal"

DATASET_CONFIGS = {
    "UCF": {
        "INPUT_CSV":  "./data/UCF_data/test_ground_truth.csv",
        "VIDEO_DIR":  "./datasets/UCF-crime/data/",
        "OUTPUT_DIR": "./data/UCF_data/test/",
        "OUTPUT_CSV": "./data/UCF_data/test_frames.csv",
    },
    "XD": {
        "INPUT_CSV":  "./data/XD_data/test_ground_truth.csv",
        "VIDEO_DIR":  "./datasets/XD-violence/test/",
        "OUTPUT_DIR": "./data/XD_data/test/",
        "OUTPUT_CSV": "./data/XD_data/test_frames.csv",
    },
    "UBnormal": {
        "INPUT_CSV":  "./datasets/UBnormal/test_gt.csv",
        "VIDEO_DIR":  "./datasets/UBnormal",
        "OUTPUT_DIR": "./data/UBnormal_data/test/",
        "OUTPUT_CSV": "./data/UBnormal_data/test_frames.csv",
    },
}

_cfg       = DATASET_CONFIGS[DATASET]
INPUT_CSV  = _cfg["INPUT_CSV"]
VIDEO_DIR  = _cfg["VIDEO_DIR"]
OUTPUT_DIR = _cfg["OUTPUT_DIR"]
OUTPUT_CSV = _cfg["OUTPUT_CSV"]

CLIP_LENGTH    = 24
K_FRAMES       = 4
TARGET_SIZE    = (336, 336)
NUM_WORKERS    = 96
IO_THREADS     = 4
JPEG_QUALITY   = 95
RANDOM_SEED    = 42
CLIPS_PER_TASK = 50

# ============================================================
# Utility Functions
# ============================================================


def parse_anomaly_frames(anomaly_str):
    """
    Parse anomaly frame ranges from multiple formats:
    - UCF/XD:    "50-120;200-260"  (semicolon-separated)
    - UBnormal:  "50-120,200-260"  (comma-separated)
    - Normal:    "0", empty string, or NaN
    """
    if pd.isna(anomaly_str):
        return set()
    anomaly_str = str(anomaly_str).strip()
    if anomaly_str == '' or anomaly_str == '0':
        return set()

    anomaly_frames = set()
    # Normalize separators: replace commas with semicolons, then split
    unified = anomaly_str.replace(',', ';')
    for seg in unified.split(';'):
        seg = seg.strip()
        if '-' in seg:
            try:
                s, e = map(lambda x: int(x.strip()), seg.split('-'))
                anomaly_frames.update(range(s, e + 1))
            except (ValueError, IndexError):
                continue
    return anomaly_frames


def get_clip_label(clip_start, clip_length, anomaly_frames):
    if not anomaly_frames:
        return 0
    return 1 if any(
        f in anomaly_frames for f in range(clip_start, clip_start + clip_length)
    ) else 0


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def ubnormal_video_name_to_rel_path(video_name):
    """
    Convert a UBnormal video name to its relative path.
    Format: {normal|abnormal}_scene_{N}_scenario_{M}[_suffix]
    """
    parts = video_name.split("_")
    scene_num = parts[2]
    return f"Scene{scene_num}/{video_name}.mp4"


# ============================================================
# Phase 1: Scan Videos and Build Clip-level Task List
# ============================================================

def build_filename_index(video_dir):
    index = {}
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv')):
                rel = os.path.relpath(os.path.join(root, f), video_dir)
                if f not in index:
                    index[f] = rel
    print(f"File index: found {len(index)} video files under {video_dir}")
    return index


def resolve_video_rel_path(video_name, filename_index):
    """
    Resolve a video_name from the CSV to a path relative to VIDEO_DIR.
    - UBnormal: constructed directly from the naming convention
    - UCF/XD:   looked up via the filename index
    """
    if DATASET == "UBnormal":
        return ubnormal_video_name_to_rel_path(video_name)
    else:
        if video_name in filename_index:
            return filename_index[video_name]
        return video_name


def get_anomaly_column(df):
    """Auto-detect the anomaly range column name in the CSV."""
    for col in ['anomaly_ranges', 'anomaly_frames']:
        if col in df.columns:
            return col
    raise KeyError(f"No anomaly annotation column found. Available columns: {list(df.columns)}")


def build_clip_tasks(input_csv, video_dir, clip_length, clips_per_task):
    df = pd.read_csv(input_csv)
    anomaly_col = get_anomaly_column(df)
    print(f"Dataset: {DATASET}, Videos: {len(df)}, Anomaly column: '{anomaly_col}'")

    filename_index = build_filename_index(video_dir)

    # Parallel frame count scan
    video_infos = []
    print("Scanning video frame counts...")
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, 64)) as pool:
        futures = {}
        for _, row in df.iterrows():
            video_name = row['video_name']
            rel_path = resolve_video_rel_path(video_name, filename_index)
            vpath = os.path.join(video_dir, rel_path)
            fut = pool.submit(get_video_frame_count, vpath)
            futures[fut] = (rel_path, row.get(anomaly_col, ''))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            rel_path, anomaly_str = futures[fut]
            n_frames = fut.result()
            if n_frames > 0:
                video_infos.append((rel_path, anomaly_str, n_frames))
            else:
                print(f"  Warning: skipping unreadable video: {rel_path}")

    # Split into clip intervals
    clip_items = []
    global_clip_idx = 0
    video_clip_offset = {}

    for rel_path, anomaly_str, n_frames in video_infos:
        video_clip_offset[rel_path] = global_clip_idx
        frame_cursor = 0
        while frame_cursor + K_FRAMES <= n_frames:
            clip_end = min(frame_cursor + clip_length, n_frames)
            actual_len = clip_end - frame_cursor
            if actual_len < K_FRAMES:
                break
            clip_items.append((
                rel_path, anomaly_str,
                global_clip_idx,
                frame_cursor,
                clip_end,
            ))
            global_clip_idx += 1
            frame_cursor = clip_end

    total_clips = len(clip_items)
    print(f"Total clips: {total_clips}")

    # Pack into batches of clips_per_task
    task_batches = []
    for i in range(0, total_clips, clips_per_task):
        task_batches.append(clip_items[i: i + clips_per_task])

    print(f"Task batches: {len(task_batches)} (up to {clips_per_task} clips each)")
    return task_batches, total_clips, video_clip_offset


# ============================================================
# Phase 2: Worker — Process One Clip Batch
# ============================================================


def process_clip_batch(args):
    (batch, video_dir, output_dir, clip_length, k_frames,
     target_size, jpeg_quality, video_clip_offset_map) = args

    frame_data = []
    clip_counts = {0: 0, 1: 0}
    errors = []

    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    from collections import OrderedDict
    grouped = OrderedDict()
    for item in batch:
        vpath = item[0]
        grouped.setdefault(vpath, []).append(item)

    for video_rel_path, items in grouped.items():
        full_path = os.path.join(video_dir, video_rel_path)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            errors.append(f"Cannot open: {video_rel_path}")
            continue

        video_basename = os.path.splitext(os.path.basename(video_rel_path))[0]
        first_global = video_clip_offset_map.get(video_rel_path, 0)
        current_pos = 0

        for (_, anomaly_str, global_clip_idx, clip_start, clip_end) in items:
            anomaly_frames = parse_anomaly_frames(anomaly_str)
            actual_len = clip_end - clip_start

            if clip_start != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                current_pos = clip_start

            chunk = []
            for _ in range(actual_len):
                ret, frame = cap.read()
                if not ret:
                    break
                chunk.append(frame)
            current_pos += len(chunk)

            if len(chunk) < k_frames:
                continue

            label = get_clip_label(clip_start, len(chunk), anomaly_frames)
            clip_counts[label] += 1

            indices = np.linspace(0, len(chunk) - 1, k_frames, dtype=int)
            per_video_clip_idx = global_clip_idx - first_global

            for i, idx in enumerate(indices):
                frame = chunk[idx]
                if target_size:
                    frame = cv2.resize(frame, target_size)
                save_name = f"{video_basename}_clip{per_video_clip_idx:04d}_{i:02d}.jpg"
                cv2.imwrite(os.path.join(output_dir, save_name), frame, params)
                frame_data.append([save_name, label])

            chunk.clear()

        cap.release()

    return frame_data, len(batch), clip_counts, errors


# ============================================================
# Main
# ============================================================


def extract_frames_parallel():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.dirname(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Phase 1
    task_batches, total_clips, video_clip_offset = build_clip_tasks(
        INPUT_CSV, VIDEO_DIR, CLIP_LENGTH, CLIPS_PER_TASK)

    offset_map = dict(video_clip_offset)

    # Phase 2
    worker_args = [
        (batch, VIDEO_DIR, OUTPUT_DIR, CLIP_LENGTH, K_FRAMES,
         TARGET_SIZE, JPEG_QUALITY, offset_map)
        for batch in task_batches
    ]

    all_frames = []
    total_clip_counts = {0: 0, 1: 0}
    errors = []

    print(f"\nParallel processing: {NUM_WORKERS} workers, "
          f"clip_length={CLIP_LENGTH}, k_frames={K_FRAMES}, "
          f"batch_size={CLIPS_PER_TASK}")

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as exe:
        futures = {exe.submit(process_clip_batch, a): idx
                   for idx, a in enumerate(worker_args)}

        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for fut in as_completed(futures):
                try:
                    frames, n_clips, counts, errs = fut.result()
                    all_frames.extend(frames)
                    total_clip_counts[0] += counts[0]
                    total_clip_counts[1] += counts[1]
                    errors.extend(errs)
                except Exception as e:
                    errors.append(str(e))
                pbar.update(1)

    # Write output CSV
    all_frames.sort(key=lambda x: x[0])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_name', 'label'])
        writer.writerows(all_frames)

    print("\n" + "=" * 50)
    print(f"Done! Dataset={DATASET}")
    print(f"  Total clips  = {total_clips}")
    print(f"  Total frames = {len(all_frames)}")
    print(f"  Normal clips = {total_clip_counts[0]}")
    print(f"  Anomaly clips = {total_clip_counts[1]}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    - {e}")


if __name__ == "__main__":
    extract_frames_parallel()