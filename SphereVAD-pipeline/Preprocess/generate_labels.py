"""
Unified annotation generation script.
Supported datasets: UBnormal / UCF-Crime / XD-Violence
Controlled by the DATASET switch in the configuration section.
"""

import cv2
import numpy as np
import os
import csv

# ======================================================================
#                         Configuration
# ======================================================================
DATASET = "UBnormal"  # Options: "UBnormal" / "UCF" / "XD"

DATASET_CONFIGS = {
    "UBnormal": {
        "VIDEO_DIR":   "/path/to/UBnormal",
        "SPLIT_DIR":   "/path/to/UBnormal/Split",
        "OUTPUT_FILE": "/path/to/UBnormal/test_gt.csv",
    },
    "UCF": {
        "DATA_DIR":        "/path/to/UCF-crime/data/",
        "ANNOTATION_FILE": "/path/to/UCF-crime/annotations.txt",
        "OUTPUT_FILE":     "/path/to/UCF_data/test_ground_truth.csv",
    },
    "XD": {
        "DATA_DIR":        "/path/to/XD-violence/test/",
        "ANNOTATION_FILE": "/path/to/XD-violence/annotations.txt",
        "OUTPUT_FILE":     "/path/to/XD_data/test_ground_truth.csv",
    },
}

assert DATASET in DATASET_CONFIGS, \
    f"DATASET must be one of {list(DATASET_CONFIGS.keys())}, got: {DATASET}"

_cfg = DATASET_CONFIGS[DATASET]


# ======================================================================
#                          Shared Utilities
# ======================================================================

def normalize_name(name: str) -> str:
    """Strip whitespace, normalize path separators, and return the stem (no extension)."""
    name = name.strip().replace("\\", "/")
    basename = os.path.basename(name)
    if basename.lower().endswith(".mp4"):
        return basename[:-4]
    return basename


def find_all_test_videos(data_dir: str):
    """Recursively scan data_dir and return a list of (relative_path, filename) for all .mp4 files."""
    video_list = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, data_dir)
                video_list.append((rel_path, filename))
    return video_list


def get_number_of_frames(video_path: str) -> int:
    """Return the total frame count of a video via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def ensure_dir(path: str):
    """Create the parent directory of path if it does not exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ======================================================================
#                        UBnormal Functions
# ======================================================================

def _merge_intervals(intervals: list) -> list:
    """Sort and merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(seg) for seg in merged]


def generate_labels_ubnormal(video_dir: str, split_dir: str, output_file: str):
    """
    Main annotation generation function for UBnormal.

    CSV format: video_name, total_frames, anomaly_ranges
      - Normal video:   anomaly_ranges = "0"
      - Abnormal video: anomaly_ranges = "start1-end1,start2-end2,..."
    """
    normal_test_file   = os.path.join(split_dir, "normal_test_video_names.txt")
    abnormal_test_file = os.path.join(split_dir, "abnormal_test_video_names.txt")

    normal_names   = sorted(np.loadtxt(normal_test_file,   dtype=str).reshape(-1))
    abnormal_names = sorted(np.loadtxt(abnormal_test_file, dtype=str).reshape(-1))

    ensure_dir(output_file)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "total_frames", "anomaly_ranges"])

        print("=== Processing normal test videos ===")
        for video_name in normal_names:
            scene_name = video_name.split("_")[2]
            video_path = os.path.join(video_dir, f"Scene{scene_name}", video_name + ".mp4")
            num_frames = get_number_of_frames(video_path)
            writer.writerow([video_name, num_frames, "0"])
            print(f"  {video_name}: {num_frames} frames -> 0")

        print("=== Processing abnormal test videos ===")
        for video_name in abnormal_names:
            scene_name = video_name.split("_")[2]
            video_path = os.path.join(video_dir, f"Scene{scene_name}", video_name + ".mp4")
            num_frames = get_number_of_frames(video_path)

            tracks_path = os.path.join(
                video_dir, f"Scene{scene_name}",
                video_name + "_annotations",
                video_name + "_tracks.txt",
            )
            tracks_data = np.loadtxt(tracks_path, delimiter=",")
            if tracks_data.ndim == 1:
                tracks_data = tracks_data[np.newaxis, :]

            intervals = [(int(t[1]), int(t[2])) for t in tracks_data]
            merged = _merge_intervals(intervals)
            ranges_str = ",".join(f"{s}-{e}" for s, e in merged)

            writer.writerow([video_name, num_frames, ranges_str])
            print(f"  {video_name}: {num_frames} frames -> {ranges_str}")

    print(f"\nDone! CSV saved to: {output_file}")


# ======================================================================
#                        UCF-Crime Functions
# ======================================================================

def parse_annotations_ucf(annotation_path: str):
    """
    Parse the Temporal_Anomaly_Annotation file for UCF-Crime.

    Line format: video_name  EventType  start1  end1  start2  end2
      - Normal video:   all start/end values are -1
      - Abnormal video: at least one start/end pair > 0

    Returns:
      anno_dict      : { normalized_name: ['start1-end1', ...] }  abnormal videos only
      all_test_videos: { normalized_name }                         all test videos
    """
    anno_dict = {}
    all_test_videos = set()

    if not os.path.exists(annotation_path):
        print(f"Error: annotation file not found '{annotation_path}'")
        return {}, set()

    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        clean_key = normalize_name(parts[0])
        all_test_videos.add(clean_key)

        frame_values = parts[2:]
        segments = []
        for i in range(0, len(frame_values) - 1, 2):
            try:
                s, e = int(frame_values[i]), int(frame_values[i + 1])
            except ValueError:
                continue
            if s > 0 and e > 0:
                segments.append(f"{s}-{e}")

        if segments:
            anno_dict[clean_key] = segments

    print("Annotation file parsed:")
    print(f"  - Total test videos:    {len(all_test_videos)}")
    print(f"  - Abnormal videos:      {len(anno_dict)}")
    print(f"  - Normal videos:        {len(all_test_videos) - len(anno_dict)}")
    return anno_dict, all_test_videos


def generate_labels_ucf(data_dir: str, annotation_file: str, output_file: str):
    """Main annotation generation function for UCF-Crime."""
    annotations, all_test_videos = parse_annotations_ucf(annotation_file)

    if not os.path.exists(data_dir):
        print(f"Error: data directory not found '{data_dir}'")
        return

    print(f"\nScanning data directory: {data_dir} ...")
    all_videos = find_all_test_videos(data_dir)
    print(f"Found {len(all_videos)} video files in data directory")

    data = []
    matched_abnormal, matched_normal = set(), set()

    for _, filename in all_videos:
        name_clean = normalize_name(filename)
        if name_clean not in all_test_videos:
            continue

        if name_clean in annotations:
            label = 1
            frames_str = "; ".join(annotations[name_clean])
            matched_abnormal.add(name_clean)
        else:
            label = 0
            frames_str = ""
            matched_normal.add(name_clean)

        data.append([filename, label, frames_str])

    ensure_dir(output_file)
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "label", "anomaly_frames"])
            writer.writerows(data)

        found_count = len(matched_abnormal) + len(matched_normal)
        print("-" * 40)
        print("Processing complete!")
        print(f"Test videos in annotation file: {len(all_test_videos)}")
        print(f"Test videos found on disk:      {found_count}")
        print(f"  - Abnormal: {len(matched_abnormal)}")
        print(f"  - Normal:   {len(matched_normal)}")
        print(f"Results saved to: {output_file}")
        print("-" * 40)

        missing = all_test_videos - (matched_abnormal | matched_normal)
        if missing:
            print(f"[WARNING] {len(missing)} test video(s) not found on disk:")
            for i, v in enumerate(sorted(missing)[:20], 1):
                print(f"  {i}. {v}")
            if len(missing) > 20:
                print(f"  ... {len(missing)} total")
        else:
            print("All test videos matched successfully.")

    except IOError as e:
        print(f"Failed to save CSV: {e}")


# ======================================================================
#                        XD-Violence Functions
# ======================================================================

def parse_annotations_xd(annotation_path: str) -> dict:
    """Parse the annotations.txt file for XD-Violence."""
    anno_dict = {}

    if not os.path.exists(annotation_path):
        print(f"Error: annotation file not found '{annotation_path}'")
        return {}

    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        clean_key = normalize_name(parts[0])
        frames = parts[1:]
        segments = [
            f"{frames[i]}-{frames[i + 1]}"
            for i in range(0, len(frames) - 1, 2)
        ]
        anno_dict[clean_key] = segments

    print(f"Annotation file parsed: {len(anno_dict)} abnormal video annotations found.")
    return anno_dict


def generate_labels_xd(video_folder: str, annotation_file: str, output_file: str):
    """Main annotation generation function for XD-Violence."""
    annotations = parse_annotations_xd(annotation_file)
    all_annotated_names = set(annotations.keys())

    if not os.path.exists(video_folder):
        print(f"Error: video folder not found '{video_folder}'")
        return

    print(f"Scanning test folder: {video_folder} ...")

    video_files = sorted(
        f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")
    )

    data = []
    matched_names = set()

    for filename in video_files:
        name_clean = normalize_name(filename)
        if name_clean in annotations:
            label = 1
            frames_str = "; ".join(annotations[name_clean])
            matched_names.add(name_clean)
        else:
            label = 0
            frames_str = ""
        data.append([filename, label, frames_str])

    ensure_dir(output_file)
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "label", "anomaly_frames"])
            writer.writerows(data)

        print("-" * 40)
        print("Processing complete!")
        print(f"Total videos on disk:    {len(video_files)}")
        print(f"Matched abnormal videos: {len(matched_names)}")
        print(f"Normal videos:           {len(video_files) - len(matched_names)}")
        print(f"Results saved to: {output_file}")

        missing = all_annotated_names - matched_names
        if missing:
            print("-" * 40)
            print(f"[WARNING] {len(missing)} annotated video(s) not found in folder:")
            for i, v in enumerate(sorted(missing), 1):
                print(f"  {i}. {v}")
        else:
            print("All annotated videos matched successfully.")
        print("-" * 40)

    except IOError as e:
        print(f"Failed to save CSV: {e}")


# ======================================================================
#                              Entry Point
# ======================================================================

def main():
    print("=" * 50)
    print(f"  Annotation Generation Script — Dataset: {DATASET}")
    print("=" * 50)

    if DATASET == "UBnormal":
        generate_labels_ubnormal(
            _cfg["VIDEO_DIR"],
            _cfg["SPLIT_DIR"],
            _cfg["OUTPUT_FILE"],
        )
    elif DATASET == "UCF":
        generate_labels_ucf(
            _cfg["DATA_DIR"],
            _cfg["ANNOTATION_FILE"],
            _cfg["OUTPUT_FILE"],
        )
    elif DATASET == "XD":
        generate_labels_xd(
            _cfg["DATA_DIR"],
            _cfg["ANNOTATION_FILE"],
            _cfg["OUTPUT_FILE"],
        )
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")


if __name__ == "__main__":
    main()