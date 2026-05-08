"""
Synthetic dataset image generation pipeline.
Generates paired normal/abnormal images via API and writes a unified pair_index.csv.
"""

import os
import json
import csv
import base64
import requests
import time
import re
from collections import Counter, defaultdict

# ======================================================================
# Configuration
# ======================================================================

API_KEY = "YOUR_API_KEY_HERE"
URL     = "YOUR_API_ENDPOINT_HERE"

LABELS_TO_GENERATE = [
    "Violent Conflict", "Crime", "Traffic Accident",
    "Personal Emergency", "Environmental Incident", "Public Misconduct",
]

RAW_JSON_DIR    = "./API_json"
BASE_DATASET_DIR = "./Syn-4img"
PAIR_INDEX_CSV  = os.path.join(BASE_DATASET_DIR, "pair_index.csv")

MAX_RETRIES       = 2
FAILED_THRESHOLD  = 3
RETRY_DELAY       = 8
TIMEOUT_PREFIX    = "TIMEOUT:"

STYLE_PREFIX = (
    "realistic photography, surveillance camera footage, photorealistic, "
    "raw photo, 2x2 grid layout, four sequential frames, "
    "left to right top to bottom."
)

PAIR_INDEX_FIELDS = ["pair_id", "source_label", "normal_path", "abnormal_path"]
SUCCESS_FIELDS    = ["pair_id", "image_name", "image_path", "Po", "label", "full_prompt"]
FAILED_FIELDS     = ["pair_id", "Po", "label", "full_prompt", "fail_reason"]


# ======================================================================
# Utilities
# ======================================================================

def assemble_prompt(shared_context, branch_data):
    pc = shared_context.get("Pc", "")
    if isinstance(pc, dict):
        scene = f"{pc.get('time', '')} {pc.get('location', '')}, {pc.get('detail', '')}".strip(", ")
    else:
        scene = str(pc)

    pchar = shared_context.get("Pchar", "")
    char_desc = str(pchar) if pchar else "characters"

    f = branch_data.get("frames", {})
    return (
        f"{STYLE_PREFIX} {scene} "
        f"Frame 1: {f.get('F1_pre', '')}; "
        f"Frame 2: {f.get('F2_start', '')}; "
        f"Frame 3: {f.get('F3_peak', '')}; "
        f"Frame 4: {f.get('F4_post', '')}. "
        f"Same {char_desc} throughout, surveillance view."
    )


def is_timeout(error_msg):
    return isinstance(error_msg, str) and error_msg.startswith(TIMEOUT_PREFIX)


def call_api(prompt):
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "1:1"},
        },
    }
    try:
        resp = requests.post(URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        candidates = result.get("candidates", [])
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []):
                if "inlineData" in part:
                    return part["inlineData"]["data"], None
                if "data" in part:
                    return part["data"], None
        return None, "NO_CANDIDATES"

    except requests.exceptions.ConnectTimeout as e:
        return None, f"{TIMEOUT_PREFIX}Connection timed out: {e}"
    except requests.exceptions.ReadTimeout as e:
        return None, f"{TIMEOUT_PREFIX}Read timed out: {e}"
    except requests.exceptions.Timeout as e:
        return None, f"{TIMEOUT_PREFIX}Request timed out: {e}"
    except Exception as e:
        msg = str(e)
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            return None, f"{TIMEOUT_PREFIX}{msg}"
        return None, msg


# ======================================================================
# State Loading
# ======================================================================

def load_pair_index(csv_path):
    """
    Load existing pair_index.csv.
    Returns a dict: pair_id -> {'normal_path': ..., 'abnormal_path': ..., 'source_label': ...}
    """
    pairs = {}
    if not os.path.exists(csv_path):
        return pairs
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pairs[row["pair_id"]] = {
                "source_label":  row["source_label"],
                "normal_path":   row["normal_path"],
                "abnormal_path": row["abnormal_path"],
            }
    return pairs


def load_label_state(label_root, label_name):
    """
    Load per-label success and failure records.

    Returns:
        success_set      : set of (pair_id, b_label) already generated
        permanently_failed: set of (pair_id, b_label) exceeding failure threshold
        next_counter     : next image index to use
    """
    csv_path    = os.path.join(label_root, f"{label_name}.csv")
    failed_path = os.path.join(label_root, f"{label_name}_failed.csv")

    success_set = set()
    max_idx     = 0

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                success_set.add((row["pair_id"], int(row["label"])))
                m = re.search(r"_(\d+)$", row["image_name"])
                if m:
                    max_idx = max(max_idx, int(m.group(1)))

    permanently_failed = set()
    if os.path.exists(failed_path):
        fail_counter = Counter()
        with open(failed_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if not is_timeout(row.get("fail_reason", "")):
                    fail_counter[(row["pair_id"], int(row["label"]))] += 1
        for key, count in fail_counter.items():
            if count >= FAILED_THRESHOLD:
                permanently_failed.add(key)

    return success_set, permanently_failed, max_idx + 1


def write_pair_index(csv_path, pairs):
    """Rewrite the full pair_index.csv from the in-memory dict."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_INDEX_FIELDS)
        writer.writeheader()
        for pair_id, info in pairs.items():
            writer.writerow({
                "pair_id":      pair_id,
                "source_label": info["source_label"],
                "normal_path":  info["normal_path"],
                "abnormal_path": info["abnormal_path"],
            })


def append_success_row(csv_path, row_dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUCCESS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def append_failed_row(csv_path, row_dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ======================================================================
# Core Pipeline
# ======================================================================

def process_branch(pair_id, b_key, branch, shared, label, label_root,
                   success_set, skip_set, counter,
                   csv_path, failed_path):
    """
    Generate one branch image (normal or abnormal).

    Returns:
        (updated_counter, rel_path_or_None, branch_type)
        rel_path is relative to BASE_DATASET_DIR, or None if generation failed/skipped.
    """
    is_ab   = "abnormal" if "abnormal" in b_key else "normal"
    b_label = 1 if is_ab == "abnormal" else 0

    if (pair_id, b_label) in success_set:
        # Already done — recover the path from the success CSV so we can
        # still update pair_index if needed.
        existing_path = _find_existing_path(csv_path, pair_id, b_label, label)
        return counter, existing_path, is_ab

    if (pair_id, b_label) in skip_set:
        print(f"  [Skip] {pair_id} ({is_ab}): exceeded failure threshold.")
        return counter, None, is_ab

    prompt   = assemble_prompt(shared, branch)
    img_name = f"{label}_{counter:04d}"
    rel_path = os.path.join(label, is_ab, f"{img_name}.jpg")
    full_path = os.path.join(BASE_DATASET_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    img_data, error = None, "UNKNOWN"
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Generating {img_name} | pair={pair_id} | {is_ab} | attempt {attempt}")
        img_data, error = call_api(prompt)
        if img_data:
            break
        if is_timeout(error):
            print(f"  Timeout (not counted toward threshold): {error}")
        time.sleep(RETRY_DELAY)

    if img_data:
        with open(full_path, "wb") as f_img:
            f_img.write(base64.b64decode(img_data))

        append_success_row(csv_path, {
            "pair_id":    pair_id,
            "image_name": img_name,
            "image_path": rel_path,
            "Po":         branch.get("Po"),
            "label":      b_label,
            "full_prompt": prompt,
        })
        print(f"  Saved: {rel_path}")
        return counter + 1, rel_path, is_ab

    else:
        tag = "Timeout" if is_timeout(error) else "Failed"
        print(f"  {tag}: {pair_id} ({is_ab}) — {error}")
        append_failed_row(failed_path, {
            "pair_id":    pair_id,
            "Po":         branch.get("Po"),
            "label":      b_label,
            "full_prompt": prompt,
            "fail_reason": error,
        })
        return counter, None, is_ab


def _find_existing_path(csv_path, pair_id, b_label, label):
    """Look up the image_path for an already-generated branch."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["pair_id"] == pair_id and int(row["label"]) == b_label:
                return row["image_path"]
    return None


def start_pipeline():
    os.makedirs(BASE_DATASET_DIR, exist_ok=True)

    # Load the shared pair_index once; update it incrementally.
    pair_index = load_pair_index(PAIR_INDEX_CSV)

    for label in LABELS_TO_GENERATE:
        print(f"\nProcessing label: [{label}]")
        label_root  = os.path.join(BASE_DATASET_DIR, label)
        csv_path    = os.path.join(label_root, f"{label}.csv")
        failed_path = os.path.join(label_root, f"{label}_failed.csv")
        os.makedirs(os.path.join(label_root, "abnormal"), exist_ok=True)
        os.makedirs(os.path.join(label_root, "normal"),   exist_ok=True)

        success_set, skip_set, counter = load_label_state(label_root, label)

        json_files = sorted(
            f for f in os.listdir(RAW_JSON_DIR)
            if f.startswith(label) and f.endswith(".json")
        )

        for j_file in json_files:
            with open(os.path.join(RAW_JSON_DIR, j_file), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    print(f"  Failed to parse {j_file}, skipping.")
                    continue

            pairs = data if isinstance(data, list) else [data]

            for pair in pairs:
                pair_id = pair.get("pair_id", "Unknown")
                shared  = pair.get("shared_context", {})

                # Ensure the pair entry exists in the index.
                if pair_id not in pair_index:
                    pair_index[pair_id] = {
                        "source_label":  label,
                        "normal_path":   "",
                        "abnormal_path": "",
                    }

                for b_key in ["abnormal_branch", "normal_branch"]:
                    if b_key not in pair:
                        continue

                    counter, rel_path, is_ab = process_branch(
                        pair_id, b_key, pair[b_key], shared, label, label_root,
                        success_set, skip_set, counter,
                        csv_path, failed_path,
                    )

                    if rel_path:
                        pair_index[pair_id][f"{is_ab}_path"] = rel_path
                        # Persist pair_index after every successful image.
                        write_pair_index(PAIR_INDEX_CSV, pair_index)

                    time.sleep(1.2)

    print(f"\nPipeline complete. pair_index.csv -> {PAIR_INDEX_CSV}")
    print(f"Total pairs indexed: {len(pair_index)}")


if __name__ == "__main__":
    start_pipeline()