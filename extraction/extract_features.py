"""
Unified multi-GPU feature extraction script for vision-language models.
Supports four dataset modes: SYN / UCF / XD / UBnormal
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict, deque
from transformers import AutoProcessor, AutoModelForImageTextToText
import warnings
import multiprocessing as mp
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import sys
import traceback

warnings.filterwarnings('ignore')

# ====================================================================
#  Configuration
# ====================================================================

DATASET = "UBnormal"  # SYN / UCF / XD / UBnormal

MODEL_PATH = ""

DATASET_CONFIGS = {
    "SYN": {
        "CSV_PATH": "",
        "IMAGE_ROOT": "",
        "FEATURE_OUT_DIR": "",
        "RESULT_CSV_PATH": "",
        "GPU_PROCESS_CONFIG": {0: 0, 1: 0, 2: 4, 3: 4},
    },
    "UCF": {
        "CSV_PATH": "",
        "IMAGE_ROOT": "",
        "FEATURE_OUT_DIR": "",
        "RESULT_CSV_PATH": "",
        "GPU_PROCESS_CONFIG": {0: 3, 1: 0, 2: 0, 3: 4},
    },
    "XD": {
        "CSV_PATH": "",
        "IMAGE_ROOT": "",
        "FEATURE_OUT_DIR": "",
        "RESULT_CSV_PATH": "",
        "GPU_PROCESS_CONFIG": {0: 1, 1: 4, 2: 4, 3: 4},
    },
    "UBnormal": {
        "CSV_PATH": "",
        "IMAGE_ROOT": "",
        "FEATURE_OUT_DIR": "",
        "RESULT_CSV_PATH": "",
        "GPU_PROCESS_CONFIG": {0: 0, 1: 3, 2: 0, 3: 3},
    },
}

assert DATASET in DATASET_CONFIGS, \
    f"DATASET must be one of {list(DATASET_CONFIGS.keys())}, got: {DATASET}"

_cfg               = DATASET_CONFIGS[DATASET]
CSV_PATH           = _cfg["CSV_PATH"]
IMAGE_ROOT         = _cfg["IMAGE_ROOT"]
FEATURE_OUT_DIR    = _cfg["FEATURE_OUT_DIR"]
RESULT_CSV_PATH    = _cfg["RESULT_CSV_PATH"]
GPU_PROCESS_CONFIG = _cfg["GPU_PROCESS_CONFIG"]

IS_SYN = (DATASET == "SYN")

PROMPT_PART1 = (
    "You are a professional video security analysis assistant. "
    "The following four consecutive video frames record the temporal "
    "evolution of the same scene.\n\n"
    "【Anomaly Whitelist】\n"
    "Anomalous events are limited to the following 6 categories:\n"
    "[Violent Conflict], [Crime], [Traffic Accident], "
    "[Personal Emergency], [Environmental Hazard], [Public Misconduct].\n\n"
    "With the above classification criteria in mind, carefully observe "
    "the following frames to determine whether a matching anomalous "
    "event is present:"
)

PROMPT_PART2 = (
    "\nBased on the above frames, strictly follow the 4-step output "
    "format below (always start with 'Yes' or 'No'):\n\n"
    "1. Final determination: [Yes or No].\n"
    "2. Anomaly category match: [Format: Category - Specific sub-label. "
    "If No, output: None].\n"
    "3. Spatiotemporal action description: [Briefly describe character "
    "interactions, action continuity, and object state changes over time].\n"
    "4. Confidence assessment: [High / Medium / Low. If category is "
    "'None', output: None]."
)

GRID_ROWS = 2
GRID_COLS = 2
RESIZE_SUBIMAGES = True
TARGET_WIDTH  = 336
TARGET_HEIGHT = 336

K_FRAMES = 4
USE_FP16 = False
RANDOM_SEED = 42
ENABLE_THINKING = False
HIDDEN_DIM = 4096
TOTAL_MODEL_LAYERS = 32

EXTRACT_LAYER = 16

KEEP_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values", "image_grid_thw",
    "pixel_values_videos", "video_grid_thw",
    "position_ids",
}

PREFETCH_THREADS = 3
PIN_MEMORY = True
REFRESH_INTERVAL = 5

WORKER_TIMEOUT     = 300
HEARTBEAT_INTERVAL = 30
QUEUE_TIMEOUT      = 30


# ====================================================================
#  Utilities
# ====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    else:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


# ====================================================================
#  Vision Token Localization
# ====================================================================

def find_vision_mask(input_ids, processor, k_frames=K_FRAMES):
    start_candidates = ['<|vision_start|>', '<|img_start|>']
    end_candidates   = ['<|vision_end|>',   '<|img_end|>']

    for start_name, end_name in zip(start_candidates, end_candidates):
        start_id = processor.tokenizer.convert_tokens_to_ids(start_name)
        end_id   = processor.tokenizer.convert_tokens_to_ids(end_name)
        if (start_id != processor.tokenizer.unk_token_id and
                end_id != processor.tokenizer.unk_token_id):
            start_positions = (input_ids == start_id).nonzero(as_tuple=True)[0]
            end_positions   = (input_ids == end_id).nonzero(as_tuple=True)[0]
            n_images = min(len(start_positions), len(end_positions))
            if n_images >= k_frames:
                all_vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for i in range(k_frames):
                    s = start_positions[i].item()
                    e = end_positions[i].item()
                    all_vision_mask[s + 1: e] = True
                if all_vision_mask.any():
                    vis_positions = all_vision_mask.nonzero(as_tuple=True)[0]
                    vis_last_pos  = vis_positions[-1].item()
                    return all_vision_mask, vis_last_pos

    candidate_tokens = ['<|image_pad|>', '<|vision_pad|>', '<|img_pad|>']
    for token_name in candidate_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token_name)
        if token_id != processor.tokenizer.unk_token_id:
            all_vision_mask = (input_ids == token_id)
            if all_vision_mask.any():
                vis_positions = all_vision_mask.nonzero(as_tuple=True)[0]
                vis_last_pos  = vis_positions[-1].item()
                return all_vision_mask, vis_last_pos

    return None, None


# ====================================================================
#  Feature Extraction (single layer, last_token + vis_last only)
# ====================================================================

def extract_features(outputs, input_ids_cpu, processor,
                     extract_layer, k_frames):
    _, vis_last_pos = find_vision_mask(
        input_ids_cpu, processor, k_frames
    )

    n_states = len(outputs.hidden_states)

    if extract_layer < n_states:
        h = outputs.hidden_states[extract_layer][0]
        last_token = h[-1, :]
        if vis_last_pos is not None:
            vis_last = h[vis_last_pos, :]
        else:
            vis_last = last_token.clone()
    else:
        device = outputs.hidden_states[-1].device
        dtype  = outputs.hidden_states[-1].dtype
        last_token = torch.zeros(HIDDEN_DIM, device=device, dtype=dtype)
        vis_last   = torch.zeros(HIDDEN_DIM, device=device, dtype=dtype)

    return {
        'feat_last_token': last_token,
        'feat_vis_last':   vis_last,
    }


# ====================================================================
#  Preprocessing: PIL images -> model input tensors
# ====================================================================

def _tokenize_images(processor, images, use_pin_memory=True):
    content = [{"type": "text", "text": PROMPT_PART1}]
    for img in images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_PART2})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = processor(
        text=[text], images=images,
        padding=True, return_tensors="pt",
    )

    input_ids_cpu = inputs['input_ids'][0].clone()

    inputs_cpu = {}
    for k, v in inputs.items():
        if k in KEEP_KEYS and isinstance(v, torch.Tensor):
            if use_pin_memory and v.is_floating_point():
                inputs_cpu[k] = v.pin_memory()
            else:
                inputs_cpu[k] = v

    return inputs_cpu, input_ids_cpu


# ====================================================================
#  SYN: grid splitting + data preparation
# ====================================================================

def split_grid(image_path, rows=GRID_ROWS, cols=GRID_COLS):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    cell_w, cell_h = W // cols, H // rows
    crops = []
    for r in range(rows):
        for c in range(cols):
            box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
            crops.append(img.crop(box))
    return crops


def resize_subimage(img, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
    if target_w <= 0 and target_h <= 0:
        return img
    W, H = img.size
    if target_w <= 0:
        target_w = int(W * target_h / H)
    if target_h <= 0:
        target_h = int(H * target_w / W)
    return img.resize((target_w, target_h), Image.LANCZOS)


def _prepare_single_branch(processor, image_path, use_pin_memory=True):
    sub_images = split_grid(image_path, GRID_ROWS, GRID_COLS)
    if RESIZE_SUBIMAGES:
        sub_images = [resize_subimage(img, TARGET_WIDTH, TARGET_HEIGHT)
                      for img in sub_images]
    return _tokenize_images(processor, sub_images, use_pin_memory)


def prepare_syn_tasks(pair_index_csv, base_dataset_dir):
    df = pd.read_csv(pair_index_csv)
    tasks = []
    skipped_missing = 0

    for _, row in df.iterrows():
        pair_id = row['pair_id']
        source_label = row['source_label']
        branches = []

        if pd.notna(row.get('normal_path', None)) and str(row['normal_path']).strip():
            normal_abs = os.path.join(base_dataset_dir, str(row['normal_path']))
            if os.path.exists(normal_abs):
                branches.append({
                    'branch_type': 'normal',
                    'image_path': normal_abs,
                    'label': 'Normal',
                })
            else:
                skipped_missing += 1

        if pd.notna(row.get('abnormal_path', None)) and str(row['abnormal_path']).strip():
            abnormal_abs = os.path.join(base_dataset_dir, str(row['abnormal_path']))
            if os.path.exists(abnormal_abs):
                branches.append({
                    'branch_type': 'abnormal',
                    'image_path': abnormal_abs,
                    'label': source_label,
                })
            else:
                skipped_missing += 1

        if branches:
            tasks.append({
                'pair_id': pair_id,
                'source_label': source_label,
                'branches': branches,
                'n_items': len(branches),
            })

    if skipped_missing > 0:
        print(f"  Skipped {skipped_missing} branches (file not found)")

    return tasks


# ====================================================================
#  UCF / XD / UBnormal: frame parsing + data preparation
# ====================================================================

def parse_frame_name(frame_name):
    stem = Path(frame_name).stem
    clip_pos = stem.rfind('_clip')
    if clip_pos == -1:
        raise ValueError(f"Invalid frame name format: {frame_name}")
    video_name = stem[:clip_pos]
    clip_frame_part = stem[clip_pos + 5:]
    parts = clip_frame_part.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid frame name format: {frame_name}")
    clip_idx  = int(parts[0])
    frame_idx = int(parts[1])
    return video_name, clip_idx, frame_idx


def group_frames_by_video_and_clip(csv_path):
    df = pd.read_csv(csv_path)
    video_dict = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        frame_name = row['frame_name']
        label = row['label']
        try:
            video_name, clip_idx, frame_idx = parse_frame_name(frame_name)
            video_dict[video_name][clip_idx].append((frame_name, frame_idx, label))
        except ValueError:
            continue
    for video_name in video_dict:
        for clip_idx in video_dict[video_name]:
            video_dict[video_name][clip_idx] = sorted(
                video_dict[video_name][clip_idx], key=lambda x: x[1]
            )
    return video_dict


def prepare_video_tasks(video_dict, image_root, k_frames=4):
    tasks = []
    image_root = Path(image_root)
    for video_name, clips_dict in video_dict.items():
        video_clips = []
        clip_labels = []
        for clip_idx in sorted(clips_dict.keys()):
            frame_list = list(clips_dict[clip_idx])
            if len(frame_list) == 0:
                continue
            while len(frame_list) < k_frames:
                frame_list.append(frame_list[-1])
            frame_list = frame_list[:k_frames]
            frame_paths = []
            for frame_name, _, _ in frame_list:
                img_path = image_root / frame_name
                if img_path.exists():
                    frame_paths.append(str(img_path))
            if len(frame_paths) == 0:
                continue
            while len(frame_paths) < k_frames:
                frame_paths.append(frame_paths[-1])
            if len(frame_paths) == k_frames:
                clip_label = frame_list[0][2]
                clip_labels.append(clip_label)
                video_clips.append({
                    'clip_idx': clip_idx,
                    'frame_paths': frame_paths,
                    'label': clip_label,
                })
        if video_clips:
            tasks.append({
                'video_name': video_name,
                'clips': video_clips,
                'clip_labels': clip_labels,
                'n_items': len(video_clips),
            })
    return tasks


def _prepare_single_clip(processor, frame_paths, use_pin_memory=True):
    sub_images = [Image.open(fp).convert("RGB") for fp in frame_paths]
    return _tokenize_images(processor, sub_images, use_pin_memory)


def split_tasks_for_workers(tasks_to_process, total_workers):
    if len(tasks_to_process) >= total_workers:
        return tasks_to_process, {}

    queue_tasks    = []
    pending_merges = {}

    for task in tasks_to_process:
        n_clips = task['n_items']
        if n_clips >= total_workers * 2 and len(tasks_to_process) < total_workers:
            clips       = task['clips']
            clip_labels = task['clip_labels']
            n_chunks    = total_workers
            chunk_size  = n_clips // n_chunks
            remainder   = n_clips % n_chunks

            merge_info = {
                'video_name':      task['video_name'],
                'clip_labels':     clip_labels,
                'n_clips':         n_clips,
                'n_chunks':        n_chunks,
                'chunks_received': 0,
                'chunk_paths':     {},
            }

            offset = 0
            for chunk_id in range(n_chunks):
                this_chunk_size = chunk_size + (1 if chunk_id < remainder else 0)
                chunk_clips = clips[offset: offset + this_chunk_size]
                offset += this_chunk_size

                chunk_task = {
                    'video_name': task['video_name'],
                    'clips': chunk_clips,
                    'clip_labels': [c['label'] for c in chunk_clips],
                    'n_items': task['n_items'],
                    'is_chunk': True,
                    'chunk_id': chunk_id,
                }
                queue_tasks.append(chunk_task)

            pending_merges[task['video_name']] = merge_info
            print(f"  Split '{task['video_name']}' ({n_clips} clips) into {n_chunks} chunks")
        else:
            queue_tasks.append(task)

    return queue_tasks, pending_merges


def merge_chunks(merge_info, feature_out_dir, csv_writer):
    video_name  = merge_info['video_name']
    n_chunks    = merge_info['n_chunks']
    chunk_paths = merge_info['chunk_paths']
    clip_labels = merge_info['clip_labels']
    n_clips     = merge_info['n_clips']

    all_feat_last_token = []
    all_feat_vis_last   = []

    for chunk_id in range(n_chunks):
        tmp_path   = chunk_paths[chunk_id]
        chunk_data = torch.load(tmp_path, map_location='cpu', weights_only=True)
        all_feat_last_token.append(chunk_data['feat_last_token'])
        all_feat_vis_last.append(chunk_data['feat_vis_last'])
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    video_data = {
        'feat_last_token': torch.cat(all_feat_last_token, dim=0),
        'feat_vis_last':   torch.cat(all_feat_vis_last, dim=0),
    }

    pt_filename = f"{video_name}.pt"
    pt_path     = os.path.join(feature_out_dir, pt_filename)
    torch.save(video_data, pt_path)

    label_str  = ','.join(map(str, clip_labels))
    feat_shape = str(list(video_data['feat_last_token'].shape))

    csv_writer.add_record({
        'pt_name':       pt_filename,
        'label':         label_str,
        'n_clips':       n_clips,
        'feature_shape': feat_shape,
    })

    print(f"\n  Merged {n_chunks} chunks -> {pt_filename} (shape: {feat_shape})")


# ====================================================================
#  CSV Writer
# ====================================================================

class CSVWriter:
    def __init__(self, csv_path, columns):
        self.csv_path = csv_path
        self.columns  = columns
        self.lock     = threading.Lock()
        self.records  = []
        self.items_saved = 0

        if os.path.exists(csv_path):
            try:
                existing_df  = pd.read_csv(csv_path)
                self.records = existing_df.to_dict('records')
                for r in self.records:
                    count_key = 'n_branches' if IS_SYN else 'n_clips'
                    if count_key in r:
                        self.items_saved += r[count_key]
                print(f"Loaded {len(self.records)} existing records "
                      f"({self.items_saved} items) from CSV")
            except Exception:
                self.records = []

        csv_dir = os.path.dirname(csv_path)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

    def add_record(self, record):
        with self.lock:
            self.records.append(record)
            count_key = 'n_branches' if IS_SYN else 'n_clips'
            if count_key in record:
                self.items_saved += record[count_key]
            self._write_to_file()

    def _write_to_file(self):
        df = pd.DataFrame(self.records, columns=self.columns)
        df.to_csv(self.csv_path, index=False)

    def get_existing_pt_names(self):
        with self.lock:
            return set(r.get('pt_name', '') for r in self.records)

    def get_record_count(self):
        with self.lock:
            return len(self.records)

    def get_items_saved(self):
        with self.lock:
            return self.items_saved


# ====================================================================
#  Progress Tracker
# ====================================================================

class ProgressTracker:
    def __init__(self, initial_items_done=0):
        self.lock = threading.Lock()
        self.worker_status = {}
        self.items_processed_this_run = 0
        self.tasks_completed = 0
        self.initial_items_done = initial_items_done

    def worker_start_task(self, worker_id, task_name, n_items):
        with self.lock:
            self.worker_status[worker_id] = {
                'task':           task_name,
                'current_item':   0,
                'total_items':    n_items,
                'last_heartbeat': time.time(),
            }

    def worker_item_done(self, worker_id, item_idx):
        with self.lock:
            if worker_id in self.worker_status:
                self.worker_status[worker_id]['current_item'] = item_idx + 1
                self.worker_status[worker_id]['last_heartbeat'] = time.time()
                self.items_processed_this_run += 1

    def worker_task_done(self, worker_id):
        with self.lock:
            if worker_id in self.worker_status:
                del self.worker_status[worker_id]
            self.tasks_completed += 1

    def worker_heartbeat(self, worker_id):
        with self.lock:
            if worker_id in self.worker_status:
                self.worker_status[worker_id]['last_heartbeat'] = time.time()

    def get_status(self):
        with self.lock:
            active_workers = len(self.worker_status)
            current_items  = sum(s['current_item'] for s in self.worker_status.values())
            total_items    = sum(s['total_items']   for s in self.worker_status.values())
            return {
                'active_workers':          active_workers,
                'current_items_in_progress': current_items,
                'total_items_in_progress':   total_items,
                'items_processed_this_run':  self.items_processed_this_run,
                'worker_details':            dict(self.worker_status),
            }

    def check_stuck_workers(self, timeout=WORKER_TIMEOUT):
        with self.lock:
            current_time   = time.time()
            stuck_workers  = []
            for worker_id, status in self.worker_status.items():
                if current_time - status['last_heartbeat'] > timeout:
                    stuck_workers.append((worker_id, status['task'],
                                          current_time - status['last_heartbeat']))
            return stuck_workers


# ====================================================================
#  Stacking Helper
# ====================================================================

def _stack_item_dicts(item_dicts):
    return {
        'feat_last_token': torch.stack(
            [d['feat_last_token'] for d in item_dicts], dim=0),
        'feat_vis_last': torch.stack(
            [d['feat_vis_last'] for d in item_dicts], dim=0),
    }


# ====================================================================
#  Worker: SYN mode
# ====================================================================

def _worker_syn(worker_id, gpu_id, task_queue, result_queue,
                feature_out_dir, extract_layer, seed,
                processor, model, device):
    prefetch_pool = ThreadPoolExecutor(max_workers=PREFETCH_THREADS)
    success_count = 0
    error_count   = 0
    prefetch_queue = deque()
    first_logged = False

    def _prefetch_task(task):
        futures = []
        for branch in task['branches']:
            f = prefetch_pool.submit(
                _prepare_single_branch,
                processor, branch['image_path'], PIN_MEMORY,
            )
            futures.append(f)
        return futures

    def _try_prefetch_next():
        try:
            next_task = task_queue.get_nowait()
            if next_task is not None:
                futures = _prefetch_task(next_task)
                prefetch_queue.append((next_task, futures))
            else:
                prefetch_queue.append((None, None))
        except Exception:
            pass

    with torch.no_grad():
        for _ in range(2):
            _try_prefetch_next()

        while True:
            if prefetch_queue:
                task, branch_futures = prefetch_queue.popleft()
            else:
                try:
                    task = task_queue.get(timeout=5)
                except Exception:
                    if task_queue.empty():
                        break
                    continue
                branch_futures = None

            if task is None:
                break

            _try_prefetch_next()

            if branch_futures is None:
                branch_futures = _prefetch_task(task)

            pair_id      = task['pair_id']
            source_label = task['source_label']
            branches     = task['branches']
            n_branches   = task['n_items']

            try:
                result_queue.put(('task_start', worker_id, pair_id, n_branches))

                all_branch_dicts = []
                branch_labels    = []
                branch_types     = []
                last_reported    = 0

                for bi, (branch, future) in enumerate(zip(branches, branch_futures)):
                    inputs_cpu, input_ids_cpu = future.result()
                    inputs_gpu = {k: v.to(device, non_blocking=True)
                                  for k, v in inputs_cpu.items()}

                    outputs = model(**inputs_gpu,
                                    output_hidden_states=True,
                                    return_dict=True)

                    if outputs.hidden_states is None:
                        raise RuntimeError("hidden_states not returned")

                    if not first_logged:
                        first_logged = True
                        n_states = len(outputs.hidden_states)
                        all_vis_mask, _ = find_vision_mask(
                            input_ids_cpu, processor, K_FRAMES)
                        seq_len = input_ids_cpu.shape[0]
                        if all_vis_mask is not None and all_vis_mask.any():
                            n_vision = all_vis_mask.sum().item()
                            vis_pos  = all_vis_mask.nonzero(as_tuple=True)[0]
                            print(f"[Worker {worker_id}] "
                                  f"seq_len={seq_len}, "
                                  f"vision_tokens={n_vision} "
                                  f"(pos {vis_pos[0].item()}~{vis_pos[-1].item()}), "
                                  f"hs_count={n_states}")
                        else:
                            print(f"[Worker {worker_id}] "
                                  f"seq_len={seq_len}, vision tokens not found, "
                                  f"hs_count={n_states}")

                    feat_dict = extract_features(
                        outputs=outputs,
                        input_ids_cpu=input_ids_cpu,
                        processor=processor,
                        extract_layer=extract_layer,
                        k_frames=K_FRAMES,
                    )

                    feat_cpu = {
                        'feat_last_token': feat_dict['feat_last_token'].cpu().float(),
                        'feat_vis_last':   feat_dict['feat_vis_last'].cpu().float(),
                    }
                    all_branch_dicts.append(feat_cpu)
                    branch_labels.append(branch['label'])
                    branch_types.append(branch['branch_type'])

                    current_done = bi + 1
                    if (current_done - last_reported >= REFRESH_INTERVAL
                            or current_done == n_branches):
                        result_queue.put(('item_progress', worker_id, pair_id,
                                          current_done, n_branches))
                        last_reported = current_done

                    del outputs, inputs_gpu, inputs_cpu

                pair_data = _stack_item_dicts(all_branch_dicts)

                pt_filename = f"{pair_id}.pt"
                pt_path = os.path.join(feature_out_dir, pt_filename)
                torch.save(pair_data, pt_path)

                label_str  = ','.join(
                    f"{bt}:{bl}" for bt, bl in zip(branch_types, branch_labels))
                feat_shape = str(list(pair_data['feat_last_token'].shape))

                result_queue.put(('success', worker_id, {
                    'pt_name':       pt_filename,
                    'pair_id':       pair_id,
                    'source_label':  source_label,
                    'branch_info':   label_str,
                    'n_branches':    n_branches,
                    'feature_shape': feat_shape,
                }))
                success_count += 1

            except Exception as e:
                error_count += 1
                result_queue.put(('error', worker_id, pair_id, str(e)))
                traceback.print_exc()
                for f in branch_futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass

            if (success_count + error_count) % 20 == 0:
                torch.cuda.empty_cache()

    prefetch_pool.shutdown(wait=False)
    return success_count, error_count


# ====================================================================
#  Worker: UCF / XD / UBnormal mode
# ====================================================================

def _worker_test(worker_id, gpu_id, task_queue, result_queue,
                 feature_out_dir, extract_layer, seed,
                 processor, model, device):
    prefetch_pool     = ThreadPoolExecutor(max_workers=PREFETCH_THREADS)
    success_count     = 0
    error_count       = 0
    first_clip_logged = False

    with torch.no_grad():
        while True:
            try:
                task = task_queue.get(timeout=QUEUE_TIMEOUT)
            except Exception:
                if task_queue.empty():
                    break
                result_queue.put(('heartbeat', worker_id))
                continue

            if task is None:
                break

            video_name = task['video_name']
            clips      = task['clips']
            n_clips    = len(clips)
            is_chunk   = task.get('is_chunk', False)
            chunk_id   = task.get('chunk_id', 0)

            try:
                result_queue.put(('task_start', worker_id, video_name, n_clips))
                prefetch_futures = deque()

                for i in range(min(PREFETCH_THREADS, n_clips)):
                    future = prefetch_pool.submit(
                        _prepare_single_clip, processor,
                        clips[i]['frame_paths'], PIN_MEMORY,
                    )
                    prefetch_futures.append(future)

                all_clip_dicts = []

                for clip_idx, clip_info in enumerate(clips):
                    next_prefetch_idx = clip_idx + PREFETCH_THREADS
                    if next_prefetch_idx < n_clips:
                        future = prefetch_pool.submit(
                            _prepare_single_clip, processor,
                            clips[next_prefetch_idx]['frame_paths'], PIN_MEMORY,
                        )
                        prefetch_futures.append(future)

                    inputs_cpu, input_ids_cpu = prefetch_futures.popleft().result()
                    inputs_gpu = {k: v.to(device, non_blocking=True)
                                  for k, v in inputs_cpu.items()}

                    outputs = model(**inputs_gpu,
                                    output_hidden_states=True,
                                    return_dict=True)

                    if outputs.hidden_states is None:
                        raise RuntimeError("hidden_states not returned")

                    if not first_clip_logged:
                        first_clip_logged = True
                        n_states = len(outputs.hidden_states)
                        all_vis_mask, _ = find_vision_mask(
                            input_ids_cpu, processor, K_FRAMES)
                        seq_len = input_ids_cpu.shape[0]
                        if all_vis_mask is not None and all_vis_mask.any():
                            n_vision = all_vis_mask.sum().item()
                            vis_pos  = all_vis_mask.nonzero(as_tuple=True)[0]
                            print(f"[Worker {worker_id}] "
                                  f"seq_len={seq_len}, "
                                  f"vision_tokens={n_vision} "
                                  f"(pos {vis_pos[0].item()}~{vis_pos[-1].item()}), "
                                  f"hs_count={n_states}")
                        else:
                            print(f"[Worker {worker_id}] "
                                  f"seq_len={seq_len}, vision tokens not found, "
                                  f"hs_count={n_states}")

                    feat_dict = extract_features(
                        outputs=outputs,
                        input_ids_cpu=input_ids_cpu,
                        processor=processor,
                        extract_layer=extract_layer,
                        k_frames=K_FRAMES,
                    )

                    feat_cpu = {
                        'feat_last_token': feat_dict['feat_last_token'].cpu().float(),
                        'feat_vis_last':   feat_dict['feat_vis_last'].cpu().float(),
                    }
                    all_clip_dicts.append(feat_cpu)

                    result_queue.put(('item_done', worker_id, video_name,
                                      clip_idx, n_clips))
                    del outputs, inputs_gpu, inputs_cpu

                    if (clip_idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()

                if not is_chunk:
                    clip_labels = task['clip_labels']
                    video_data  = _stack_item_dicts(all_clip_dicts)

                    pt_filename = f"{video_name}.pt"
                    pt_path     = os.path.join(feature_out_dir, pt_filename)
                    torch.save(video_data, pt_path)

                    label_str  = ','.join(map(str, clip_labels))
                    feat_shape = str(list(video_data['feat_last_token'].shape))

                    result_queue.put(('success', worker_id, {
                        'pt_name':       pt_filename,
                        'label':         label_str,
                        'n_clips':       task['n_items'],
                        'feature_shape': feat_shape,
                    }))
                    success_count += 1
                else:
                    tmp_filename = f"{video_name}_chunk{chunk_id}.pt"
                    tmp_path     = os.path.join(feature_out_dir, tmp_filename)
                    chunk_data   = _stack_item_dicts(all_clip_dicts)
                    torch.save(chunk_data, tmp_path)

                    result_queue.put(('chunk_done', worker_id, {
                        'video_name':       video_name,
                        'chunk_id':         chunk_id,
                        'tmp_path':         tmp_path,
                        'n_clips_in_chunk': n_clips,
                    }))
                    success_count += 1

            except Exception as e:
                error_count += 1
                result_queue.put(('error', worker_id, video_name, str(e)))
                traceback.print_exc()
                for f in prefetch_futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                prefetch_futures.clear()

            if (success_count + error_count) % 5 == 0:
                torch.cuda.empty_cache()

    prefetch_pool.shutdown(wait=False)
    return success_count, error_count


# ====================================================================
#  Worker Process Entry
# ====================================================================

def worker_process(worker_id, gpu_id, task_queue, result_queue,
                   feature_out_dir, extract_layer, seed, dataset_mode):
    set_seed(seed + worker_id)
    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id} (mode={dataset_mode})")

    try:
        dtype = torch.float16 if USE_FP16 else torch.bfloat16

        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True)

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=dtype, device_map=device, trust_remote_code=True)
        model.eval()

        mem_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
        print(f"[Worker {worker_id}] Model loaded on GPU {gpu_id}, "
              f"Memory: {mem_gb:.2f} GB")

        if dataset_mode == "SYN":
            success_count, error_count = _worker_syn(
                worker_id, gpu_id, task_queue, result_queue,
                feature_out_dir, extract_layer, seed,
                processor, model, device)
        else:
            success_count, error_count = _worker_test(
                worker_id, gpu_id, task_queue, result_queue,
                feature_out_dir, extract_layer, seed,
                processor, model, device)

        result_queue.put(('done', worker_id, success_count, error_count))
        print(f"[Worker {worker_id}] Finished: "
              f"{success_count} success, {error_count} errors")

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        traceback.print_exc()
        result_queue.put(('fatal', worker_id, str(e)))


# ====================================================================
#  Progress Monitor
# ====================================================================

def progress_monitor(result_queue, total_tasks_to_process, total_items_to_process,
                     total_tasks_all, total_items_all,
                     already_done_tasks, already_done_items,
                     num_workers, csv_writer,
                     pending_merges, feature_out_dir):
    task_label = "Pairs" if IS_SYN else "Videos"
    item_label = "Branches" if IS_SYN else "Clips"

    finished_workers    = 0
    total_success       = 0
    total_errors        = 0
    tracker             = ProgressTracker(initial_items_done=already_done_items)
    start_time          = time.time()
    last_print_time     = 0
    last_stuck_check    = time.time()

    def print_progress(force=False):
        nonlocal last_print_time
        current_time = time.time()
        if not force and current_time - last_print_time < 0.3:
            return
        last_print_time = current_time

        elapsed = current_time - start_time
        tasks_done_this_run = csv_writer.get_record_count() - already_done_tasks
        status = tracker.get_status()
        items_in_progress    = status['current_items_in_progress']
        total_in_progress    = status['total_items_in_progress']
        active_workers       = status['active_workers']
        items_done_this_run  = status['items_processed_this_run']

        if items_done_this_run > 0:
            eta     = elapsed / items_done_this_run * (total_items_to_process - items_done_this_run)
            eta_str = format_time(eta)
        else:
            eta_str = "???"

        pct1     = tasks_done_this_run / total_tasks_to_process if total_tasks_to_process > 0 else 0
        bar1_len = 15
        bar1     = "#" * int(bar1_len * pct1) + "-" * (bar1_len - int(bar1_len * pct1))

        pct2     = items_done_this_run / total_items_to_process if total_items_to_process > 0 else 0
        bar2_len = 15
        bar2     = "#" * int(bar2_len * pct2) + "-" * (bar2_len - int(bar2_len * pct2))

        total_tasks_done = already_done_tasks + tasks_done_this_run
        total_items_done = already_done_items + items_done_this_run

        line = (
            f"\rTasks {tasks_done_this_run}/{total_tasks_to_process} [{bar1}] "
            f"[{format_time(elapsed)}<{eta_str}] | "
            f"Items {items_done_this_run}/{total_items_to_process} [{bar2}] "
            f"({active_workers}w, {items_in_progress}/{total_in_progress}) | "
            f"Total: {total_tasks_done}/{total_tasks_all}t, "
            f"{total_items_done}/{total_items_all}i"
        )
        sys.stdout.write(line + " " * 5)
        sys.stdout.flush()

    def check_stuck():
        nonlocal last_stuck_check
        current_time = time.time()
        if current_time - last_stuck_check < 60:
            return
        last_stuck_check = current_time
        stuck = tracker.check_stuck_workers(timeout=WORKER_TIMEOUT)
        if stuck:
            sys.stdout.write(f"\nStuck workers detected: {stuck}\n")
            sys.stdout.flush()

    while finished_workers < num_workers:
        try:
            msg = result_queue.get(timeout=5)

            if msg[0] == 'task_start':
                _, wid, task_name, n_items = msg
                tracker.worker_start_task(wid, task_name, n_items)
                print_progress()

            elif msg[0] == 'item_progress':
                _, wid, _name, current_done, _total = msg
                tracker.worker_item_done(wid, current_done - 1)
                print_progress()

            elif msg[0] == 'item_done':
                _, wid, _name, clip_idx, _total = msg
                tracker.worker_item_done(wid, clip_idx)
                print_progress()

            elif msg[0] == 'heartbeat':
                _, wid = msg
                tracker.worker_heartbeat(wid)

            elif msg[0] == 'success':
                _, wid, record = msg
                tracker.worker_task_done(wid)
                csv_writer.add_record(record)
                total_success += 1
                print_progress(force=True)

            elif msg[0] == 'chunk_done':
                _, wid, chunk_info = msg
                tracker.worker_task_done(wid)
                vname    = chunk_info['video_name']
                cid      = chunk_info['chunk_id']
                tmp_path = chunk_info['tmp_path']

                if vname in pending_merges:
                    mi = pending_merges[vname]
                    mi['chunk_paths'][cid] = tmp_path
                    mi['chunks_received'] += 1
                    if mi['chunks_received'] == mi['n_chunks']:
                        merge_chunks(mi, feature_out_dir, csv_writer)
                        del pending_merges[vname]
                        total_success += 1
                print_progress(force=True)

            elif msg[0] == 'error':
                _, wid, task_name, errmsg = msg
                tracker.worker_task_done(wid)
                total_errors += 1
                sys.stdout.write(f"\nError [{task_name}]: {errmsg[:100]}\n")
                sys.stdout.flush()
                print_progress(force=True)

            elif msg[0] == 'done':
                _, wid, s, e = msg
                finished_workers += 1
                sys.stdout.write(
                    f"\nWorker {wid} done ({s} ok, {e} err)\n")
                sys.stdout.flush()
                print_progress()

            elif msg[0] == 'fatal':
                _, wid, errmsg = msg
                finished_workers += 1
                sys.stdout.write(f"\nWorker {wid} fatal: {errmsg}\n")
                sys.stdout.flush()

            check_stuck()

        except Exception:
            print_progress()
            check_stuck()

    print()
    print(f"\nFinal CSV saved to: {csv_writer.csv_path}")
    print(f"Total records: {csv_writer.get_record_count()}")
    return total_success, total_errors, csv_writer.get_items_saved()


# ====================================================================
#  Main
# ====================================================================

def main():
    set_seed(RANDOM_SEED)
    total_workers = sum(GPU_PROCESS_CONFIG.values())

    task_label = "Pairs" if IS_SYN else "Videos"
    item_label = "Branches" if IS_SYN else "Clips"

    print("=" * 70)
    print(f"  Feature Extraction (Dataset: {DATASET})")
    print(f"  Features: last_token + vis_last (single layer hs[{EXTRACT_LAYER}])")
    print("=" * 70)
    print(f"  GPU config: {GPU_PROCESS_CONFIG} ({total_workers} workers)")
    print(f"  Precision: {'float16' if USE_FP16 else 'bfloat16'}")
    print(f"  Extract layer: hs[{EXTRACT_LAYER}] -> {HIDDEN_DIM} dim")
    print(f"  Per-sample sub-features:")
    print(f"    last_token : last sequence token ({HIDDEN_DIM}d)")
    print(f"    vis_last   : last vision token   ({HIDDEN_DIM}d)")
    print("=" * 70)

    os.makedirs(FEATURE_OUT_DIR, exist_ok=True)

    if IS_SYN:
        csv_columns = ['pt_name', 'pair_id', 'source_label', 'branch_info',
                        'n_branches', 'feature_shape']
    else:
        csv_columns = ['pt_name', 'label', 'n_clips', 'feature_shape']

    csv_writer = CSVWriter(RESULT_CSV_PATH, columns=csv_columns)

    print(f"\n[1/4] Loading data ({DATASET})...")

    if IS_SYN:
        all_tasks = prepare_syn_tasks(CSV_PATH, IMAGE_ROOT)
        id_key = 'pair_id'
    else:
        video_dict = group_frames_by_video_and_clip(CSV_PATH)
        all_tasks  = prepare_video_tasks(video_dict, IMAGE_ROOT, k_frames=K_FRAMES)
        id_key = 'video_name'

    total_tasks_all = len(all_tasks)
    total_items_all = sum(t['n_items'] for t in all_tasks)
    print(f"  Total {task_label.lower()}: {total_tasks_all}")
    print(f"  Total {item_label.lower()}: {total_items_all}")

    existing_pt  = set(os.listdir(FEATURE_OUT_DIR)) if os.path.exists(FEATURE_OUT_DIR) else set()
    existing_csv = csv_writer.get_existing_pt_names()

    tasks_to_process   = []
    items_to_process   = 0
    already_done_items = 0

    for task in all_tasks:
        pt_fn = f"{task[id_key]}.pt"
        if pt_fn in existing_pt and pt_fn in existing_csv:
            already_done_items += task['n_items']
            continue
        tasks_to_process.append(task)
        items_to_process += task['n_items']

    already_done_tasks = total_tasks_all - len(tasks_to_process)

    print(f"  Already done: {already_done_tasks} {task_label.lower()} "
          f"({already_done_items} {item_label.lower()})")
    print(f"  To process:   {len(tasks_to_process)} {task_label.lower()} "
          f"({items_to_process} {item_label.lower()})")

    if not tasks_to_process:
        print(f"\nAll {task_label.lower()} already processed!")
        return

    print(f"\n[2/4] Creating task queue...")

    pending_merges = {}

    if IS_SYN:
        queue_tasks = tasks_to_process
        queue_tasks.sort(key=lambda x: x['n_items'], reverse=True)
    else:
        queue_tasks, pending_merges = split_tasks_for_workers(
            tasks_to_process, total_workers)
        non_chunk = [t for t in queue_tasks if not t.get('is_chunk', False)]
        chunk_lst = [t for t in queue_tasks if t.get('is_chunk', False)]
        non_chunk.sort(key=lambda x: x.get('n_items', 0), reverse=True)
        queue_tasks = chunk_lst + non_chunk
        print(f"  Queue: {len(non_chunk)} whole-video tasks + "
              f"{len(chunk_lst)} chunk tasks")

    manager      = Manager()
    task_queue    = manager.Queue()
    result_queue  = manager.Queue()

    for task in queue_tasks:
        task_queue.put(task)
    for _ in range(total_workers):
        task_queue.put(None)

    if pending_merges:
        print(f"  Pending merges: {list(pending_merges.keys())}")

    print(f"\n[3/4] Starting {total_workers} workers...")
    processes = []
    wid = 0
    for gpu_id, np_ in GPU_PROCESS_CONFIG.items():
        for pi in range(np_):
            p = Process(
                target=worker_process,
                args=(wid, gpu_id, task_queue, result_queue,
                      FEATURE_OUT_DIR, EXTRACT_LAYER, RANDOM_SEED, DATASET),
            )
            p.start()
            processes.append(p)
            print(f"  Worker {wid} -> GPU {gpu_id} ({pi + 1}/{np_})")
            wid += 1
            time.sleep(3)

    print(f"\n[4/4] Extracting features...\n")

    total_success, total_errors, total_items_saved = progress_monitor(
        result_queue=result_queue,
        total_tasks_to_process=len(tasks_to_process),
        total_items_to_process=items_to_process,
        total_tasks_all=total_tasks_all,
        total_items_all=total_items_all,
        already_done_tasks=already_done_tasks,
        already_done_items=already_done_items,
        num_workers=total_workers,
        csv_writer=csv_writer,
        pending_merges=pending_merges,
        feature_out_dir=FEATURE_OUT_DIR,
    )

    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            print(f"Force terminating stuck process {p.pid}")
            p.terminate()

    print(f"\n{'=' * 70}")
    print(f"  Done! (Dataset: {DATASET})")
    print(f"  {task_label}: {total_success} processed | "
          f"{item_label}: {total_items_saved} | Errors: {total_errors}")
    print(f"  Output per {item_label.lower()[:-1]}:")
    print(f"       feat_last_token : [N, {HIDDEN_DIM}]")
    print(f"       feat_vis_last   : [N, {HIDDEN_DIM}]")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()