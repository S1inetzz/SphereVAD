"""
VLM hidden-state feature extraction for synthetic paired datasets.
Extracts last-token features from all intermediate layers with async prefetch pipeline.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import warnings
import multiprocessing as mp
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time
import threading
import sys

warnings.filterwarnings('ignore')

# ======================================================================
#                          Configuration
# ======================================================================

MODEL_PATH       = "/path/to/model"
PAIR_INDEX_CSV   = "/path/to/pair_index.csv"
BASE_DATASET_DIR = "/path/to/dataset/"
FEATURE_OUT_DIR  = "/path/to/output/features/"
RESULT_CSV_PATH  = "/path/to/output/train_label.csv"

PROMPT_PART1 = (
    "You are a professional video security analysis assistant. "
    "The following four consecutive video frames capture the temporal "
    "evolution of the same scene.\n\n"
    "[Anomaly Categories]\n"
    "Anomalous events are limited to the following 6 groups:\n"
    "[Violent Conflict], [Crime], [Traffic Accident], "
    "[Personal Emergency], [Environmental Hazard], [Public Misconduct].\n\n"
    "Please carefully observe the following frames with the above criteria "
    "and determine whether any matching anomalous event is present:"
)

PROMPT_PART2 = (
    "\nBased on the above frames, strictly follow these 4 steps "
    "(always start with 'Yes' or 'No'):\n\n"
    "1. Final determination: [Output only 'Yes' or 'No'].\n"
    "2. Anomaly category: [Format: Group - Sub-label. If No, output: None].\n"
    "3. Spatiotemporal description: [Briefly describe interactions, "
    "action continuity, and object state changes across frames].\n"
    "4. Confidence: [Output only: High/Medium/Low. If None, output: None]."
)

GRID_ROWS = 2
GRID_COLS = 2

RESIZE_SUBIMAGES = True
TARGET_WIDTH     = 336
TARGET_HEIGHT    = 336

GPU_PROCESS_CONFIG = {0: 0, 1: 4, 2: 4, 3: 4}

USE_FP16 = False

EXTRACT_LAYERS   = list(range(1, 33))
N_EXTRACT_LAYERS = len(EXTRACT_LAYERS)
HIDDEN_DIM       = 4096

RANDOM_SEED      = 42
REFRESH_INTERVAL = 5
ENABLE_THINKING  = False

KEEP_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values", "image_grid_thw",
    "pixel_values_videos", "video_grid_thw",
    "position_ids",
}

PREFETCH_THREADS = 3
PIN_MEMORY       = True


# ======================================================================
#                           Utilities
# ======================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def split_grid(image_path, rows=GRID_ROWS, cols=GRID_COLS):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    cw, ch = W // cols, H // rows
    crops = []
    for r in range(rows):
        for c in range(cols):
            box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            crops.append(img.crop(box))
    return crops


def resize_subimage(img, tw=TARGET_WIDTH, th=TARGET_HEIGHT):
    if tw <= 0 and th <= 0:
        return img
    W, H = img.size
    if tw <= 0:
        tw = int(W * th / H)
    if th <= 0:
        th = int(H * tw / W)
    return img.resize((tw, th), Image.LANCZOS)


# ======================================================================
#                       Feature Extraction
# ======================================================================

def extract_branch_features(outputs, extract_layers, yes_token_id, no_token_id):
    n_states = len(outputs.hidden_states)
    feat_list = []

    for layer_idx in extract_layers:
        if layer_idx >= n_states:
            device = outputs.hidden_states[-1].device
            dtype  = outputs.hidden_states[-1].dtype
            feat_list.append(torch.zeros(HIDDEN_DIM, device=device, dtype=dtype))
            continue
        h = outputs.hidden_states[layer_idx][0]
        feat_list.append(h[-1, :])

    logits_last = outputs.logits[0, -1, :]
    yes_logit = logits_last[yes_token_id].float()
    no_logit  = logits_last[no_token_id].float()
    probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)

    return {
        'feat_last_token': torch.stack(feat_list, dim=0),
        'zero_shot_prob':  probs[0].item(),
    }


# ======================================================================
#                    Asynchronous Preprocessing
# ======================================================================

def _prepare_single_branch(processor, image_path, use_pin_memory=True):
    sub_images = split_grid(image_path, GRID_ROWS, GRID_COLS)
    if RESIZE_SUBIMAGES:
        sub_images = [resize_subimage(img) for img in sub_images]

    content = [{"type": "text", "text": PROMPT_PART1}]
    for img in sub_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": PROMPT_PART2})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = processor(
        text=[text], images=sub_images,
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


# ======================================================================
#                        Data Preparation
# ======================================================================

def prepare_tasks(pair_index_csv, base_dataset_dir):
    df = pd.read_csv(pair_index_csv)
    tasks = []
    skipped = 0

    for _, row in df.iterrows():
        pair_id = row['pair_id']
        source_label = row['source_label']
        branches = []

        if pd.notna(row.get('normal_path', None)) and str(row['normal_path']).strip():
            p = os.path.join(base_dataset_dir, str(row['normal_path']))
            if os.path.exists(p):
                branches.append({'branch_type': 'normal', 'image_path': p, 'label': 'Normal'})
            else:
                skipped += 1

        if pd.notna(row.get('abnormal_path', None)) and str(row['abnormal_path']).strip():
            p = os.path.join(base_dataset_dir, str(row['abnormal_path']))
            if os.path.exists(p):
                branches.append({'branch_type': 'abnormal', 'image_path': p, 'label': source_label})
            else:
                skipped += 1

        if branches:
            tasks.append({
                'pair_id': pair_id, 'source_label': source_label,
                'branches': branches, 'n_branches': len(branches),
            })

    if skipped > 0:
        print(f"Warning: skipped {skipped} missing files")
    return tasks


# ======================================================================
#                          CSV Writer
# ======================================================================

class CSVWriter:
    def __init__(self, csv_path, columns):
        self.csv_path = csv_path
        self.columns  = columns
        self.lock     = threading.Lock()
        self.records  = []

        if os.path.exists(csv_path):
            try:
                self.records = pd.read_csv(csv_path).to_dict('records')
                print(f"Loaded {len(self.records)} existing records from {csv_path}")
            except Exception:
                self.records = []

        d = os.path.dirname(csv_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def add_record(self, record):
        with self.lock:
            self.records.append(record)
            pd.DataFrame(self.records, columns=self.columns).to_csv(
                self.csv_path, index=False
            )

    def get_existing_pt_names(self):
        with self.lock:
            return set(r.get('pt_name', '') for r in self.records)

    def get_record_count(self):
        with self.lock:
            return len(self.records)


# ======================================================================
#                        Worker Process
# ======================================================================

def worker_process(worker_id, gpu_id, task_queue, result_queue,
                   feature_out_dir, extract_layers, seed, refresh_interval):
    set_seed(seed + worker_id)
    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}")

    try:
        dtype = torch.float16 if USE_FP16 else torch.bfloat16
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=dtype, device_map=device, trust_remote_code=True,
        )
        model.eval()

        mem_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
        print(f"[Worker {worker_id}] Model loaded, Memory: {mem_gb:.2f} GB")

        yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id  = processor.tokenizer.encode("No",  add_special_tokens=False)[0]

        prefetch_pool  = ThreadPoolExecutor(max_workers=PREFETCH_THREADS)
        prefetch_queue = deque()
        success_count  = 0
        error_count    = 0

        def _prefetch_task(task):
            return [
                prefetch_pool.submit(
                    _prepare_single_branch, processor, b['image_path'], PIN_MEMORY,
                )
                for b in task['branches']
            ]

        def _try_prefetch_next():
            try:
                t = task_queue.get_nowait()
                if t is not None:
                    prefetch_queue.append((t, _prefetch_task(t)))
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
                n_branches   = task['n_branches']

                try:
                    result_queue.put(('pair_start', worker_id, pair_id, n_branches))

                    all_branch_dicts = []
                    branch_labels    = []
                    branch_types     = []
                    last_reported    = 0

                    for bi, (branch, future) in enumerate(zip(branches, branch_futures)):
                        inputs_cpu, input_ids_cpu = future.result()

                        inputs_gpu = {
                            k: v.to(device, non_blocking=True)
                            for k, v in inputs_cpu.items()
                        }

                        outputs = model(
                            **inputs_gpu,
                            output_hidden_states=True,
                            return_dict=True,
                        )

                        if outputs.hidden_states is None:
                            raise RuntimeError("hidden_states not returned by model forward")

                        if success_count == 0 and bi == 0:
                            n_states = len(outputs.hidden_states)
                            seq_len  = input_ids_cpu.shape[0]
                            print(f"[Worker {worker_id}] "
                                  f"seq_len={seq_len}, hs_count={n_states}, "
                                  f"extract_layers={extract_layers[:4]}...{extract_layers[-4:]}")

                        feat_dict = extract_branch_features(
                            outputs, extract_layers, yes_token_id, no_token_id,
                        )

                        all_branch_dicts.append({
                            'feat_last_token': feat_dict['feat_last_token'].cpu().float(),
                            'zero_shot_prob':  feat_dict['zero_shot_prob'],
                        })
                        branch_labels.append(branch['label'])
                        branch_types.append(branch['branch_type'])

                        current_done = bi + 1
                        if (current_done - last_reported >= refresh_interval
                                or current_done == n_branches):
                            result_queue.put(('branch_progress', worker_id, pair_id,
                                              current_done, n_branches))
                            last_reported = current_done

                        del outputs, inputs_gpu, inputs_cpu

                    pair_data = {
                        'feat_last_token': torch.stack(
                            [d['feat_last_token'] for d in all_branch_dicts], dim=0
                        ),
                        'zero_shot_prob': torch.tensor(
                            [d['zero_shot_prob'] for d in all_branch_dicts],
                            dtype=torch.float32,
                        ),
                    }

                    pt_filename = f"{pair_id}.pt"
                    torch.save(pair_data, os.path.join(feature_out_dir, pt_filename))

                    label_str  = ','.join(
                        f"{bt}:{bl}" for bt, bl in zip(branch_types, branch_labels)
                    )
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
                    import traceback
                    traceback.print_exc()
                    for f in branch_futures:
                        try:
                            f.cancel()
                        except Exception:
                            pass

                if (success_count + error_count) % 20 == 0:
                    torch.cuda.empty_cache()

        prefetch_pool.shutdown(wait=False)
        result_queue.put(('done', worker_id, success_count, error_count))
        print(f"[Worker {worker_id}] Finished: "
              f"{success_count} success, {error_count} errors")

    except Exception as e:
        print(f"[Worker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(('fatal', worker_id, str(e)))


# ======================================================================
#                       Progress Monitor
# ======================================================================

def progress_monitor(result_queue, total_tasks, total_branches_all,
                     num_workers, csv_writer):
    finished_workers    = 0
    total_success       = 0
    total_errors        = 0
    total_branches_done = 0
    worker_status       = {}
    start_time          = time.time()
    last_print_time     = 0

    def fmt_time(s):
        if s < 60:
            return f"{int(s)}s"
        elif s < 3600:
            return f"{int(s // 60)}m{int(s % 60):02d}s"
        return f"{int(s // 3600)}h{int((s % 3600) // 60):02d}m"

    def print_progress(force=False):
        nonlocal last_print_time
        now = time.time()
        if not force and now - last_print_time < 0.5:
            return
        last_print_time = now

        elapsed    = now - start_time
        pairs_done = csv_writer.get_record_count()
        eta_str    = (fmt_time(elapsed / pairs_done * (total_tasks - pairs_done))
                      if pairs_done > 0 else "???")

        bc = sum(s['current'] for s in worker_status.values())
        bt = sum(s['total']   for s in worker_status.values())
        aw = len(worker_status)

        pct    = pairs_done / total_tasks if total_tasks > 0 else 0
        filled = int(20 * pct)
        bar    = "#" * filled + "." * (20 - filled)

        line = (
            f"\rPairs: {pairs_done}/{total_tasks} [{bar}] "
            f"[{fmt_time(elapsed)}<{eta_str}] | "
            f"Branches: {bc}/{bt} ({aw} workers) | "
            f"{total_branches_done} saved"
        )
        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()

    while finished_workers < num_workers:
        try:
            msg = result_queue.get(timeout=60)
            if msg[0] == 'pair_start':
                _, wid, pid, nb = msg
                worker_status[wid] = {'pair': pid, 'current': 0, 'total': nb}
                print_progress()
            elif msg[0] == 'branch_progress':
                _, wid, pid, cb, tb = msg
                if wid in worker_status:
                    worker_status[wid]['current'] = cb
                print_progress()
            elif msg[0] == 'success':
                _, wid, record = msg
                nb = record['n_branches']
                worker_status.pop(wid, None)
                csv_writer.add_record(record)
                total_branches_done += nb
                print_progress(force=True)
            elif msg[0] == 'error':
                _, wid, pid, errmsg = msg
                worker_status.pop(wid, None)
                sys.stdout.write(f"\n[Error] {pid}: {errmsg}\n")
                sys.stdout.flush()
                print_progress(force=True)
            elif msg[0] == 'done':
                _, wid, s, e = msg
                total_success += s
                total_errors  += e
                finished_workers += 1
                worker_status.pop(wid, None)
                print_progress()
            elif msg[0] == 'fatal':
                finished_workers += 1
        except Exception:
            pass

    print()
    print(f"\nFinal CSV saved to: {csv_writer.csv_path}")
    print(f"Total records: {csv_writer.get_record_count()}")
    return total_success, total_errors, total_branches_done


# ======================================================================
#                              Main
# ======================================================================

def main():
    set_seed(RANDOM_SEED)
    total_workers = sum(GPU_PROCESS_CONFIG.values())

    print("=" * 60)
    print("  VLM Hidden-State Feature Extraction")
    print(f"  Features: feat_last_token (layers 1~{N_EXTRACT_LAYERS})")
    print("=" * 60)
    print(f"  Prompt layout : [Part1] -> [4 x Images] -> [Part2]")
    print(f"    Part1 ({len(PROMPT_PART1)} chars): task definition + category priors")
    print(f"    Part2 ({len(PROMPT_PART2)} chars): output instructions (Yes/No first)")
    print(f"  GPU config    : {GPU_PROCESS_CONFIG} ({total_workers} workers)")
    print(f"  Precision     : {'float16' if USE_FP16 else 'bfloat16'}")
    print(f"  Grid          : {GRID_ROWS}x{GRID_COLS}")
    print(f"  Prefetch      : {PREFETCH_THREADS} threads, pin_memory={PIN_MEMORY}")
    print(f"  Extract layers: hs[{EXTRACT_LAYERS[0]}] ~ hs[{EXTRACT_LAYERS[-1]}]"
          f"  ({N_EXTRACT_LAYERS} layers)")
    print(f"  Per branch    : feat_last_token [{N_EXTRACT_LAYERS}, {HIDDEN_DIM}]"
          f" + zero_shot_prob")
    print(f"  Per pair saved: feat_last_token [N_br, {N_EXTRACT_LAYERS}, {HIDDEN_DIM}]"
          f" + zero_shot_prob [N_br]")
    print("=" * 60)

    os.makedirs(FEATURE_OUT_DIR, exist_ok=True)

    csv_columns = ['pt_name', 'pair_id', 'source_label', 'branch_info',
                   'n_branches', 'feature_shape']
    csv_writer = CSVWriter(RESULT_CSV_PATH, columns=csv_columns)

    print("\n[1/4] Loading data...")
    all_tasks      = prepare_tasks(PAIR_INDEX_CSV, BASE_DATASET_DIR)
    total_branches = sum(t['n_branches'] for t in all_tasks)
    print(f"  Total pairs: {len(all_tasks)}, Total branches: {total_branches}")

    existing_pt  = (set(os.listdir(FEATURE_OUT_DIR))
                    if os.path.exists(FEATURE_OUT_DIR) else set())
    existing_csv = csv_writer.get_existing_pt_names()

    tasks_to_process = [
        t for t in all_tasks
        if f"{t['pair_id']}.pt" not in existing_pt
        or f"{t['pair_id']}.pt" not in existing_csv
    ]
    branches_to_process = sum(t['n_branches'] for t in tasks_to_process)

    print(f"  Already done : {len(all_tasks) - len(tasks_to_process)} pairs")
    print(f"  To process   : {len(tasks_to_process)} pairs"
          f" ({branches_to_process} branches)")

    if not tasks_to_process:
        print("\nAll done!")
        return

    print("\n[2/4] Creating queues...")
    manager      = Manager()
    task_queue   = manager.Queue()
    result_queue = manager.Queue()

    tasks_to_process.sort(key=lambda x: x['n_branches'], reverse=True)
    for task in tasks_to_process:
        task_queue.put(task)
    for _ in range(total_workers):
        task_queue.put(None)

    print("\n[3/4] Starting workers...")
    processes = []
    wid = 0
    for gpu_id, np_ in GPU_PROCESS_CONFIG.items():
        for pi in range(np_):
            p = Process(
                target=worker_process,
                args=(wid, gpu_id, task_queue, result_queue,
                      FEATURE_OUT_DIR, EXTRACT_LAYERS,
                      RANDOM_SEED, REFRESH_INTERVAL),
            )
            p.start()
            processes.append(p)
            print(f"  Worker {wid} -> GPU {gpu_id} ({pi + 1}/{np_})")
            wid += 1
            time.sleep(3)

    print("\n[4/4] Extracting...\n")
    total_success, total_errors, total_bd = progress_monitor(
        result_queue, len(tasks_to_process), branches_to_process,
        total_workers, csv_writer,
    )

    for p in processes:
        p.join()

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  Pairs: {total_success} | Branches: {total_bd} | Errors: {total_errors}")
    print(f"  Per-pair tensors:")
    print(f"    feat_last_token : [N_br, {N_EXTRACT_LAYERS}, {HIDDEN_DIM}]")
    print(f"    zero_shot_prob  : [N_br]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()