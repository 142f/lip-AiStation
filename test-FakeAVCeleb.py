import argparse
import csv
import json
import os
import re
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from models import build_model


try:
    cv2.setNumThreads(0)
except Exception:
    pass


FAKE_TYPE_MAP = {
    "realvideo-realaudio": "Real",
    "realvideo-fakeaudio": "Audio Fake",
    "fakevideo-realaudio": "Video Fake",
    "fakevideo-fakeaudio": "Audio + Video Fake",
}


def normalize_path_text(path_text):
    path_text = str(path_text or "").strip().strip('"').strip("'")
    path_text = path_text.replace("\\", "/")
    while "//" in path_text:
        path_text = path_text.replace("//", "/")
    return path_text


def is_windows_abs_path(path_text):
    path_text = normalize_path_text(path_text)
    return re.match(r"^[A-Za-z]:/", path_text) is not None


def safe_abspath(path_text):
    if not path_text:
        return ""
    return os.path.abspath(os.path.normpath(path_text))


@lru_cache(maxsize=4096)
def find_dir_by_basename(search_root, dir_name):
    """
    当 CSV 里的绝对路径已经失效时，尝试在 dataset_root 下按目录名搜索。
    例如：
      CSV: E:/data/FakeAVCeleb-test/fake/id_00123
      服务器: /3240609021/FakeAVCeleb-test/fake/id_00123
    """
    search_root = safe_abspath(search_root)
    dir_name = str(dir_name or "").strip()

    if not search_root or not os.path.isdir(search_root) or not dir_name:
        return ""

    target = dir_name.lower()
    for root, dirs, _ in os.walk(search_root):
        for d in dirs:
            if d.lower() == target:
                return os.path.join(root, d)
    return ""


def parse_label(row):
    label = str(row.get("label", "")).strip().lower()
    if label in {"real", "0", "false"}:
        return 0
    if label in {"fake", "1", "true"}:
        return 1

    original_type = str(row.get("original_type", "")).strip().lower()
    if original_type == "realvideo-realaudio":
        return 0
    if original_type:
        return 1
    return None


def parse_fake_type(row):
    original_type = str(row.get("original_type", "")).strip()
    key = original_type.lower()
    if key in FAKE_TYPE_MAP:
        return FAKE_TYPE_MAP[key]

    audio_label = str(row.get("audio_label", "")).strip().lower()
    video_label = str(row.get("video_label", "")).strip().lower()

    if audio_label == "real" and video_label == "real":
        return "Real"
    if audio_label == "fake" and video_label == "real":
        return "Audio Fake"
    if audio_label == "real" and video_label == "fake":
        return "Video Fake"
    if audio_label == "fake" and video_label == "fake":
        return "Audio + Video Fake"
    return "Unknown"


def resolve_output_dir(raw_dir, manifest_dir, dataset_root):
    """
    更稳的路径解析：
    1. 直接存在的路径优先
    2. 兼容 Windows 绝对路径 / Linux 绝对路径
    3. 如果 raw_dir 里包含 FakeAVCeleb-test 之后的相对部分，自动拼到 dataset_root
    4. 如果路径里能找到 real / fake / 0_real / 1_fake 作为锚点，自动拼到 dataset_root
    5. 最后用目录名 basename 在 dataset_root 下搜索
    """
    raw_dir = normalize_path_text(raw_dir)
    manifest_dir = safe_abspath(manifest_dir)
    dataset_root = safe_abspath(dataset_root) if dataset_root else ""

    if not raw_dir:
        return ""

    candidates = []
    seen = set()

    def add_candidate(path_text):
        if not path_text:
            return
        path_text = safe_abspath(path_text)
        if path_text and path_text not in seen:
            seen.add(path_text)
            candidates.append(path_text)

    # 0) 直接路径先试
    if os.path.isdir(raw_dir):
        return safe_abspath(raw_dir)

    # 1) Linux 绝对路径
    if os.path.isabs(raw_dir):
        add_candidate(raw_dir)

    # 2) Windows 绝对路径：例如 E:/data/FakeAVCeleb-test/fake/xxx
    raw_lower = raw_dir.lower()

    if dataset_root:
        dataset_base = os.path.basename(dataset_root.rstrip("/\\")).lower()

        # 2.1) 如果路径里包含数据集根目录名，比如 FakeAVCeleb-test
        if dataset_base and dataset_base in raw_lower:
            idx = raw_lower.find(dataset_base)
            suffix = raw_dir[idx + len(dataset_base):].lstrip("/")
            add_candidate(os.path.join(dataset_root, suffix))

        # 2.2) 常见锚点 real / fake / 0_real / 1_fake
        parts = [p for p in raw_dir.split("/") if p and p != "."]
        lowered_parts = [p.lower() for p in parts]
        for anchor in ("real", "fake", "0_real", "1_fake"):
            if anchor in lowered_parts:
                start_idx = lowered_parts.index(anchor)
                suffix = os.path.join(*parts[start_idx:]) if start_idx < len(parts) else ""
                add_candidate(os.path.join(dataset_root, suffix))
                break

        # 2.3) 普通相对路径
        add_candidate(os.path.join(dataset_root, raw_dir.lstrip("/")))

    # 3) 相对于 manifest 所在目录
    add_candidate(os.path.join(manifest_dir, raw_dir.lstrip("/")))

    # 4) 原样当相对路径
    add_candidate(raw_dir)

    # 5) 先返回第一个真实存在的目录
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # 6) 最后兜底：按目录名 basename 在 dataset_root 中搜索
    tail_name = os.path.basename(raw_dir.rstrip("/\\"))
    if dataset_root and tail_name:
        found = find_dir_by_basename(dataset_root, tail_name)
        if found and os.path.isdir(found):
            return safe_abspath(found)

    # 7) 都找不到时，返回最可能的第一个候选，方便调试
    return candidates[0] if candidates else ""


def load_manifest(manifest_csv, status_allow, dataset_root="", debug_resolve_limit=5):
    manifest_csv = safe_abspath(manifest_csv)
    manifest_dir = os.path.dirname(manifest_csv)

    records = []
    debug_count = 0

    with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {manifest_csv}")

        has_status = "status" in set(reader.fieldnames)

        for row in reader:
            status = str(row.get("status", "ok")).strip()
            if has_status and status_allow and status not in status_allow:
                continue

            label = parse_label(row)
            if label is None:
                continue

            raw_output_frame_dir = normalize_path_text(row.get("output_frame_dir", ""))
            output_frame_dir = resolve_output_dir(
                raw_dir=raw_output_frame_dir,
                manifest_dir=manifest_dir,
                dataset_root=dataset_root,
            )

            if debug_count < debug_resolve_limit:
                print(f"[DEBUG] raw output_frame_dir      : {raw_output_frame_dir}")
                print(f"[DEBUG] resolved output_frame_dir : {output_frame_dir}")
                print(f"[DEBUG] exists?                   : {os.path.isdir(output_frame_dir)}")
                print("-" * 80)
                debug_count += 1

            records.append(
                {
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "label": label,
                    "label_text": "fake" if label == 1 else "real",
                    "original_type": str(row.get("original_type", "")).strip(),
                    "fake_type": parse_fake_type(row),
                    "race": str(row.get("race", "")).strip() or "Unknown",
                    "gender": str(row.get("gender", "")).strip() or "Unknown",
                    "status": status or "ok",
                    "raw_output_frame_dir": raw_output_frame_dir,
                    "output_frame_dir": output_frame_dir,
                }
            )
    return records


def list_images(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for filename in files:
            suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if suffix in {"png", "jpg", "jpeg"}:
                image_paths.append(os.path.join(root, filename))
    image_paths.sort()
    return image_paths


class ManifestImageDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_rows):
        self.samples = []
        self.missing_dirs = []
        self.empty_dirs = []

        for row in manifest_rows:
            frame_dir = row["output_frame_dir"]

            if not frame_dir or (not os.path.isdir(frame_dir)):
                self.missing_dirs.append(
                    {
                        "sample_id": row["sample_id"],
                        "label_text": row["label_text"],
                        "fake_type": row["fake_type"],
                        "race": row["race"],
                        "gender": row["gender"],
                        "status": row["status"],
                        "raw_output_frame_dir": row.get("raw_output_frame_dir", ""),
                        "resolved_output_frame_dir": frame_dir,
                    }
                )
                continue

            images = list_images(frame_dir)
            if not images:
                self.empty_dirs.append(
                    {
                        "sample_id": row["sample_id"],
                        "label_text": row["label_text"],
                        "fake_type": row["fake_type"],
                        "race": row["race"],
                        "gender": row["gender"],
                        "status": row["status"],
                        "raw_output_frame_dir": row.get("raw_output_frame_dir", ""),
                        "resolved_output_frame_dir": frame_dir,
                    }
                )
                continue

            for image_path in images:
                self.samples.append(
                    {
                        "image_path": image_path,
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "label_text": row["label_text"],
                        "fake_type": row["fake_type"],
                        "original_type": row["original_type"],
                        "race": row["race"],
                        "gender": row["gender"],
                        "status": row["status"],
                        "raw_output_frame_dir": row.get("raw_output_frame_dir", ""),
                        "output_frame_dir": frame_dir,
                    }
                )

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.crop_resize = transforms.Resize((224, 224))
        self.crop_idx = [(28, 196), (61, 163)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        label = sample["label"]

        image_cv = cv2.imread(image_path)
        if image_cv is None:
            zero_img = torch.zeros((3, 1120, 1120), dtype=torch.uint8)
            zero_crop = torch.zeros((3, 224, 224), dtype=torch.float32)
            crops = [[zero_crop.clone() for _ in range(5)] for _ in range(3)]
            return zero_img, crops, label, sample

        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        crops_level0 = []
        for i in range(5):
            patch = image_cv[500:, i * 500: i * 500 + 500, :]
            if patch.size == 0:
                patch = np.zeros((500, 500, 3), dtype=np.uint8)
            patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            patch_tensor = self.normalize(self.to_tensor(patch))
            crops_level0.append(patch_tensor)

        crops = [crops_level0, [], []]
        for patch in crops_level0:
            c1 = patch[:, self.crop_idx[0][0]: self.crop_idx[0][1], self.crop_idx[0][0]: self.crop_idx[0][1]]
            c2 = patch[:, self.crop_idx[1][0]: self.crop_idx[1][1], self.crop_idx[1][0]: self.crop_idx[1][1]]
            crops[1].append(self.crop_resize(c1))
            crops[2].append(self.crop_resize(c2))

        img_global = cv2.resize(image_cv, (1120, 1120), interpolation=cv2.INTER_LINEAR)
        img_tensor = self.to_tensor(img_global)

        return img_tensor, crops, label, sample


def custom_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    num_scales = len(batch[0][1])
    num_regions = len(batch[0][1][0])

    crops_batched = []
    for scale_idx in range(num_scales):
        scale_crops = []
        for region_idx in range(num_regions):
            region_tensors = [item[1][scale_idx][region_idx] for item in batch]
            scale_crops.append(torch.stack(region_tensors))
        crops_batched.append(scale_crops)

    metas = [item[3] for item in batch]
    return imgs, crops_batched, labels, metas


def remove_state_dict_prefixes(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[7:]
            elif name.startswith("_orig_mod."):
                name = name[10:]
        new_state_dict[name] = value
    return new_state_dict


def build_threshold_grid(y_prob):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = y_prob[np.isfinite(y_prob)]
    if y_prob.size == 0:
        return np.asarray([0.5], dtype=np.float64)

    unique_probs = np.unique(y_prob)
    if unique_probs.size > 2000:
        quantiles = np.linspace(0.0, 1.0, 2000)
        unique_probs = np.unique(np.quantile(unique_probs, quantiles))

    base = np.asarray([0.4, 0.45, 0.5], dtype=np.float64)
    grid = np.unique(np.concatenate([unique_probs, base]))
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]
    return grid if grid.size > 0 else np.asarray([0.5], dtype=np.float64)


def select_threshold(y_true, y_prob, strategy="youden_j", fixed_threshold=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if strategy == "fixed":
        if fixed_threshold is None:
            return 0.5, "fixed_default_0.5"
        return float(fixed_threshold), "fixed_user"

    if len(np.unique(y_true)) < 2:
        return 0.5, "fixed_0.5_single_class"

    if strategy == "youden_j":
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr_curve - fpr_curve
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        if not np.isfinite(best_threshold):
            return 0.5, "fixed_0.5_non_finite"
        return best_threshold, "youden_j"

    grid = build_threshold_grid(y_prob)
    best_threshold = 0.5
    best_score = -1.0

    for thr in grid:
        metrics = compute_metrics(y_true, y_prob, float(thr))
        if strategy == "f1":
            score = metrics["f1"] if metrics["f1"] is not None else -1.0
        elif strategy == "balanced_acc":
            score = metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        if score > best_score:
            best_score = score
            best_threshold = float(thr)

    return best_threshold, strategy


def parse_threshold_list(text):
    if text is None:
        return []
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            values.append(value)
    return sorted(set(values))


def safe_float(value):
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    total = int(y_true.shape[0])
    if total == 0:
        return {
            "total": 0,
            "num_real": 0,
            "num_fake": 0,
            "threshold": safe_float(threshold),
            "accuracy": None,
            "auc": None,
            "ap": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "best_f1_curve": None,
            "real_recall": None,
            "fake_recall": None,
            "balanced_accuracy": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }

    y_pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    real_recall = tn / max(tn + fp, 1)
    fake_recall = tp / max(tp + fn, 1)
    balanced_accuracy = 0.5 * (real_recall + fake_recall)

    auc = None
    ap = None
    best_f1_curve = None
    if len(np.unique(y_true)) >= 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = None
        try:
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            ap = None
        try:
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_f1_curve = float(np.max(f1_scores))
        except ValueError:
            best_f1_curve = None

    return {
        "total": total,
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
        "threshold": safe_float(threshold),
        "accuracy": safe_float(accuracy),
        "auc": safe_float(auc),
        "ap": safe_float(ap),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1": safe_float(f1),
        "best_f1_curve": safe_float(best_f1_curve),
        "real_recall": safe_float(real_recall),
        "fake_recall": safe_float(fake_recall),
        "balanced_accuracy": safe_float(balanced_accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_metrics(records, key_fields, threshold):
    buckets = defaultdict(lambda: {"y_true": [], "y_prob": []})
    for row in records:
        key = tuple((str(row.get(field, "")).strip() or "Unknown") for field in key_fields)
        buckets[key]["y_true"].append(row["y_true"])
        buckets[key]["y_prob"].append(row["y_prob_fake"])

    rows = []
    for key, values in buckets.items():
        metrics = compute_metrics(values["y_true"], values["y_prob"], threshold)
        result = {field: key[i] for i, field in enumerate(key_fields)}
        result.update(metrics)
        rows.append(result)

    rows.sort(key=lambda x: tuple(str(x[field]) for field in key_fields))
    return rows


def build_balanced_subset(records, seed=42):
    real_rows = [row for row in records if int(row.get("y_true", -1)) == 0]
    fake_rows = [row for row in records if int(row.get("y_true", -1)) == 1]

    if not real_rows or not fake_rows:
        return []

    n_each = min(len(real_rows), len(fake_rows))
    rng = np.random.default_rng(seed)

    real_idx = rng.choice(len(real_rows), size=n_each, replace=False)
    fake_idx = rng.choice(len(fake_rows), size=n_each, replace=False)

    selected = [real_rows[i] for i in real_idx] + [fake_rows[i] for i in fake_idx]
    selected.sort(key=lambda x: str(x.get("image_path", "")))
    return selected


def build_threshold_scan_rows(y_true, y_prob, thresholds):
    rows = []
    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold)
        row = {"threshold": float(threshold)}
        row.update(metrics)
        rows.append(row)
    rows.sort(key=lambda x: x["threshold"])
    return rows


def to_serializable(value):
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(logit_fake, temperature):
    t = float(max(temperature, 1e-6))
    return sigmoid_np(np.asarray(logit_fake, dtype=np.float64) / t)


def fit_temperature_scaling(logit_fake, y_true, max_iter=200):
    logits = torch.tensor(np.asarray(logit_fake, dtype=np.float32))
    labels = torch.tensor(np.asarray(y_true, dtype=np.float32))

    if logits.numel() == 0:
        return 1.0

    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-6, max=100.0)
        loss = criterion(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t_final = float(torch.exp(log_t).detach().cpu().item())
    if not np.isfinite(t_final) or t_final <= 0:
        return 1.0
    return t_final


def run_inference(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []
    y_logit = []
    records = []

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for img, raw_crops, label, metas in tqdm(loader, desc="Testing", leave=True):
            img = img.to(device, non_blocking=True)

            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            img_tens = img.sub(mean).div(std)

            crops_tens = []
            for scale_list in raw_crops:
                processed_scale = []
                for crop_batch in scale_list:
                    processed_scale.append(crop_batch.to(device, non_blocking=True))
                crops_tens.append(processed_scale)

            features = model.get_features(img_tens)
            logits = model(crops_tens, features)[0]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            logit_fake = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            labels = label.detach().cpu().numpy()

            for i in range(len(metas)):
                rec = dict(metas[i])
                rec["y_true"] = int(labels[i])
                rec["y_prob_fake_raw"] = float(probs[i])
                rec["y_logit_fake_raw"] = float(logit_fake[i])
                rec["y_prob_fake"] = float(probs[i])
                y_true.append(rec["y_true"])
                y_prob.append(rec["y_prob_fake"])
                y_logit.append(rec["y_logit_fake_raw"])
                records.append(rec)

    return records, np.asarray(y_true), np.asarray(y_prob), np.asarray(y_logit)


def build_dataset_loader(manifest_csv, status_allow, dataset_root, debug_resolve_limit, batch_size, workers, tag):
    manifest_rows = load_manifest(
        manifest_csv,
        status_allow,
        dataset_root=dataset_root,
        debug_resolve_limit=debug_resolve_limit,
    )
    print(f"[Info] {tag} manifest rows after status filter: {len(manifest_rows)}")
    if len(manifest_rows) == 0:
        raise RuntimeError(f"No manifest rows available after filtering for {tag}.")

    dataset = ManifestImageDataset(manifest_rows)
    print(f"[Info] {tag} expanded image samples: {len(dataset)}")
    print(f"[Info] {tag} missing frame directories: {len(dataset.missing_dirs)}")
    print(f"[Info] {tag} empty frame directories: {len(dataset.empty_dirs)}")

    if dataset.missing_dirs:
        print(f"[Warn] First 5 missing dirs in {tag}:")
        for item in dataset.missing_dirs[:5]:
            print(
                f"  sample_id={item['sample_id']} | "
                f"raw={item['raw_output_frame_dir']} | "
                f"resolved={item['resolved_output_frame_dir']}"
            )

    if len(dataset) == 0:
        raise RuntimeError(f"No image samples found for {tag}. Check CSV output_frame_dir and status filter.")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    return manifest_rows, dataset, loader


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--manifest_csv", type=str, required=True, help="CSV with sample-level metadata.")
    parser.add_argument("--val_manifest_csv", type=str, default="", help="Optional validation CSV for threshold selection/calibration.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Folder to save analysis files.")
    parser.add_argument("--dataset_root", type=str, default="", help="Optional root for resolving relative paths in CSV.")
    parser.add_argument("--val_dataset_root", type=str, default="", help="Optional root for val_manifest_csv.")
    parser.add_argument(
        "--status_allow",
        type=str,
        default="ok,skipped_existing",
        help="Comma-separated allowed status values. Ignored if CSV has no status column.",
    )
    parser.add_argument(
        "--val_status_allow",
        type=str,
        default="",
        help="Comma-separated status filter for validation CSV. Default uses --status_allow.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default="youden_j",
        choices=["youden_j", "f1", "balanced_acc", "fixed"],
        help="How to choose final threshold on threshold-source scores.",
    )
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="val_if_available",
        choices=["val_if_available", "val", "test"],
        help="Which split to use when selecting threshold.",
    )
    parser.add_argument("--fixed_threshold", type=float, default=None, help="Used when threshold_strategy=fixed.")
    parser.add_argument(
        "--threshold_scan_list",
        type=str,
        default="0.4,0.45,0.5",
        help="Comma-separated thresholds to compare (always includes selected threshold).",
    )
    parser.add_argument(
        "--balanced_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate a random 1:1 balanced subset as supplementary results.",
    )
    parser.add_argument("--balanced_seed", type=int, default=42, help="Random seed for 1:1 subset sampling.")
    parser.add_argument(
        "--temperature_scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit temperature on threshold-source split and apply calibration to probabilities.",
    )
    parser.add_argument("--temperature_max_iter", type=int, default=200, help="Max LBFGS iterations for temperature scaling.")
    parser.add_argument("--debug_resolve_limit", type=int, default=0, help="Print first N path resolve results.")

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    if not os.path.isfile(opt.manifest_csv):
        raise FileNotFoundError(f"manifest_csv not found: {opt.manifest_csv}")
    if not os.path.isfile(opt.ckpt):
        raise FileNotFoundError(f"ckpt not found: {opt.ckpt}")

    status_allow_test = {x.strip() for x in str(opt.status_allow).split(",") if x.strip()}
    status_allow_val = (
        {x.strip() for x in str(opt.val_status_allow).split(",") if x.strip()}
        if str(opt.val_status_allow).strip()
        else status_allow_test
    )

    manifest_rows, dataset, loader = build_dataset_loader(
        manifest_csv=opt.manifest_csv,
        status_allow=status_allow_test,
        dataset_root=opt.dataset_root,
        debug_resolve_limit=opt.debug_resolve_limit,
        batch_size=opt.batch_size,
        workers=opt.workers,
        tag="test",
    )

    val_manifest_rows = []
    val_dataset = None
    val_loader = None
    if str(opt.val_manifest_csv).strip():
        if not os.path.isfile(opt.val_manifest_csv):
            raise FileNotFoundError(f"val_manifest_csv not found: {opt.val_manifest_csv}")
        val_root = opt.val_dataset_root if str(opt.val_dataset_root).strip() else opt.dataset_root
        val_manifest_rows, val_dataset, val_loader = build_dataset_loader(
            manifest_csv=opt.val_manifest_csv,
            status_allow=status_allow_val,
            dataset_root=val_root,
            debug_resolve_limit=0,
            batch_size=opt.batch_size,
            workers=opt.workers,
            tag="val",
        )

    print(f"[Info] Build model: {opt.arch}")
    model = build_model(opt.arch)
    model.to(device)

    print(f"[Info] Loading checkpoint: {opt.ckpt}")
    checkpoint = torch.load(opt.ckpt, map_location="cpu")
    if "model_ema" in checkpoint:
        state_dict = checkpoint["model_ema"]
        print("[Info] Use model_ema weights.")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("[Info] Use model weights.")
    else:
        state_dict = checkpoint
        print("[Info] Use raw state_dict.")

    state_dict = remove_state_dict_prefixes(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] Load result: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[Warn] First 5 missing keys: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"[Warn] First 5 unexpected keys: {msg.unexpected_keys[:5]}")

    records, y_true, y_prob, y_logit = run_inference(model, loader, device)
    if len(records) == 0:
        raise RuntimeError("No inference results generated.")

    val_records = []
    val_y_true = np.asarray([], dtype=np.int64)
    val_y_prob = np.asarray([], dtype=np.float64)
    val_y_logit = np.asarray([], dtype=np.float64)
    if val_loader is not None:
        val_records, val_y_true, val_y_prob, val_y_logit = run_inference(model, val_loader, device)

    if opt.threshold_source == "val":
        if val_loader is None:
            raise RuntimeError("threshold_source=val requires --val_manifest_csv.")
        threshold_source_name = "val"
    elif opt.threshold_source == "test":
        threshold_source_name = "test"
    else:
        threshold_source_name = "val" if val_loader is not None else "test"

    temperature = 1.0
    temperature_source = "none"
    if opt.temperature_scaling:
        if threshold_source_name == "val":
            cal_logits, cal_true = val_y_logit, val_y_true
            temperature_source = "val"
        else:
            cal_logits, cal_true = y_logit, y_true
            temperature_source = "test"

        if len(np.unique(cal_true)) < 2:
            print(f"[Warn] Temperature scaling skipped: {temperature_source} has single class.")
        else:
            temperature = fit_temperature_scaling(cal_logits, cal_true, max_iter=opt.temperature_max_iter)
            print(f"[Info] Temperature scaling fitted on {temperature_source}: T={temperature:.6f}")

    if opt.temperature_scaling:
        y_prob = apply_temperature(y_logit, temperature)
        for i, row in enumerate(records):
            row["y_prob_fake"] = float(y_prob[i])
        if len(val_records) > 0:
            val_y_prob = apply_temperature(val_y_logit, temperature)
            for i, row in enumerate(val_records):
                row["y_prob_fake"] = float(val_y_prob[i])

    if threshold_source_name == "val":
        threshold_y_true = val_y_true
        threshold_y_prob = val_y_prob
    else:
        threshold_y_true = y_true
        threshold_y_prob = y_prob

    threshold, threshold_rule = select_threshold(
        y_true=threshold_y_true,
        y_prob=threshold_y_prob,
        strategy=opt.threshold_strategy,
        fixed_threshold=opt.fixed_threshold,
    )

    for row in records:
        row["y_pred"] = int(row["y_prob_fake"] >= threshold)
    for row in val_records:
        row["y_pred"] = int(row["y_prob_fake"] >= threshold)

    overall = compute_metrics(y_true, y_prob, threshold)
    by_fake_type = aggregate_metrics(records, ["fake_type"], threshold)
    by_race = aggregate_metrics(records, ["race"], threshold)
    by_fake_type_race = aggregate_metrics(records, ["fake_type", "race"], threshold)
    val_overall = compute_metrics(val_y_true, val_y_prob, threshold) if len(val_records) > 0 else None

    output_dir = safe_abspath(opt.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prediction_fields = [
        "image_path",
        "sample_id",
        "label",
        "label_text",
        "original_type",
        "fake_type",
        "race",
        "gender",
        "status",
        "raw_output_frame_dir",
        "output_frame_dir",
        "y_true",
        "y_prob_fake_raw",
        "y_logit_fake_raw",
        "y_prob_fake",
        "y_pred",
    ]
    metrics_fields = [
        "total",
        "num_real",
        "num_fake",
        "threshold",
        "accuracy",
        "auc",
        "ap",
        "precision",
        "recall",
        "f1",
        "best_f1_curve",
        "real_recall",
        "fake_recall",
        "balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    scan_thresholds = parse_threshold_list(opt.threshold_scan_list)
    scan_thresholds = sorted(set(scan_thresholds + [float(threshold)]))
    threshold_scan_full = build_threshold_scan_rows(y_true, y_prob, scan_thresholds)
    threshold_scan_val = []
    if len(val_records) > 0:
        threshold_scan_val = build_threshold_scan_rows(val_y_true, val_y_prob, scan_thresholds)

    balanced_records = []
    balanced_overall = None
    balanced_by_fake_type = []
    balanced_by_race = []
    balanced_by_fake_type_race = []
    threshold_scan_balanced = []
    if opt.balanced_eval:
        balanced_records = build_balanced_subset(records, seed=opt.balanced_seed)
        if balanced_records:
            y_true_bal = np.asarray([int(x["y_true"]) for x in balanced_records], dtype=np.int64)
            y_prob_bal = np.asarray([float(x["y_prob_fake"]) for x in balanced_records], dtype=np.float64)
            for row in balanced_records:
                row["y_pred"] = int(row["y_prob_fake"] >= threshold)

            balanced_overall = compute_metrics(y_true_bal, y_prob_bal, threshold)
            balanced_by_fake_type = aggregate_metrics(balanced_records, ["fake_type"], threshold)
            balanced_by_race = aggregate_metrics(balanced_records, ["race"], threshold)
            balanced_by_fake_type_race = aggregate_metrics(balanced_records, ["fake_type", "race"], threshold)
            threshold_scan_balanced = build_threshold_scan_rows(y_true_bal, y_prob_bal, scan_thresholds)

    write_csv(os.path.join(output_dir, "predictions_with_meta.csv"), records, prediction_fields)
    write_csv(os.path.join(output_dir, "stats_by_fake_type.csv"), by_fake_type, ["fake_type"] + metrics_fields)
    write_csv(os.path.join(output_dir, "stats_by_race.csv"), by_race, ["race"] + metrics_fields)
    write_csv(
        os.path.join(output_dir, "stats_by_fake_type_race.csv"),
        by_fake_type_race,
        ["fake_type", "race"] + metrics_fields,
    )
    write_csv(
        os.path.join(output_dir, "threshold_scan_full.csv"),
        threshold_scan_full,
        ["threshold"] + metrics_fields,
    )
    if threshold_scan_val:
        write_csv(
            os.path.join(output_dir, "threshold_scan_val.csv"),
            threshold_scan_val,
            ["threshold"] + metrics_fields,
        )
    if val_records:
        write_csv(
            os.path.join(output_dir, "predictions_with_meta_val.csv"),
            val_records,
            prediction_fields,
        )

    if balanced_records:
        write_csv(
            os.path.join(output_dir, "predictions_with_meta_balanced_1to1.csv"),
            balanced_records,
            prediction_fields,
        )
        write_csv(
            os.path.join(output_dir, "stats_by_fake_type_balanced_1to1.csv"),
            balanced_by_fake_type,
            ["fake_type"] + metrics_fields,
        )
        write_csv(
            os.path.join(output_dir, "stats_by_race_balanced_1to1.csv"),
            balanced_by_race,
            ["race"] + metrics_fields,
        )
        write_csv(
            os.path.join(output_dir, "stats_by_fake_type_race_balanced_1to1.csv"),
            balanced_by_fake_type_race,
            ["fake_type", "race"] + metrics_fields,
        )
        write_csv(
            os.path.join(output_dir, "threshold_scan_balanced_1to1.csv"),
            threshold_scan_balanced,
            ["threshold"] + metrics_fields,
        )

    # 调试表
    write_csv(
        os.path.join(output_dir, "missing_frame_dirs.csv"),
        dataset.missing_dirs,
        [
            "sample_id",
            "label_text",
            "fake_type",
            "race",
            "gender",
            "status",
            "raw_output_frame_dir",
            "resolved_output_frame_dir",
        ],
    )
    write_csv(
        os.path.join(output_dir, "empty_frame_dirs.csv"),
        dataset.empty_dirs,
        [
            "sample_id",
            "label_text",
            "fake_type",
            "race",
            "gender",
            "status",
            "raw_output_frame_dir",
            "resolved_output_frame_dir",
        ],
    )

    summary = {
        "manifest_csv": safe_abspath(opt.manifest_csv),
        "val_manifest_csv": safe_abspath(opt.val_manifest_csv) if str(opt.val_manifest_csv).strip() else "",
        "ckpt": safe_abspath(opt.ckpt),
        "dataset_root": safe_abspath(opt.dataset_root) if opt.dataset_root else "",
        "val_dataset_root": safe_abspath(opt.val_dataset_root) if str(opt.val_dataset_root).strip() else "",
        "device": str(device),
        "threshold_strategy": opt.threshold_strategy,
        "threshold_source": threshold_source_name,
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "threshold_scan_list": scan_thresholds,
        "temperature_scaling": {
            "enabled": bool(opt.temperature_scaling),
            "source": temperature_source,
            "temperature": float(temperature),
        },
        "num_manifest_rows": len(manifest_rows),
        "num_val_manifest_rows": len(val_manifest_rows),
        "num_missing_dirs": len(dataset.missing_dirs),
        "num_empty_dirs": len(dataset.empty_dirs),
        "num_image_samples": len(records),
        "overall_full_distribution": overall,
        "overall_val_distribution": val_overall,
        "balanced_1to1": {
            "enabled": bool(opt.balanced_eval),
            "seed": int(opt.balanced_seed),
            "available": bool(balanced_overall is not None),
            "num_image_samples": int(len(balanced_records)),
            "overall": balanced_overall,
        },
    }
    with open(os.path.join(output_dir, "overall_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_serializable(summary), f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 70)
    print("Final Test Analysis Report")
    print("[Main] Full distribution (original ratio)")
    print(f"Image samples          : {overall['total']} (real={overall['num_real']}, fake={overall['num_fake']})")
    print(f"Threshold source       : {threshold_source_name}")
    print(f"Threshold ({threshold_rule}) : {threshold:.6f}")
    print(
        f"Temperature scaling    : ON (T={temperature:.6f}, source={temperature_source})"
        if opt.temperature_scaling
        else "Temperature scaling    : OFF"
    )
    print(f"Accuracy               : {overall['accuracy']:.4f}" if overall["accuracy"] is not None else "Accuracy               : N/A")
    print(f"AUC                    : {overall['auc']:.4f}" if overall["auc"] is not None else "AUC                    : N/A")
    print(f"AP                     : {overall['ap']:.4f}" if overall["ap"] is not None else "AP                     : N/A")
    print(f"F1                     : {overall['f1']:.4f}" if overall["f1"] is not None else "F1                     : N/A")
    print(
        f"Recall(real/fake)      : {overall['real_recall']:.4f} / {overall['fake_recall']:.4f}"
        if overall["real_recall"] is not None and overall["fake_recall"] is not None
        else "Recall(real/fake)      : N/A"
    )
    print(
        f"Balanced ACC           : {overall['balanced_accuracy']:.4f}"
        if overall["balanced_accuracy"] is not None
        else "Balanced ACC           : N/A"
    )
    print(f"Confusion Matrix       : [TN: {overall['tn']}  FP: {overall['fp']} | FN: {overall['fn']}  TP: {overall['tp']}]")

    if balanced_overall is not None:
        print("-" * 70)
        print("[Supplement] 1:1 balanced subset")
        print(
            f"Image samples          : {balanced_overall['total']} "
            f"(real={balanced_overall['num_real']}, fake={balanced_overall['num_fake']})"
        )
        print(
            f"Accuracy               : {balanced_overall['accuracy']:.4f}"
            if balanced_overall["accuracy"] is not None
            else "Accuracy               : N/A"
        )
        print(
            f"AUC                    : {balanced_overall['auc']:.4f}"
            if balanced_overall["auc"] is not None
            else "AUC                    : N/A"
        )
        print(
            f"AP                     : {balanced_overall['ap']:.4f}"
            if balanced_overall["ap"] is not None
            else "AP                     : N/A"
        )
        print(
            f"F1                     : {balanced_overall['f1']:.4f}"
            if balanced_overall["f1"] is not None
            else "F1                     : N/A"
        )
        print(
            f"Recall(real/fake)      : {balanced_overall['real_recall']:.4f} / {balanced_overall['fake_recall']:.4f}"
            if balanced_overall["real_recall"] is not None and balanced_overall["fake_recall"] is not None
            else "Recall(real/fake)      : N/A"
        )
        print(
            f"Balanced ACC           : {balanced_overall['balanced_accuracy']:.4f}"
            if balanced_overall["balanced_accuracy"] is not None
            else "Balanced ACC           : N/A"
        )
        print(
            f"Confusion Matrix       : [TN: {balanced_overall['tn']}  FP: {balanced_overall['fp']} | "
            f"FN: {balanced_overall['fn']}  TP: {balanced_overall['tp']}]"
        )
    print("#" * 70)
    print(f"[Info] Saved: {os.path.join(output_dir, 'predictions_with_meta.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_fake_type.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_race.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_fake_type_race.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'threshold_scan_full.csv')}")
    if threshold_scan_val:
        print(f"[Info] Saved: {os.path.join(output_dir, 'threshold_scan_val.csv')}")
    if val_records:
        print(f"[Info] Saved: {os.path.join(output_dir, 'predictions_with_meta_val.csv')}")
    if balanced_records:
        print(f"[Info] Saved: {os.path.join(output_dir, 'predictions_with_meta_balanced_1to1.csv')}")
        print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_fake_type_balanced_1to1.csv')}")
        print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_race_balanced_1to1.csv')}")
        print(f"[Info] Saved: {os.path.join(output_dir, 'stats_by_fake_type_race_balanced_1to1.csv')}")
        print(f"[Info] Saved: {os.path.join(output_dir, 'threshold_scan_balanced_1to1.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'missing_frame_dirs.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'empty_frame_dirs.csv')}")
    print(f"[Info] Saved: {os.path.join(output_dir, 'overall_metrics.json')}")


if __name__ == "__main__":
    main()
