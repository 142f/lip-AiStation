import argparse
import csv
import json
import os
import re
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def normalize_path_text(path_text):
    path_text = str(path_text or "").strip().strip('"').strip("'")
    path_text = path_text.replace("\\", "/")
    while "//" in path_text:
        path_text = path_text.replace("//", "/")
    return path_text


def safe_abspath(path_text):
    if not path_text:
        return ""
    return os.path.abspath(os.path.normpath(path_text))


def is_windows_abs_path(path_text):
    path_text = normalize_path_text(path_text)
    return re.match(r"^[A-Za-z]:/", path_text) is not None


def write_text_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def count_images_recursive(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS:
                total += 1
    return total


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


def resolve_fakeavceleb_paths(opt):
    dataset_root = safe_abspath(getattr(opt, "dataset_root", "") or "")
    manifest_csv = str(getattr(opt, "manifest_csv", "") or "").strip()

    if manifest_csv:
        manifest_csv = safe_abspath(manifest_csv)
    elif dataset_root:
        candidate_with_status = os.path.join(dataset_root, "test_manifest_with_status.csv")
        candidate_plain = os.path.join(dataset_root, "test_manifest.csv")
        if os.path.isfile(candidate_with_status):
            manifest_csv = candidate_with_status
        elif os.path.isfile(candidate_plain):
            manifest_csv = candidate_plain

    if not dataset_root and manifest_csv:
        dataset_root = safe_abspath(os.path.dirname(manifest_csv))

    if not dataset_root:
        raise ValueError("dataset_root is required. Provide --dataset_root or a manifest_csv under the dataset root.")
    if not manifest_csv:
        raise FileNotFoundError(
            "manifest_csv not found. Provide --manifest_csv, or make sure "
            "<dataset_root>/test_manifest_with_status.csv exists."
        )

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"manifest_csv not found: {manifest_csv}")

    return dataset_root, manifest_csv


def resolve_class_dirs(dataset_root):
    # Support both naming conventions:
    # 1) real/fake
    # 2) 0_real/1_fake
    for real_name, fake_name in (("real", "fake"), ("0_real", "1_fake")):
        real_dir = os.path.join(dataset_root, real_name)
        fake_dir = os.path.join(dataset_root, fake_name)
        if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
            return real_dir, fake_dir, f"{real_name}/{fake_name}"
    return "", "", ""


@lru_cache(maxsize=4096)
def find_dir_by_basename(search_root, dir_name):
    search_root = safe_abspath(search_root)
    dir_name = str(dir_name or "").strip()

    if not search_root or not os.path.isdir(search_root) or not dir_name:
        return ""

    target = dir_name.lower()
    for root, dirs, _ in os.walk(search_root):
        for name in dirs:
            if name.lower() == target:
                return os.path.join(root, name)
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

    if os.path.isdir(raw_dir):
        return safe_abspath(raw_dir)

    if os.path.isabs(raw_dir) or is_windows_abs_path(raw_dir):
        add_candidate(raw_dir)

    if dataset_root:
        dataset_base = os.path.basename(dataset_root.rstrip("/\\")).lower()
        raw_lower = raw_dir.lower()

        if dataset_base and dataset_base in raw_lower:
            idx = raw_lower.find(dataset_base)
            suffix = raw_dir[idx + len(dataset_base) :].lstrip("/")
            add_candidate(os.path.join(dataset_root, suffix))

        parts = [part for part in raw_dir.split("/") if part and part != "."]
        lowered_parts = [part.lower() for part in parts]
        for anchor in ("real", "fake", "0_real", "1_fake"):
            if anchor in lowered_parts:
                start_idx = lowered_parts.index(anchor)
                suffix = os.path.join(*parts[start_idx:]) if start_idx < len(parts) else ""
                add_candidate(os.path.join(dataset_root, suffix))
                break

        add_candidate(os.path.join(dataset_root, raw_dir.lstrip("/")))

    add_candidate(os.path.join(manifest_dir, raw_dir.lstrip("/")))
    add_candidate(raw_dir)

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    tail_name = os.path.basename(raw_dir.rstrip("/\\"))
    if dataset_root and tail_name:
        found = find_dir_by_basename(dataset_root, tail_name)
        if found and os.path.isdir(found):
            return safe_abspath(found)

    return candidates[0] if candidates else ""


def load_manifest(manifest_csv, status_allow, dataset_root="", debug_resolve_limit=0):
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
            _, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTS:
                image_paths.append(os.path.join(root, filename))
    image_paths.sort()
    return image_paths


class FakeAVCelebDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_rows):
        self.samples = []
        self.missing_dirs = []
        self.empty_dirs = []

        for row in manifest_rows:
            frame_dir = row["output_frame_dir"]

            if not frame_dir or not os.path.isdir(frame_dir):
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

            image_paths = list_images(frame_dir)
            if not image_paths:
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

            for image_path in image_paths:
                self.samples.append(
                    {
                        "image_path": safe_abspath(image_path),
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "label_text": row["label_text"],
                        "original_type": row["original_type"],
                        "fake_type": row["fake_type"],
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
            patch = image_cv[500:, i * 500 : i * 500 + 500, :]
            if patch.size == 0:
                patch = np.zeros((500, 500, 3), dtype=np.uint8)
            patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            patch_tensor = self.normalize(self.to_tensor(patch))
            crops_level0.append(patch_tensor)

        crops = [crops_level0, [], []]
        for patch in crops_level0:
            c1 = patch[:, self.crop_idx[0][0] : self.crop_idx[0][1], self.crop_idx[0][0] : self.crop_idx[0][1]]
            c2 = patch[:, self.crop_idx[1][0] : self.crop_idx[1][1], self.crop_idx[1][0] : self.crop_idx[1][1]]
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


def export_error_reports(output_dir, report_prefix, records, threshold, save_all_predictions=False):
    os.makedirs(output_dir, exist_ok=True)
    prefix = str(report_prefix or "test").strip() or "test"

    real_as_fake = []
    fake_as_real = []
    all_failed = []
    prediction_rows = []

    for row in records:
        true_label = int(row["y_true"])
        pred_label = int(row["y_pred"])
        image_path = row["image_path"]

        if true_label == 0 and pred_label == 1:
            error_type = "real_as_fake"
            real_as_fake.append(image_path)
            all_failed.append(image_path)
        elif true_label == 1 and pred_label == 0:
            error_type = "fake_as_real"
            fake_as_real.append(image_path)
            all_failed.append(image_path)
        else:
            error_type = "correct"

        if save_all_predictions:
            prediction_rows.append(
                {
                    "image_path": image_path,
                    "sample_id": row.get("sample_id", ""),
                    "label_text": row.get("label_text", ""),
                    "fake_type": row.get("fake_type", ""),
                    "race": row.get("race", ""),
                    "gender": row.get("gender", ""),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "prob_fake": float(row["y_prob_fake"]),
                    "error_type": error_type,
                }
            )

    real_as_fake_path = os.path.join(output_dir, f"{prefix}_misclassified_real_as_fake.txt")
    fake_as_real_path = os.path.join(output_dir, f"{prefix}_misclassified_fake_as_real.txt")
    all_failed_path = os.path.join(output_dir, f"{prefix}_misclassified_all.txt")
    summary_path = os.path.join(output_dir, f"{prefix}_misclassified_summary.json")
    pred_tsv_path = os.path.join(output_dir, f"{prefix}_predictions.tsv")

    write_text_lines(real_as_fake_path, real_as_fake)
    write_text_lines(fake_as_real_path, fake_as_real)
    write_text_lines(all_failed_path, all_failed)

    if save_all_predictions:
        with open(pred_tsv_path, "w", encoding="utf-8") as f:
            f.write("image_path\tsample_id\tlabel_text\tfake_type\trace\tgender\ttrue_label\tpred_label\tprob_fake\tthreshold\terror_type\n")
            for row in prediction_rows:
                f.write(
                    f"{row['image_path']}\t{row['sample_id']}\t{row['label_text']}\t{row['fake_type']}\t"
                    f"{row['race']}\t{row['gender']}\t{row['true_label']}\t{row['pred_label']}\t"
                    f"{row['prob_fake']:.8f}\t{float(threshold):.8f}\t{row['error_type']}\n"
                )

    summary = {
        "threshold": float(threshold),
        "num_samples": int(len(records)),
        "num_misclassified_total": int(len(all_failed)),
        "num_real_as_fake": int(len(real_as_fake)),
        "num_fake_as_real": int(len(fake_as_real)),
        "real_as_fake_list": normalize_path_text(real_as_fake_path),
        "fake_as_real_list": normalize_path_text(fake_as_real_path),
        "all_failed_list": normalize_path_text(all_failed_path),
        "predictions_tsv": normalize_path_text(pred_tsv_path) if save_all_predictions else "",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_json"] = normalize_path_text(summary_path)
    return summary


def test(model, loader, gpu_id, errors_output_dir="", errors_prefix="test", save_all_predictions=False):
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_true = []
    y_prob = []
    records = []

    print("\n" + "=" * 20 + " Begin Testing Loop " + "=" * 20)

    with torch.no_grad():
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

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
            labels = label.detach().cpu().numpy()

            y_true.extend(labels.tolist())
            y_prob.extend(probs.tolist())

            for idx, meta in enumerate(metas):
                record = dict(meta)
                record["image_path"] = safe_abspath(record.get("image_path", ""))
                record["y_true"] = int(labels[idx])
                record["y_prob_fake"] = float(probs[idx])
                records.append(record)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    single_class = len(np.unique(y_true)) < 2
    if single_class:
        print("Warning: test set has one class only. AUC/AP and Youden threshold use default values.")
        auc = 0.0
        ap = 0.0
        best_threshold = 0.5
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError as exc:
            print(f"Warning: failed to compute AUC: {exc}")
            auc = 0.0

        ap = average_precision_score(y_true, y_prob)
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr_curve - fpr_curve
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])

    y_pred_binary = np.where(y_prob >= best_threshold, 1, 0)

    acc = accuracy_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    if single_class:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        best_f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    else:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_f1 = float(np.max(f1_scores))

    for idx, record in enumerate(records):
        record["y_pred"] = int(y_pred_binary[idx])

    print("\n" + "#" * 60)
    print("Final Test Report")
    print(f"Samples    : {len(y_true)}")
    print(f"ConfMatrix : [TN: {tn}  FP: {fp} | FN: {fn}  TP: {tp}]")
    print("-" * 30)
    print(f"AUC        : {auc:.4f}")
    print(f"AP         : {ap:.4f}")
    print(f"ACC        : {acc:.4f}")
    print(f"Best F1    : {best_f1:.4f}")
    print(f"Threshold  : {best_threshold:.4f}")
    print("#" * 60 + "\n")

    if str(errors_output_dir).strip():
        summary = export_error_reports(
            output_dir=errors_output_dir,
            report_prefix=errors_prefix,
            records=records,
            threshold=best_threshold,
            save_all_predictions=save_all_predictions,
        )
        print(f"[Info] Misclassified reports saved to: {normalize_path_text(errors_output_dir)}")
        print(
            f"[Info] Misclassified: total={summary['num_misclassified_total']} | "
            f"real->fake={summary['num_real_as_fake']} | fake->real={summary['num_fake_as_real']}"
        )
        print(f"[Info] Summary: {summary['summary_json']}")

    return auc, ap, acc, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_root = "/3240609021/FakeAVCeleb-test"
    default_errors_output_dir = os.path.join(script_dir, "error_reports")

    parser.add_argument("--dataset_root", type=str, default=default_dataset_root)
    parser.add_argument("--manifest_csv", type=str, default="", help="Defaults to <dataset_root>/test_manifest_with_status.csv.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--status_allow", type=str, default="ok,skipped_existing")
    parser.add_argument("--debug_resolve_limit", type=int, default=0)
    parser.add_argument("--errors_output_dir", type=str, default=default_errors_output_dir)
    parser.add_argument("--errors_prefix", type=str, default="test")
    parser.add_argument("--save_all_predictions", action="store_true")

    opt = parser.parse_args()

    opt.dataset_root, opt.manifest_csv = resolve_fakeavceleb_paths(opt)
    real_dir, fake_dir, class_layout = resolve_class_dirs(opt.dataset_root)

    print(f"[Info] dataset_root : {normalize_path_text(opt.dataset_root)}")
    print(f"[Info] manifest_csv : {normalize_path_text(opt.manifest_csv)}")
    if real_dir and fake_dir:
        print(f"[Info] class layout : {class_layout}")
        print(f"[Info] real_dir     : {normalize_path_text(real_dir)}")
        print(f"[Info] fake_dir     : {normalize_path_text(fake_dir)}")
        print(f"[Info] images(real/fake): {count_images_recursive(real_dir)}/{count_images_recursive(fake_dir)}")
    else:
        print("[Warn] class dirs not found under dataset_root (checked real/fake and 0_real/1_fake).")
        print("[Warn] continue with manifest-driven sample loading.")

    status_allow = {item.strip() for item in str(opt.status_allow).split(",") if item.strip()}
    manifest_rows = load_manifest(
        manifest_csv=opt.manifest_csv,
        status_allow=status_allow,
        dataset_root=opt.dataset_root,
        debug_resolve_limit=int(opt.debug_resolve_limit),
    )
    print(f"[Info] manifest rows after status filter: {len(manifest_rows)}")

    dataset = FakeAVCelebDataset(manifest_rows)
    print(f"[Info] test image samples: {len(dataset)}")
    print(f"[Info] missing frame dirs: {len(dataset.missing_dirs)}")
    print(f"[Info] empty frame dirs  : {len(dataset.empty_dirs)}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check manifest_csv, dataset_root, and output_frame_dir.")

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device: {device}")

    print(f"[Info] build model: {opt.arch}")
    model = build_model(opt.arch)
    model.to(device)

    if not os.path.exists(opt.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {opt.ckpt}")

    print(f"[Info] loading checkpoint: {opt.ckpt}")
    checkpoint = torch.load(opt.ckpt, map_location="cpu")
    if "model_ema" in checkpoint:
        state_dict = checkpoint["model_ema"]
        print("   -> using model_ema weights")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("   -> using model weights")
    else:
        state_dict = checkpoint
        print("   -> using raw state_dict")

    state_dict = remove_state_dict_prefixes(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] load result: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[Warn] first 5 missing keys: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"[Warn] first 5 unexpected keys: {msg.unexpected_keys[:5]}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    test(
        model,
        loader,
        gpu_id=[opt.gpu],
        errors_output_dir=opt.errors_output_dir,
        errors_prefix=opt.errors_prefix,
        save_all_predictions=bool(opt.save_all_predictions),
    )
