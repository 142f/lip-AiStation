import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


def nested_crops_collate(batch):
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

    return imgs, crops_batched, labels


def nested_crops_with_meta_collate(batch):
    imgs, crops_batched, labels = nested_crops_collate(batch)
    metas = [item[3] for item in batch]
    return imgs, crops_batched, labels, metas


def _normalize_and_forward(model, img, raw_crops, device, mean, std):
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
    probs = torch.softmax(logits, dim=1)
    return logits, probs


def run_binary_inference(model, loader, device, desc="Inference", leave=False):
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        for batch in tqdm(loader, desc=desc, leave=leave):
            if len(batch) < 3:
                raise ValueError("Batch must contain at least (img, raw_crops, label).")
            img, raw_crops, label = batch[0], batch[1], batch[2]
            _, probs = _normalize_and_forward(model, img, raw_crops, device, mean, std)
            y_prob.extend(probs[:, 1].flatten().tolist())
            y_true.extend(label.flatten().tolist())

    return np.asarray(y_true), np.asarray(y_prob)


def run_binary_inference_with_logits(model, loader, device, desc="Inference", leave=False):
    model.eval()

    y_true, y_prob, y_logit = [], [], []
    extras = []
    with torch.no_grad():
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        for batch in tqdm(loader, desc=desc, leave=leave):
            if len(batch) < 3:
                raise ValueError("Batch must contain at least (img, raw_crops, label).")
            img, raw_crops, label = batch[0], batch[1], batch[2]
            logits, probs = _normalize_and_forward(model, img, raw_crops, device, mean, std)
            y_prob.extend(probs[:, 1].flatten().tolist())
            y_true.extend(label.flatten().tolist())
            y_logit.extend((logits[:, 1] - logits[:, 0]).flatten().tolist())
            if len(batch) > 3:
                extras.extend(batch[3])

    return np.asarray(y_true), np.asarray(y_prob), np.asarray(y_logit), extras


def evaluate_fixed_threshold(y_true, y_prob, threshold=0.5):
    if len(np.unique(y_true)) < 2:
        return {
            "auc": 0.0,
            "ap": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "tn": int((y_true == 0).sum()),
            "fp": 0,
            "fn": int((y_true == 1).sum()),
            "tp": 0,
            "single_class": True,
            "threshold": float(threshold),
        }

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0

    ap = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc": auc,
        "ap": ap,
        "acc": acc,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "single_class": False,
        "threshold": float(threshold),
    }


def evaluate_youden_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return {
            "ap": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "acc": 0.0,
            "auc": 0.0,
            "f1": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "single_class": True,
            "threshold": 0.5,
            "f1_at_youden": 0.0,
        }

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0

    ap = float(average_precision_score(y_true, y_prob))

    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr_curve - fpr_curve
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    y_pred = np.where(y_prob >= best_threshold, 1, 0)
    acc = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at_youden = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = float(np.max(f1_scores))

    return {
        "ap": ap,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "auc": auc,
        "f1": best_f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "single_class": False,
        "threshold": best_threshold,
        "f1_at_youden": float(f1_at_youden),
    }
