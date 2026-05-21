import argparse
import torch
import numpy as np
import os
import re
import torch.utils.data
import utils
from models import build_model

# ==========================================
# 数据集导入模块
# ==========================================
# 注意：如果您的 AVLip 定义在 data/__init__.py，请改为 from data import AVLip
# 如果定义在 data/datasets.py，请保持如下：
try:
    from data.datasets import AVLip
except ImportError:
    from data import AVLip

from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve
from tqdm import tqdm


# ==========================================
# 自定义 collate_fn 处理嵌套 List 结构
# ==========================================
def custom_collate_fn(batch):
    """
    自定义 collate 函数，正确处理 AVLip 返回的嵌套 List 结构
    batch: List of (img, crops, label)
           其中 crops 是 List[List[Tensor]]，形状为 [3个尺度][5个区域]
    """
    imgs = torch.stack([item[0] for item in batch])  # (B, C, H, W)
    labels = torch.tensor([item[2] for item in batch])  # (B,)
    
    # 处理 crops：将 [B][3][5] 转换为 [3][5][B, C, H, W]
    # 即：每个尺度、每个区域的所有 batch 样本堆叠在一起
    num_scales = len(batch[0][1])  # 3
    num_regions = len(batch[0][1][0])  # 5
    
    crops_batched = []
    for scale_idx in range(num_scales):
        scale_crops = []
        for region_idx in range(num_regions):
            # 收集所有 batch 中同一尺度、同一区域的 tensor
            region_tensors = [item[1][scale_idx][region_idx] for item in batch]
            scale_crops.append(torch.stack(region_tensors))  # (B, C, H, W)
        crops_batched.append(scale_crops)
    
    return imgs, crops_batched, labels


LABEL_DIR_NAMES = {"0_real", "1_fake", "real", "fake"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRAME_FILE_RE = re.compile(r"^(?:group|frame)_\d+$", re.IGNORECASE)
LAV_REAL_PAIR_RE = re.compile(r"^(?P<pair>.+?_pair_.+?)_seg\d+_\d+$", re.IGNORECASE)
LAV_FAKE_RE = re.compile(r"^(?P<video>.+?)_seg\d+_\d+$", re.IGNORECASE)
TRAILING_FRAME_RE = re.compile(r"^(?P<video>.+)_\d+$")
VIDEO_AGG_METHODS = ("mean", "max", "top3_mean", "top5_mean", "median")


def get_sorted_image_list(path):
    """
    test.py 专用图片列表读取：
    - 支持 .png/.jpg/.jpeg 的大小写变体
    - 目录和文件名排序，保证预测顺序与视频聚合顺序稳定
    """
    image_list = []
    for root, dirs, files in os.walk(str(path).replace("\x00", "")):
        dirs.sort()
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_list.append(os.path.join(root, filename))
    return image_list


def count_images_recursive(path):
    return len(get_sorted_image_list(path)) if path else 0


def _clean_path_arg(path):
    return str(path or "").strip().strip('"').strip("'")


def _find_label_dir(root, names):
    for name in names:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(root):
        wanted = {name.lower() for name in names}
        for child in sorted(os.listdir(root)):
            candidate = os.path.join(root, child)
            if child.lower() in wanted and os.path.isdir(candidate):
                return candidate
    return ""


def resolve_test_paths(opt):
    """Resolve old real/fake inputs and preprocess.py output roots.

    Supported layouts:
    - --real_list_path <root>/0_real --fake_list_path <root>/1_fake
    - --data_root <root> where <root>/0_real and <root>/1_fake exist
    - --data_root <root> where <root>/test/0_real and <root>/test/1_fake exist
    - --real_list_path <root> with no --fake_list_path, if root contains both labels
    """
    real_path = _clean_path_arg(getattr(opt, "real_list_path", ""))
    fake_path = _clean_path_arg(getattr(opt, "fake_list_path", ""))
    data_root = _clean_path_arg(getattr(opt, "data_root", ""))

    if real_path and not fake_path and os.path.isdir(real_path):
        maybe_real = _find_label_dir(real_path, ("0_real", "real"))
        maybe_fake = _find_label_dir(real_path, ("1_fake", "fake"))
        if maybe_real and maybe_fake:
            data_root = real_path
            real_path = ""

    if data_root and (not real_path or not fake_path):
        root = os.path.abspath(os.path.normpath(data_root))
        split_root = os.path.join(root, "test") if os.path.isdir(os.path.join(root, "test")) else root
        if not real_path:
            real_path = _find_label_dir(split_root, ("0_real", "real"))
        if not fake_path:
            fake_path = _find_label_dir(split_root, ("1_fake", "fake"))

    if not real_path or not fake_path:
        raise ValueError(
            "Provide --real_list_path and --fake_list_path, or provide --data_root "
            "pointing to a preprocess.py output folder that contains 0_real and 1_fake."
        )

    real_path = os.path.abspath(os.path.normpath(real_path))
    fake_path = os.path.abspath(os.path.normpath(fake_path))

    if not os.path.isdir(real_path):
        raise FileNotFoundError(f"real_list_path not found: {real_path}")
    if not os.path.isdir(fake_path):
        raise FileNotFoundError(f"fake_list_path not found: {fake_path}")

    return real_path, fake_path


def _split_path_parts(path):
    normalized = os.path.normpath(str(path))
    drive, tail = os.path.splitdrive(normalized)
    parts = [part for part in tail.split(os.sep) if part]
    if drive:
        parts.insert(0, drive)
    return normalized, parts


def _label_dir_index(parts):
    for idx in range(len(parts) - 2, -1, -1):
        if parts[idx].lower() in LABEL_DIR_NAMES:
            return idx
    return None


def _make_video_key(parts, label_idx, sample_name):
    if label_idx is None:
        return os.path.normpath(sample_name)
    key_parts = parts[: label_idx + 1]
    if key_parts and key_parts[0].endswith(":"):
        return os.path.normpath(os.path.join(key_parts[0] + os.sep, *key_parts[1:], sample_name))
    return os.path.normpath(os.path.join(*key_parts, sample_name))


def extract_video_key(frame_path):
    """
    从帧路径中提取视频级 key，兼容当前三类测试集：
    1) test-pro: 0_real/2514_0.png -> 0_real/2514
    2) FakeAVCeleb: 0_real/African_men_id00478_RR/group_000.png -> 0_real/African_men_id00478_RR
    3) LAV-DF: 1_fake/000173_seg0_0.png -> 1_fake/000173
       LAV-DF real pair: 0_real/000171_pair_000173_seg0_0.png -> 0_real/000171_pair_000173

    key 中保留类别目录，避免 real/fake 同名视频被错误合并。
    """
    normalized, parts = _split_path_parts(frame_path)
    label_idx = _label_dir_index(parts)
    stem = os.path.splitext(parts[-1])[0]

    # 目录式样本：FakeAVCeleb 和 InsightFace 预处理输出常见为 sample_dir/group_000.png 或 frame_000000.jpg。
    if FRAME_FILE_RE.match(stem) and len(parts) >= 2:
        parent = parts[-2]
        return _make_video_key(parts, label_idx, parent)

    # LAV-DF real 对照样本，保留 real/fake pair，避免同一 real 被不同 fake pair 合并。
    lav_real_match = LAV_REAL_PAIR_RE.match(stem)
    if lav_real_match:
        return _make_video_key(parts, label_idx, lav_real_match.group("pair"))

    # LAV-DF fake 样本：同一个 fake video 的多个 seg 合并成视频级。
    lav_fake_match = LAV_FAKE_RE.match(stem)
    if lav_fake_match:
        return _make_video_key(parts, label_idx, lav_fake_match.group("video"))

    # test-pro 和通用平铺帧：xxx_0.png / xxx_001.jpg -> xxx。
    trailing_match = TRAILING_FRAME_RE.match(stem)
    if trailing_match:
        return _make_video_key(parts, label_idx, trailing_match.group("video"))

    return normalized


def compute_binary_metrics(y_true, y_pred_prob):
    """
    使用 Youden 最佳阈值计算二分类指标。
    """
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)

    if len(np.unique(y_true)) < 2:
        return None

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e:
        print(f"Warning: 计算 AUC 时出错: {e}")
        auc = 0.0

    ap = average_precision_score(y_true, y_pred_prob)

    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred_binary = np.where(y_pred_prob >= best_threshold, 1, 0)
    acc = accuracy_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    precisions, recalls, _ = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)

    return {
        "auc": auc,
        "ap": ap,
        "acc": acc,
        "best_f1": best_f1,
        "fpr": fpr,
        "fnr": fnr,
        "threshold": best_threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_video_groups(y_true, y_pred_prob, sample_paths):
    """
    将帧级预测按视频聚合成分组。
    """
    if sample_paths is None or len(sample_paths) != len(y_true):
        return None, 0

    video_groups = {}
    for frame_path, label, prob in zip(sample_paths, y_true, y_pred_prob):
        video_key = extract_video_key(frame_path)
        if video_key not in video_groups:
            video_groups[video_key] = {"labels": [], "probs": []}
        video_groups[video_key]["labels"].append(int(label))
        video_groups[video_key]["probs"].append(float(prob))

    inconsistent_label_count = 0
    for group in video_groups.values():
        labels = np.asarray(group["labels"], dtype=np.int64)
        if len(np.unique(labels)) > 1:
            inconsistent_label_count += 1

    return video_groups, inconsistent_label_count


def aggregate_video_score(probs, method):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.size == 0:
        return 0.0

    if method == "mean":
        return float(np.mean(probs))
    if method == "max":
        return float(np.max(probs))
    if method == "median":
        return float(np.median(probs))
    if method.startswith("top") and method.endswith("_mean"):
        k_text = method[3:-5]
        k = int(k_text)
        k = max(1, min(k, probs.size))
        topk = np.partition(probs, -k)[-k:]
        return float(np.mean(topk))

    raise ValueError(f"Unknown video aggregation method: {method}")


def build_video_level_arrays(video_groups, agg_method="mean"):
    """
    将视频分组转换为视频级标签和预测分数。
    视频标签使用同一视频内的多数标签，视频分数由 agg_method 控制。
    """
    if video_groups is None:
        return None, None

    video_true = []
    video_pred_prob = []

    for group in video_groups.values():
        labels = np.asarray(group["labels"], dtype=np.int64)
        label_counts = np.bincount(labels, minlength=2)
        video_true.append(int(np.argmax(label_counts)))
        video_pred_prob.append(aggregate_video_score(group["probs"], agg_method))

    return np.asarray(video_true), np.asarray(video_pred_prob)


def print_video_report(video_groups, inconsistent_label_count, main_agg_method="top3_mean"):
    if video_groups is None:
        print("[Warning] 无法生成视频级结果：dataset.total_list 与帧级预测数量不一致。")
        return None

    frame_counts = [len(group["probs"]) for group in video_groups.values()]
    all_metrics = {}
    all_arrays = {}

    for method in VIDEO_AGG_METHODS:
        video_true, video_pred_prob = build_video_level_arrays(video_groups, method)
        all_arrays[method] = (video_true, video_pred_prob)
        if len(np.unique(video_true)) < 2:
            all_metrics[method] = None
        else:
            all_metrics[method] = compute_binary_metrics(video_true, video_pred_prob)

    video_true, video_pred_prob = all_arrays[main_agg_method]
    if len(np.unique(video_true)) < 2:
        print("[Warning] 视频级数据中只包含一类，无法计算视频级交叉指标 (AUC/AP/FPR/FNR)。")
        return None

    metrics = all_metrics[main_agg_method]
    if metrics is None:
        return None

    print("\n" + "#"*60)
    print(f"【视频级测试报告 / Video-level Test Report】")
    print(f"视频数   : {len(video_true)} 个")
    print(f"主聚合   : {main_agg_method}(fake_prob)")
    print(f"每视频帧数: min {min(frame_counts)} | mean {np.mean(frame_counts):.2f} | max {max(frame_counts)}")
    if inconsistent_label_count > 0:
        print(f"标签警告 : {inconsistent_label_count} 个视频内出现了混合标签，已使用多数标签")
    print(f"混淆矩阵 : [TN: {metrics['tn']}  FP: {metrics['fp']} | FN: {metrics['fn']}  TP: {metrics['tp']}]")
    print("-" * 30)
    print(f"AUC       : {metrics['auc']:.4f}")
    print(f"AP        : {metrics['ap']:.4f}")
    print(f"ACC       : {metrics['acc']:.4f}")
    print(f"Best F1   : {metrics['best_f1']:.4f}")
    print("-" * 30)
    print(f"FPR (误报): {metrics['fpr']:.4f}  <- 视频级假阳性率")
    print(f"FNR (漏报): {metrics['fnr']:.4f}  <- 视频级假阴性率")
    print(f"最佳阈值  : {metrics['threshold']:.4f}")
    print("#"*60 + "\n")

    print("【视频级聚合方式对比 / Video Aggregation Comparison】")
    print(f"{'Agg':<10} {'AUC':>8} {'AP':>8} {'ACC':>8} {'BestF1':>8} {'FPR':>8} {'FNR':>8} {'TH':>8}")
    print("-" * 74)
    for method in VIDEO_AGG_METHODS:
        item = all_metrics[method]
        if item is None:
            print(f"{method:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        print(
            f"{method:<10} "
            f"{item['auc']:>8.4f} {item['ap']:>8.4f} {item['acc']:>8.4f} "
            f"{item['best_f1']:>8.4f} {item['fpr']:>8.4f} {item['fnr']:>8.4f} "
            f"{item['threshold']:>8.4f}"
        )
    print("")

    return metrics


# ==========================================
# 核心测试函数
# ==========================================
def test(model, loader, gpu_id):
    """
    在测试集上评估模型性能，包含鲁棒性异常处理与高阶指标计算
    """
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    y_true, y_pred = [], []
    
    print("\n" + "="*20 + " 开始测试循环 (Testing Loop) " + "="*20)
    
    with torch.no_grad():
        # --- [GPU 归一化准备] ---
        # 移到循环外，避免每个 batch 重复创建 tensor
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        
        def process(tensor):
            # [鲁棒性修复] 增加 dtype 检查
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div_(255.0)
            return (tensor - mean) / std

        for i, (img, raw_crops, label) in enumerate(tqdm(loader, desc="Testing", leave=True)):
            # 1. 接收数据
            img = img.to(device, non_blocking=True)
            
            # 2. 全局输入归一化 (与 validate.py 保持同步)
            # 数据集返回的是 [0, 1] 浮点型张量
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            img_tens = img.sub(mean).div(std)
            
            # 3. 处理人脸裁剪区域 (Crops)
            # raw_crops 是通过 custom_collate_fn 转换得到的 List[List[Tensor]]
            # 数据集已经完成了 crop 的归一化，直接转移至 GPU
            crops_tens = []
            for scale_list in raw_crops:
                processed_scale = []
                for crop_batch in scale_list:
                    c = crop_batch.to(device, non_blocking=True)
                    processed_scale.append(c)
                crops_tens.append(processed_scale)
            
            # 4. 模型推理
            # 注意：get_features 返回的 tensor 已经在 model 所在的 device 上
            features = model.get_features(img_tens)
            # model.forward 返回 (logits, weights_max, weights_org)
            logits = model(crops_tens, features)[0]
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            pred_score = probs[:, 1]  # 取 Fake 类的概率作为风险得分
            
            y_pred.extend(pred_score.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # --- 指标计算与边界防护 ---
    if len(np.unique(y_true)) < 2:
        print("Warning: 测试集中只包含一类数据，无法计算交叉指标 (AUC/AP/FPR/FNR)。")
        # 为保持返回值结构一致性，全部返回 0.0
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # AUC
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e:
        print(f"Warning: 计算 AUC 时出错: {e}")
        auc = 0.0

    # AP (Average Precision)
    ap = average_precision_score(y_true, y_pred_prob)
    
    # 寻找尤登指数 (Youden's J statistic) 对应的最佳阈值
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 基于最佳阈值进行二值化预测
    y_pred_binary = np.where(y_pred_prob >= best_threshold, 1, 0)
    
    # ACC & Confusion Matrix
    acc = accuracy_score(y_true, y_pred_binary)
    # 【鲁棒性拦截】显式声明 labels=[0,1]，防止单一预测导致矩阵坍缩为 1x1，引发解包崩溃
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 【新增指标计算与除零防护】
    # 防止分母为 0 导致系统告警崩溃。如果分母为 0，意味着对应的真实样本不存在，此时该比率在业务上定义为 0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # F1 Score
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
    # 增加 1e-10 防止分母为 0
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    
    print("\n" + "#"*60)
    print(f"【帧级测试报告 / Frame-level Test Report】")
    print(f"帧数     : {len(y_true)} 帧")
    print(f"混淆矩阵 : [TN: {tn}  FP: {fp} | FN: {fn}  TP: {tp}]")
    print("-" * 30)
    print(f"AUC       : {auc:.4f}")
    print(f"AP        : {ap:.4f}")
    print(f"ACC       : {acc:.4f}")
    print(f"Best F1   : {best_f1:.4f}")
    print("-" * 30)
    print(f"FPR (误报): {fpr:.4f}  <- 假阳性率: 真实负例被预测为正例的比例")
    print(f"FNR (漏报): {fnr:.4f}  <- 假阴性率: 真实正例被预测为负例的比例")
    print(f"最佳阈值  : {best_threshold:.4f}")
    print("#"*60 + "\n")

    sample_paths = getattr(getattr(loader, "dataset", None), "total_list", None)
    video_groups, inconsistent_label_count = build_video_groups(y_true, y_pred_prob, sample_paths)
    print_video_report(video_groups, inconsistent_label_count)

    # 返回所有核心指标以便外部监控或日志使用
    return auc, ap, acc, best_f1, fpr, fnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 路径参数
    parser.add_argument("--real_list_path", type=str, default="")
    parser.add_argument("--fake_list_path", type=str, default="")
    parser.add_argument(
        "--data_root",
        "--preprocessed_root",
        dest="data_root",
        type=str,
        default="",
        help="preprocess.py output root. Auto uses <root>/0_real and <root>/1_fake, or <root>/test/0_real and <root>/test/1_fake.",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")
    
    # 运行参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--workers", type=int, default=4)
    
    # 数据集兼容参数
    parser.add_argument("--data_label", type=str, default="val") 
    parser.add_argument("--num_classes", type=int, default=2) 
    parser.add_argument("--fix_backbone", action='store_true')
    parser.add_argument("--fix_encoder", action='store_true')
    parser.add_argument("--name", type=str, default="test_experiment")
    parser.add_argument("--no_innov", action="store_true", help="[Ablation] Master switch: Baseline Mode")
    parser.add_argument("--no_modality_bias", action="store_true", help="[Ablation] Disable modality embedding")
    parser.add_argument("--no_attn_bias", action="store_true", help="[Ablation] Disable attention bias")
    parser.add_argument("--no_se_fusion", action="store_true", help="[Ablation] Disable SE fusion")
    parser.add_argument("--no_residual_cls", action="store_true", help="[Ablation] Disable residual CLS")
    parser.add_argument("--no_region_innov", action="store_true", help="[Region] Disable PE and SE")
    parser.add_argument("--no_region_pe", action="store_true", help="[Region] Disable positional encoding")
    parser.add_argument("--no_region_se", action="store_true", help="[Region] Disable SE attention")

    opt = parser.parse_args()

    os.environ["LIPFD_NO_INNOV"] = "1" if opt.no_innov else "0"
    os.environ["LIPFD_NO_MODALITY_BIAS"] = "1" if opt.no_modality_bias else "0"
    os.environ["LIPFD_NO_ATTN_BIAS"] = "1" if opt.no_attn_bias else "0"
    os.environ["LIPFD_NO_SE_FUSION"] = "1" if opt.no_se_fusion else "0"
    os.environ["LIPFD_NO_RESIDUAL_CLS"] = "1" if opt.no_residual_cls else "0"
    os.environ["REGION_NO_PE"] = "1" if (opt.no_region_pe or opt.no_region_innov) else "0"
    os.environ["REGION_NO_SE"] = "1" if (opt.no_region_se or opt.no_region_innov) else "0"

    print(
        "[Info] Ablation flags: "
        f"no_innov={opt.no_innov}, "
        f"no_modality_bias={opt.no_modality_bias}, "
        f"no_attn_bias={opt.no_attn_bias}, "
        f"no_se_fusion={opt.no_se_fusion}, "
        f"no_residual_cls={opt.no_residual_cls}, "
        f"no_region_pe={os.environ['REGION_NO_PE'] == '1'}, "
        f"no_region_se={os.environ['REGION_NO_SE'] == '1'}"
    )

    try:
        opt.real_list_path, opt.fake_list_path = resolve_test_paths(opt)
    except (ValueError, FileNotFoundError) as e:
        print(f"[Error] {e}")
        exit()

    print(f"[Info] real_list_path: {opt.real_list_path}")
    print(f"[Info] fake_list_path: {opt.fake_list_path}")
    print(
        f"[Info] images(real/fake): "
        f"{count_images_recursive(opt.real_list_path)}/{count_images_recursive(opt.fake_list_path)}"
    )

    # 设置设备
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 1. 构建模型
    print(f"[Info] 构建模型: {opt.arch}")
    model = build_model(opt.arch)
    model.to(device)

    # 2. 加载权重 (【核心修复】解决丢失键问题)
    if os.path.exists(opt.ckpt):
        print(f"[Info] 正在加载权重: {opt.ckpt}")
        checkpoint = torch.load(opt.ckpt, map_location="cpu")
        
        state_dict = None
        # 优先加载 EMA
        if "model_ema" in checkpoint:
            state_dict = checkpoint["model_ema"]
            print(f"   -> 发现 EMA 权重，优先加载")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print(f"   -> 加载标准 Model 权重")
        else:
            state_dict = checkpoint
            print(f"   -> 加载 Raw State Dict")

        # 【核心修复】去除 'module.' 和 '_orig_mod.' 前缀 (针对多卡训练或 torch.compile 的情况)
        # 使用循环去除所有可能的前缀组合 (如 module._orig_mod. 或 _orig_mod.module.)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            while name.startswith('module.') or name.startswith('_orig_mod.'):
                if name.startswith('module.'):
                    name = name[7:]
                elif name.startswith('_orig_mod.'):
                    name = name[10:]
            new_state_dict[name] = v
        state_dict = new_state_dict

        # 加载并检查
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Info] 权重加载结果: 丢失键 {len(msg.missing_keys)} | 多余键 {len(msg.unexpected_keys)}")
        
        if len(msg.missing_keys) > 0:
            print(f"   !!! 警告: 依然有丢失键，请检查 arch 是否匹配，或前 5 个丢失键: {msg.missing_keys[:5]}")

    else:
        print(f"[Error] 找不到权重文件: {opt.ckpt}")
        exit()

    # 3. 准备数据
    print(f"[Info] 初始化数据集...")
    # 只在 test.py 内适配测试集图片命名，不修改 utils.py 的全局实现。
    utils.get_list = get_sorted_image_list
    dataset = AVLip(opt)
    
    if len(dataset) == 0:
        print(f"[Error] 数据集为空！请检查路径是否正确。")
        exit()
        
    print(f"[Info] 测试集样本数: {len(dataset)}")
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 【关键修复】使用自定义 collate 处理嵌套列表
    )
    
    # 4. 运行测试
    # 解包增加接收 fpr 和 fnr
    test_results = test(model, loader, gpu_id=[opt.gpu])
