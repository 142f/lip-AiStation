import argparse
import torch
import numpy as np
import os
import multiprocessing
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from tqdm import tqdm

# 定义默认的target_fpr值
DEFAULT_TARGET_FPR = 0.1

def validate(model, loader, gpu_id, use_amp=False):
    # 1. 环境与模式设置
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    y_true, y_pred = [], []
    
    # [优化] 将 device_type 判断移到循环外，避免重复计算
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. 推理循环
    with torch.inference_mode():
        for img, crops, label in tqdm(loader, desc="Validation Progress", leave=True):
            img_tens = img.to(device, non_blocking=True)
            
            # 处理 crops 传输
            crops_tens = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]

            # 统一推理逻辑
            if use_amp:
                # 指定 device_type
                with torch.amp.autocast(device_type=device_type):
                    features = model.get_features(img_tens)
                    output = model(crops_tens, features)[0]
            else:
                features = model.get_features(img_tens)
                output = model(crops_tens, features)[0]

            y_pred.extend(output.sigmoid().flatten().cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # [安全性修复] 检查验证集是否包含两个类别，否则 roc_curve 会报错
    if len(np.unique(y_true)) < 2:
        print("Warning: Validation set contains only one class. Metrics like AUC/ROC cannot be calculated.")
        # 返回基础指标，防止 crash
        acc = accuracy_score(y_true, (y_pred_prob >= 0.5).astype(int))
        return 0.0, 0.0, 0.0, acc, 0.0

    # 3. 全局指标计算
    global_ap = average_precision_score(y_true, y_pred_prob)
    global_auc = roc_auc_score(y_true, y_pred_prob)

    # 4. 阈值与多点评估
    fpr_list, tpr_list, thresholds = roc_curve(y_true, y_pred_prob)
    target_fpr_list = [0.02, 0.05, 0.1, 0.15, 0.2]

    results_by_target = {}

    # 修正 thresholds[0] (sklearn 特性: thresholds[0] = max_score + 1)
    thresholds[0] = min(thresholds[0], 1.0) 
    
    print("\nValidation summary for multiple target FPRs:")
    print(f"{'target_fpr':<12} | {'threshold':<10} | {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} | {'FPR':<6} {'FNR':<6} {'ACC':<6} {'AP':<6} {'AUC':<6}")

    for target_fpr in target_fpr_list:
        # 快速找到对应 FPR 的阈值 (逻辑正确：FPR增 -> Threshold减，interp 处理有效)
        optimal_threshold = np.interp(target_fpr, fpr_list, thresholds)
        
        # 二值化
        y_pred_binary_t = (y_pred_prob >= optimal_threshold).astype(int)

        # 混淆矩阵
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_binary_t, labels=[0, 1]).ravel()

        # 计算衍生指标
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
        acc_t = accuracy_score(y_true, y_pred_binary_t)

        results_by_target[target_fpr] = {
            'threshold': optimal_threshold,
            'tn': tn_t, 'fp': fp_t, 'fn': fn_t, 'tp': tp_t,
            'fpr': fpr_t, 'fnr': fnr_t, 'acc': acc_t,
            'ap': global_ap, 'auc': global_auc
        }

        print(f"{target_fpr:<12.2f} | {optimal_threshold:.4f}     | {tp_t:<6} {fp_t:<6} {tn_t:<6} {fn_t:<6} | {fpr_t:.3f}  {fnr_t:.3f}  {acc_t:.3f}  {global_ap:.3f}  {global_auc:.3f}")

    # 5. 返回默认 FPR 下的指标
    closest_t = min(results_by_target.keys(), key=lambda x: abs(x - DEFAULT_TARGET_FPR))
    res = results_by_target[closest_t]

    cm_ravel = [res['tn'], res['fp'], res['fn'], res['tp']]
    print(f"混淆矩阵展开 (Target FPR ~{closest_t}): {cm_ravel}")

    return res['ap'], res['fpr'], res['fnr'], res['acc'], res['auc']


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_list_path", type=str, default=r"E:\data\val2\0_real")
    parser.add_argument("--fake_list_path", type=str, default=r"E:\data\val2\1_fake")
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default=r"E:\data\ckpt.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    
    if os.path.exists(opt.ckpt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)
        print(f"Model loaded from {opt.ckpt}")
    else:
        print(f"Warning: Checkpoint not found at {opt.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    print("\n")

    model.to(device)

    dataset = AVLip(opt)
    
    num_workers = min(opt.workers, multiprocessing.cpu_count())
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # 在这里调用 validate，如果是独立运行，use_amp 默认为 False (即 FP32 推理)
    # 如果想测试混合精度推理，可以改为 use_amp=True，但通常 FP32 兼容性更好
    ap, fpr, fnr, acc, auc = validate(model, loader, gpu_id=[opt.gpu])
    print(f"\n(Final Result @ FPR {DEFAULT_TARGET_FPR}) acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.3f} | fnr: {fnr:.3f} | auc: {auc:.4f}")