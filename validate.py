import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from tqdm import tqdm  # 导入 tqdm 库

def validate(model, loader, gpu_id):

    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        # 使用 tqdm 包装数据加载器，显示进度条
        for img, crops, label in tqdm(loader, desc="Validation Progress"):
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)  # 保存原始概率值用于AP计算
    
    # === 针对多个 target_fpr 计算并打印阈值与指标（便于比较） ===
    fpr_list, tpr_list, thresholds = roc_curve(y_true, y_pred_prob)
    target_fpr_list = [0.02, 0.05, 0.1, 0.15, 0.2]

    # 动态平滑窗口：根据样本量调整窗口大小
    window_size = max(3, len(tpr_list) // 20)  # 至少为3
    if len(tpr_list) >= window_size:
        smoothed_tpr = np.convolve(tpr_list, np.ones(window_size) / window_size, mode='same')
    else:
        smoothed_tpr = tpr_list

    # 根据目标 FPR 使用线性插值选择阈值；当在阈值序列边界外时退回边界值
    def threshold_for_target_fpr(target_fpr):
        if len(fpr_list) == 0:
            return 0.5
        # fpr_list 是递增的
        if target_fpr <= fpr_list[0]:
            return float(thresholds[0])
        if target_fpr >= fpr_list[-1]:
            return float(thresholds[-1])
        idx = np.searchsorted(fpr_list, target_fpr)
        if idx < len(fpr_list) and fpr_list[idx] == target_fpr:
            return float(thresholds[idx])
        # 在 idx-1 和 idx 之间插值
        x0, x1 = fpr_list[idx - 1], fpr_list[idx]
        t0, t1 = thresholds[idx - 1], thresholds[idx]
        if x1 == x0:
            return float(t0)
        ratio = (target_fpr - x0) / (x1 - x0)
        return float(t0 + ratio * (t1 - t0))

    results_by_target = {}
    for target_fpr in target_fpr_list:
        # 首先用插值获取与目标 FPR 对齐的阈值
        optimal_threshold = threshold_for_target_fpr(target_fpr)

        # 尝试在目标附近的小容差范围内选取使（平滑后）TPR 最大的真实阈值
        tol = 0.005
        nearby_idx = np.where(np.abs(fpr_list - target_fpr) <= tol)[0]
        if len(nearby_idx) > 0:
            weights = 1 / (np.abs(fpr_list[nearby_idx] - target_fpr) + 1e-6)  # 权重：距离越近权重越高
            best_near = nearby_idx[np.argmax(smoothed_tpr[nearby_idx] * weights)]
            optimal_threshold = float(thresholds[best_near])

        y_pred_binary_t = (y_pred_prob >= optimal_threshold).astype(int)

        # 更稳健的混淆矩阵展开：若出现单类别缺失，逐样本计数以避免 ravel 出错
        cm = confusion_matrix(y_true, y_pred_binary_t)
        if cm.shape == (2, 2):
            tn_t, fp_t, fn_t, tp_t = cm.ravel()
        else:
            tn_t = fp_t = fn_t = tp_t = 0
            for yt, yp in zip(y_true, y_pred_binary_t):
                if yt == 0 and yp == 0:
                    tn_t += 1
                elif yt == 0 and yp == 1:
                    fp_t += 1
                elif yt == 1 and yp == 0:
                    fn_t += 1
                elif yt == 1 and yp == 1:
                    tp_t += 1

        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
        acc_t = (tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t) if (tp_t + tn_t + fp_t + fn_t) > 0 else 0.0

        # 对于 AP / AUC，需保证存在正负样本，否则这些指标未定义
        if len(np.unique(y_true)) > 1:
            ap_t = average_precision_score(y_true, y_pred_prob)
            auc_t = roc_auc_score(y_true, y_pred_prob)
        else:
            ap_t = 0.0
            auc_t = 0.0

        results_by_target[target_fpr] = {
            'threshold': float(optimal_threshold),
            'tn': int(tn_t), 'fp': int(fp_t), 'fn': int(fn_t), 'tp': int(tp_t),
            'fpr': float(fpr_t), 'fnr': float(fnr_t), 'acc': float(acc_t),
            'ap': float(ap_t), 'auc': float(auc_t)
        }

    # 打印比较表格（简洁输出）
    print("\nValidation summary for multiple target FPRs:")
    print("target_fpr | threshold |   TP   FP   TN   FN |   FPR   FNR   ACC    AP    AUC")
    for t in target_fpr_list:
        r = results_by_target[t]
        print(f"{t:9.2f} | {r['threshold']:.4f} | {r['tp']:5d} {r['fp']:5d} {r['tn']:5d} {r['fn']:5d} | {r['fpr']:.3f} {r['fnr']:.3f} {r['acc']:.3f} {r['ap']:.3f} {r['auc']:.3f}")

    # 兼容旧接口：返回 target_fpr=0.02 时的指标（供训练脚本使用）
    default_r = results_by_target.get(0.15, None)
    if default_r is None:
        # 如果未找到则取最接近的
        closest_t = min(results_by_target.keys(), key=lambda x: abs(x - 0.15))
        default_r = results_by_target[closest_t]

    y_pred_binary = np.where(y_pred_prob >= default_r['threshold'], 1, 0)
    optimal_threshold = default_r['threshold']
    # === 阈值调整结束 ===

    # Get AP (使用概率值)
    ap = average_precision_score(y_true, y_pred_prob)

    # 计算AUC值
    auc = roc_auc_score(y_true, y_pred_prob)

    # 混淆矩阵计算（使用二值化结果）
    cm = confusion_matrix(y_true, y_pred_binary)
    print("混淆矩阵展开后的顺序:", cm.ravel())

    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred_binary)

    # tn_correct, fp_correct, fn_correct, tp_correct = cm.ravel()
    # print(f"tn_correct={tn_correct}, fp_correct={fp_correct}, fn_correct={fn_correct}, tp_correct={tp_correct}")
    # fnr_correct = fn_correct / (fn_correct + tp_correct)
    # fpr_correct = fp_correct / (fp_correct + tn_correct)
    # acc_correct = accuracy_score(y_true, y_pred_binary)
    # print(f"正确计算: FNR_correct={fnr_correct:.4f}, FPR_correct={fpr_correct:.4f}, ACC_correct={acc_correct:.4f}")

    return ap, fpr, fnr, acc, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_list_path", type=str, default=r"E:\data\val2\0_real")
    parser.add_argument("--fake_list_path", type=str, default=r"E:\data\val2\1_fake")
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default=r"E:\data\ckpt.pth")
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    # 修改加载方式，允许部分匹配（strict=False）
    # 这样可以加载旧模型权重，同时保留新增模块的初始化权重
    model.load_state_dict(state_dict["model"], strict=False)
    print("Model loaded.")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("\n")

    model.eval()
    model.to(device)

    dataset = AVLip(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )
    ap, fpr, fnr, acc, auc = validate(model, loader, gpu_id=[opt.gpu])
    print(f"(Val) acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.3f} | fnr: {fnr:.3f} | auc: {auc:.4f}")