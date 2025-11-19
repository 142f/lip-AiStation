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
DEFAULT_TARGET_FPR = 0.15

def validate(model, loader, gpu_id):
    # 1. 环境与模式设置
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    # 检查混合精度支持
    use_amp = next(model.parameters()).dtype == torch.float16 if len(list(model.parameters())) > 0 else False
    
    model.eval()
    y_true, y_pred = [], []

    # 2. 推理循环 (使用 inference_mode 替代 no_grad，性能更优)
    with torch.inference_mode():
        for img, crops, label in tqdm(loader, desc="Validation Progress", leave=True):
            img_tens = img.to(device, non_blocking=True) # non_blocking 加速数据传输
            
            # 处理 crops 传输 (假设 crops 是 list of lists 结构)
            # 如果 crops 结构能改为 Tensor Stack 形式传输会更快，这里保持兼容性
            crops_tens = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]

            # 统一推理逻辑
            # 如果 gpu_id 是 list, device.type 可能是 'cuda'
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 使用 amp.autocast 包装，指定 device_type
            if use_amp:
                with torch.amp.autocast(device_type=device_type):
                    features = model.get_features(img_tens)
                    output = model(crops_tens, features)[0]
            else:
                features = model.get_features(img_tens)
                output = model(crops_tens, features)[0]

            # 收集结果 (先转cpu再转list，避免频繁显存交互)
            y_pred.extend(output.sigmoid().flatten().cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # 3. 全局指标计算 (提取到循环外，避免重复计算)
    # 检查是否只有单一类别，防止报错
    if len(np.unique(y_true)) > 1:
        global_ap = average_precision_score(y_true, y_pred_prob)
        global_auc = roc_auc_score(y_true, y_pred_prob)
    else:
        global_ap = 0.0
        global_auc = 0.0

    # 4. 阈值与多点评估
    fpr_list, tpr_list, thresholds = roc_curve(y_true, y_pred_prob)
    target_fpr_list = [0.02, 0.05, 0.1, 0.15, 0.2]

    results_by_target = {}

    # 使用 numpy 自带插值简化逻辑 (注意: fpr是递增的，thresholds是递减的，需翻转)
    # 解决 thresholds[0] 可能为 inf 的情况 (sklearn 特性)
    thresholds[0] = min(thresholds[0], 1.0) 
    
    print("\nValidation summary for multiple target FPRs:")
    print(f"{'target_fpr':<12} | {'threshold':<10} | {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} | {'FPR':<6} {'FNR':<6} {'ACC':<6} {'AP':<6} {'AUC':<6}")

    for target_fpr in target_fpr_list:
        # === 优化算法：快速找到对应 FPR 的阈值 ===
        # 使用 numpy.interp 进行线性插值。由于 thresholds 随 fpr 增加而减小，我们需要翻转数组来满足 xp 递增的要求
        optimal_threshold = np.interp(target_fpr, fpr_list, thresholds)
        
        # 二值化
        y_pred_binary_t = (y_pred_prob >= optimal_threshold).astype(int)

        # === 优化算法：混淆矩阵 ===
        # 指定 labels=[0, 1] 可强制输出 2x2 矩阵，即使某类样本缺失也不会报错，替代了原来的 for 循环
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_binary_t, labels=[0, 1]).ravel()

        # 计算衍生指标
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
        acc_t = accuracy_score(y_true, y_pred_binary_t)

        # 存入字典
        results_by_target[target_fpr] = {
            'threshold': optimal_threshold,
            'tn': tn_t, 'fp': fp_t, 'fn': fn_t, 'tp': tp_t,
            'fpr': fpr_t, 'fnr': fnr_t, 'acc': acc_t,
            'ap': global_ap, 'auc': global_auc
        }

        print(f"{target_fpr:<12.2f} | {optimal_threshold:.4f}     | {tp_t:<6} {fp_t:<6} {tn_t:<6} {fn_t:<6} | {fpr_t:.3f}  {fnr_t:.3f}  {acc_t:.3f}  {global_ap:.3f}  {global_auc:.3f}")

    # 5. 返回默认 FPR 下的指标 (兼容旧接口)
    # 寻找最接近 DEFAULT_TARGET_FPR 的 key
    closest_t = min(results_by_target.keys(), key=lambda x: abs(x - DEFAULT_TARGET_FPR))
    res = results_by_target[closest_t]

    # 混淆矩阵打印 (使用默认阈值)
    cm_ravel = [res['tn'], res['fp'], res['fn'], res['tp']]
    print(f"混淆矩阵展开 (Target FPR ~{closest_t}): {cm_ravel}")

    return res['ap'], res['fpr'], res['fnr'], res['acc'], res['auc']


if __name__ == "__main__":
    # 开启 cuDNN benchmark 可以在输入尺寸固定时加速
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
    # 添加 num_workers 参数，允许用户配置
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    
    if os.path.exists(opt.ckpt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        # 保持 strict=False 以适应微调或部分加载
        model.load_state_dict(state_dict["model"], strict=False)
        print(f"Model loaded from {opt.ckpt}")
    else:
        print(f"Warning: Checkpoint not found at {opt.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    print("\n")

    model.to(device)

    dataset = AVLip(opt)
    
    # 优化 DataLoader
    # 1. shuffle=False: 验证集不需要打乱，减少开销
    # 2. num_workers: 使用多进程加载数据 (通常设为 CPU 核心数或 4-8)
    # 3. pin_memory: 加速 CPU 到 GPU 的传输
    num_workers = min(opt.workers, multiprocessing.cpu_count())
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0) # 保持 worker 进程存活，减少重建开销
    )

    ap, fpr, fnr, acc, auc = validate(model, loader, gpu_id=[opt.gpu])
    print(f"\n(Final Result @ FPR {DEFAULT_TARGET_FPR}) acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.3f} | fnr: {fnr:.3f} | auc: {auc:.4f}")