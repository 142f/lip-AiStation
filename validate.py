import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from tqdm import tqdm
import os

def validate(model, loader, gpu_id):
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval() # 确保模型处于验证模式

    y_true, y_pred = [], []
    
    with torch.no_grad():
        # 使用 tqdm 包装数据加载器
        for img, crops, label in tqdm(loader, desc="Validation Progress", leave=False):
            # 优化：non_blocking=True 配合 pin_memory 加速数据传输
            img_tens = img.to(device, non_blocking=True)
            # 处理嵌套列表的 tensor 转移
            crops_tens = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
            
            features = model.get_features(img_tens).to(device, non_blocking=True)

            # 获取预测概率
            # --- 修改开始 ---
            # 1. 获取原始 logits，形状 [Batch, 2]
            logits = model(crops_tens, features)[0]

            # 2. 使用 Softmax 在维度 1 上归一化 (让两个概率加起来等于 1)
            probs = torch.softmax(logits, dim=1)

            # 3. 取出第 1 列 (索引 1) 作为 "Fake" 类别的概率
            # 假设: Index 0 = Real, Index 1 = Fake (通常做法)
            # 这一步后形状变为 [Batch]，与标签数量对齐
            pred_score = probs[:, 1] 

            # 4. 转为 list (如果需要 flatten 可以加，但通常一维张量不需要)
            pred_score = pred_score.flatten()
            # --- 修改结束 ---
            
            y_pred.extend(pred_score.tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # --- 核心优化：指标计算 ---
    
    # 1. 安全性检查：防止全0或全1导致报错
    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class present in y_true. AUC/AP cannot be calculated correctly.")
        return 0, 0, 0, 0, 0

    # 2. 计算 AUC (最高优先级)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.0

    # 3. 计算 AP (次高优先级)
    ap = average_precision_score(y_true, y_pred_prob)

    # 4. 寻找最佳阈值 (Best Accuracy)
    # 既然 AUC > ACC，说明我们关心分布。默认 0.5 阈值可能不是最优的。
    # 这里利用 ROC 曲线寻找 Youden's J statistic (TPR - FPR) 最大的点作为最佳阈值
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_prob)
    # Youden's J index = TPR - FPR = TPR - (1 - TNR)
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 使用最佳阈值计算二值化结果
    y_pred_binary = np.where(y_pred_prob >= best_threshold, 1, 0)
    
    # 5. 计算 ACC 及 混淆矩阵
    acc = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 防止除零错误
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Conf Matrix: [TN:{tn} FP:{fp} | FN:{fn} TP:{tp}]")
    print(f"Best Threshold found: {best_threshold:.4f} (Default is 0.5)")

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
    # 增加 worker 参数，加速数据加载
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")

    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    
    # 加载权重
    if os.path.exists(opt.ckpt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        # 处理可能存在的 'model' 键
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {opt.ckpt}")
        if missing_keys:
            print(f"Missing keys (partial load): {len(missing_keys)}")
    else:
        print(f"Warning: Checkpoint {opt.ckpt} not found!")

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {total_params / 1e6:.2f}M")
    print("\n" + "="*30)
    
    model.eval()
    model.to(device)

    dataset = AVLip(opt)
    
    # 优化 DataLoader: shuffle=False (验证集通常不需要打乱), num_workers加速, pin_memory加速
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.workers,
        pin_memory=True 
    )
    
    ap, fpr, fnr, acc, auc = validate(model, loader, gpu_id=[opt.gpu])
    
    # 优化输出格式：强调 AUC > AP > ACC
    print("="*30)
    print(f"AUC : {auc:.4f}")
    print(f"AP  : {ap:.4f}")
    print(f"ACC : {acc:.4f}")
    print(f"----------------")
    print(f"FPR : {fpr:.4f}")
    print(f"FNR : {fnr:.4f}")
    print("="*30)