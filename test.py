import argparse
import torch
import numpy as np
import os
import torch.utils.data
from models import build_model
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


# ==========================================
# 核心测试函数
# ==========================================
def test(model, loader, gpu_id):
    """
    在测试集上评估模型性能
    """
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    y_true, y_pred = [], []
    
    print("\n" + "="*20 + " 开始测试循环 (Testing Loop) " + "="*20)
    
    with torch.no_grad():
        for i, (img, crops, label) in enumerate(tqdm(loader, desc="Testing", leave=True)):
            # 1. 处理 img
            img_tens = img.to(device, non_blocking=True)
            
            # 2. 处理 crops (List[List[Tensor]])
            # collate_fn 已将其转换为 [3尺度][5区域][B, C, H, W] 结构
            # 将每个 Tensor 移动到 device
            crops_tens = [[t.to(device, non_blocking=True) for t in scale_crops] for scale_crops in crops]
            
            # 3. 模型推理
            # 注意：get_features 返回的 tensor 已经在 model 所在的 device 上
            features = model.get_features(img_tens)
            # model.forward 返回 (logits, weights_max, weights_org)
            logits = model(crops_tens, features)[0]
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            pred_score = probs[:, 1]  # 取 Fake 类概率
            
            y_pred.extend(pred_score.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # --- 指标计算 ---
    if len(np.unique(y_true)) < 2:
        print("Warning: 测试集中只包含一类数据，无法计算 AUC/AP。")
        return 0, 0, 0, 0

    # AUC
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError as e:
        print(f"Warning: 计算 AUC 时出错: {e}")
        auc = 0.0

    # AP
    ap = average_precision_score(y_true, y_pred_prob)
    
    # 寻找最佳阈值
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred_binary = np.where(y_pred_prob >= best_threshold, 1, 0)
    
    # ACC & Confusion Matrix
    acc = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # F1
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    
    print("\n" + "#"*60)
    print(f"【最终测试报告 / Test Report】")
    print(f"数据量    : {len(y_true)} 样本")
    print(f"混淆矩阵  : [TN: {tn}  FP: {fp} | FN: {fn}  TP: {tp}]")
    print("-" * 30)
    print(f"AUC       : {auc:.4f}")
    print(f"AP        : {ap:.4f}")
    print(f"ACC       : {acc:.4f}")
    print(f"Best F1   : {best_f1:.4f}")
    print(f"最佳阈值  : {best_threshold:.4f}")
    print("#"*60 + "\n")

    return auc, ap, acc, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # 路径参数
    parser.add_argument("--real_list_path", type=str, required=True)
    parser.add_argument("--fake_list_path", type=str, required=True)
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

    opt = parser.parse_args()

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
    test(model, loader, gpu_id=[opt.gpu])