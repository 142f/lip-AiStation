import argparse
import torch
import numpy as np
import os
import torch.utils.data
from models import build_model
from data import AVLip
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve
from tqdm import tqdm

# ==========================================
# 核心测试函数 (基于 validate 修改)
# ==========================================
def test(model, loader, gpu_id):
    """
    在测试集上评估模型性能
    """
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    model.eval() # 强制评估模式
    
    y_true, y_pred = [], []
    
    print("\n" + "="*20 + " 开始测试 (Testing Phase) " + "="*20)
    
    with torch.no_grad():
        # 修改描述为 Testing Progress
        for img, crops, label in tqdm(loader, desc="Testing Progress", leave=True):
            img_tens = img.to(device, non_blocking=True)
            # 同步 validate.py 的数据处理逻辑 (5D Tensor)
            crops_tens = crops.to(device, non_blocking=True)
            
            features = model.get_features(img_tens).to(device, non_blocking=True)
            
            # 推理逻辑 (保持与 validate 一致)
            logits = model(crops_tens, features)[0]
            probs = torch.softmax(logits, dim=1)
            pred_score = probs[:, 1] # 取 Fake 类的概率
            
            y_pred.extend(pred_score.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred)

    # --- 指标计算 ---
    if len(np.unique(y_true)) < 2:
        print("Warning: 测试集中只包含一类数据，无法计算 AUC/AP。")
        return 0, 0, 0, 0

    # 1. AUC
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = 0.0

    # 2. AP
    ap = average_precision_score(y_true, y_pred_prob)

    # 3. 寻找最佳阈值 (Youden's J)
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred_binary = np.where(y_pred_prob >= best_threshold, 1, 0)
    
    # 4. ACC & Confusion Matrix
    acc = accuracy_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # 5. F1 Score (Best Theoretical)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    
    # --- 打印最终测试报告 ---
    print("\n" + "#"*60)
    print(f"【最终测试报告 / Test Report】")
    print(f"数据量    : {len(y_true)} 样本")
    print(f"混淆矩阵  : [TN: {tn}  FP: {fp} | FN: {fn}  TP: {tp}]")
    print("-" * 30)
    print(f"AUC       : {auc:.4f}  <-- 核心指标")
    print(f"AP        : {ap:.4f}")
    print(f"ACC       : {acc:.4f}")
    print(f"Best F1   : {best_f1:.4f}")
    print(f"最佳阈值  : {best_threshold:.4f}")
    print("#"*60 + "\n")

    return auc, ap, acc, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ================= 配置区域 =================
    # 1. 默认路径指向你的【测试集】 (E:\data\processed_dataset_25_v2\test)
    parser.add_argument("--real_list_path", type=str, default=r"/3240608030/val-test/val-test/val/0_real")
    parser.add_argument("--fake_list_path", type=str, default=r"/3240608030/val-test/val-test/val/1_fake")
    
    # 2. 默认权重指向【最佳模型】 (best_model.pth)
    # 请确保这里的实验名称 (My_Experiment) 与你训练时设置的 --name 一致
    parser.add_argument("--ckpt", type=str, default=r"/3240608030/checkpoints/experiment_name/best_model.pth", 
                        help="Path to the best_model.pth")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--workers", type=int, default=4)
    
    # 注意：datasets.py 里通常只允许 'train' 或 'val'，这里借用 'val' 模式即可
    parser.add_argument("--data_label", type=str, default="val") 
    
    # 兼容性参数 (模型构建可能需要)
    parser.add_argument("--num_classes", type=int, default=2) 
    parser.add_argument("--fix_backbone", action='store_true')
    parser.add_argument("--fix_encoder", action='store_true')
    
    # 添加 --name 参数以避免报错，虽然在测试脚本中可能不直接使用它来创建目录，但某些组件可能需要
    parser.add_argument("--name", type=str, default="test_experiment", help="Experiment name")

    opt = parser.parse_args()

    # ================= 模型加载逻辑 (核心修改) =================
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用设备: {device}")

    # 1. 构建模型结构
    model = build_model(opt.arch)
    model.to(device)

    # 2. 加载权重 (优先加载 EMA)
    if os.path.exists(opt.ckpt):
        print(f"[Info] 正在加载权重: {opt.ckpt}")
        checkpoint = torch.load(opt.ckpt, map_location="cpu")
        
        state_dict = None
        load_source = "Unknown"

        # 优先级 1: EMA 权重 (泛化性最好)
        if "model_ema" in checkpoint:
            state_dict = checkpoint["model_ema"]
            load_source = "EMA Weights (最佳泛化性能)"
        # 优先级 2: 普通权重 (model)
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            load_source = "Standard Weights"
        # 优先级 3: 直接是权重字典
        else:
            state_dict = checkpoint
            load_source = "Raw State Dict"

        # 加载
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Success] 成功加载 {load_source}")
        if msg.missing_keys:
            print(f"         丢失键 (部分加载): {len(msg.missing_keys)} 个")
    else:
        print(f"[Error] 找不到权重文件: {opt.ckpt}")
        exit()

    # ================= 数据准备与执行 =================
    dataset = AVLip(opt)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False,      # 测试集不需要打乱
        num_workers=opt.workers,
        pin_memory=True
    )
    
    print(f"[Info] 测试集样本数: {len(dataset)}")
    
    # 开始测试
    test(model, loader, gpu_id=[opt.gpu])
