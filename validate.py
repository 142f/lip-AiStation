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
    y_pred_binary = np.where(y_pred_prob >= 0.5, 1, 0)  # 二值化用于其他指标

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
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr} auc: {auc}")