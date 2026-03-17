import argparse
import os

import torch
import torch.utils.data

from data import AVLip
from evaluation.binary import evaluate_youden_threshold, run_binary_inference
from evaluation.runtime import load_checkpoint, set_ablation_env
from models import build_model


def validate(model, loader, gpu_id):
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    y_true, y_prob = run_binary_inference(
        model=model,
        loader=loader,
        device=device,
        desc="Validation Progress",
        leave=False,
    )

    metrics = evaluate_youden_threshold(y_true, y_prob)
    if metrics["single_class"]:
        print("Warning: Only one class present in y_true. AUC/AP cannot be calculated correctly.")
        return 0, 0, 0, 0, 0, 0

    print(
        f"Conf Matrix : [TN: {metrics['tn']:>4}  FP: {metrics['fp']:>4} | "
        f"FN: {metrics['fn']:>4}  TP: {metrics['tp']:>4}]"
    )
    print(f"Threshold   : {metrics['threshold']:.4f} (Youden)")
    print(
        f"F1 Scores   : Youden: {metrics['f1_at_youden']:.4f} | "
        f"Best: {metrics['f1']:.4f}"
    )

    return (
        metrics["ap"],
        metrics["fpr"],
        metrics["fnr"],
        metrics["acc"],
        metrics["auc"],
        metrics["f1"],
    )


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
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow partial checkpoint loading")

    parser.add_argument("--no_innov", action="store_true")
    parser.add_argument("--no_modality_bias", action="store_true")
    parser.add_argument("--no_attn_bias", action="store_true")
    parser.add_argument("--no_se_fusion", action="store_true")
    parser.add_argument("--no_residual_cls", action="store_true")
    parser.add_argument("--no_region_innov", action="store_true")
    parser.add_argument("--no_region_pe", action="store_true")
    parser.add_argument("--no_region_se", action="store_true")

    opt = parser.parse_args()
    set_ablation_env(opt)

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(opt.arch)

    if os.path.exists(opt.ckpt):
        info = load_checkpoint(model, opt.ckpt, allow_partial_load=opt.allow_partial_load, prefer_ema=False)
        print(f"Model loaded from {opt.ckpt} ({info['source']})")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {opt.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {total_params / 1e6:.2f}M")
    print("\n" + "=" * 30)

    model.eval()
    model.to(device)

    dataset = AVLip(opt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
    )

    ap, fpr, fnr, acc, auc, f1 = validate(model, loader, gpu_id=[opt.gpu])

    print("=" * 30)
    print(f"AUC : {auc:.4f}")
    print(f"AP  : {ap:.4f}")
    print(f"ACC : {acc:.4f}")
    print(f"F1  : {f1:.4f}")
    print("----------------")
    print(f"FPR : {fpr:.4f}")
    print(f"FNR : {fnr:.4f}")
    print("=" * 30)
