import argparse
import os

import torch
import torch.utils.data

from evaluation.binary import evaluate_fixed_threshold, nested_crops_collate, run_binary_inference
from evaluation.runtime import load_checkpoint, set_ablation_env
from models import build_model

try:
    from data.datasets import AVLip
except ImportError:
    from data import AVLip


def test(model, loader, gpu_id, threshold=0.5):
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    y_true, y_prob = run_binary_inference(
        model=model,
        loader=loader,
        device=device,
        desc="Testing",
        leave=True,
    )

    metrics = evaluate_fixed_threshold(y_true, y_prob, threshold=threshold)
    if metrics["single_class"]:
        print("Warning: test set has one class only. AUC/AP are undefined.")

    print("\n" + "#" * 60)
    print("Final Test Report")
    print(f"Samples    : {len(y_true)}")
    print(f"Threshold  : {metrics['threshold']:.4f} (fixed)")
    print(
        f"ConfMatrix : [TN: {metrics['tn']}  FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}  TP: {metrics['tp']}]"
    )
    print("-" * 30)
    print(f"AUC        : {metrics['auc']:.4f}")
    print(f"AP         : {metrics['ap']:.4f}")
    print(f"ACC        : {metrics['acc']:.4f}")
    print(f"F1         : {metrics['f1']:.4f}")
    print("#" * 60 + "\n")

    return (
        metrics["auc"],
        metrics["ap"],
        metrics["acc"],
        metrics["f1"],
        metrics["threshold"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
        metrics["tp"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--real_list_path", type=str, required=True)
    parser.add_argument("--fake_list_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pth")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for ACC/F1")
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow partial checkpoint loading")

    parser.add_argument("--data_label", type=str, default="test")
    parser.add_argument("--fix_backbone", action="store_true")
    parser.add_argument("--fix_encoder", action="store_true")
    parser.add_argument("--name", type=str, default="test_experiment")

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
    print(f"[Info] Device: {device}")

    model = build_model(opt.arch)
    model.to(device)

    if not os.path.exists(opt.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {opt.ckpt}")

    print(f"[Info] Loading checkpoint: {opt.ckpt}")
    info = load_checkpoint(model, opt.ckpt, allow_partial_load=opt.allow_partial_load, prefer_ema=True)
    print(f"[Info] Checkpoint source: {info['source']}")

    dataset = AVLip(opt)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check real/fake paths.")

    print(f"[Info] Test samples: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=nested_crops_collate,
    )

    test(model, loader, gpu_id=[opt.gpu], threshold=float(opt.threshold))
