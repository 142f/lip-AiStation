import argparse
import json
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


def _write_text_lines(output_path, lines):
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def export_error_reports(
    output_dir,
    report_prefix,
    sample_paths,
    y_true,
    y_prob,
    threshold,
    save_all_predictions=False,
):
    os.makedirs(output_dir, exist_ok=True)
    prefix = str(report_prefix or "test").strip() or "test"

    y_pred_binary = (y_prob >= float(threshold)).astype(int)
    sample_count = int(len(y_true))

    if len(sample_paths) < sample_count:
        sample_paths = list(sample_paths) + [f"sample_{i}" for i in range(len(sample_paths), sample_count)]
    elif len(sample_paths) > sample_count:
        sample_paths = list(sample_paths[:sample_count])
    else:
        sample_paths = list(sample_paths)

    real_as_fake = []
    fake_as_real = []
    all_failed = []

    prediction_tsv_path = os.path.join(output_dir, f"{prefix}_predictions.tsv")
    if save_all_predictions:
        with open(prediction_tsv_path, "w", encoding="utf-8") as f:
            f.write("image_path\ttrue_label\tpred_label\tprob_fake\tthreshold\terror_type\n")
            for idx in range(sample_count):
                true_label = int(y_true[idx])
                pred_label = int(y_pred_binary[idx])
                prob_fake = float(y_prob[idx])
                image_path = sample_paths[idx]

                if true_label == 0 and pred_label == 1:
                    error_type = "real_as_fake"
                    real_as_fake.append(image_path)
                    all_failed.append(image_path)
                elif true_label == 1 and pred_label == 0:
                    error_type = "fake_as_real"
                    fake_as_real.append(image_path)
                    all_failed.append(image_path)
                else:
                    error_type = "correct"

                f.write(
                    f"{image_path}\t{true_label}\t{pred_label}\t{prob_fake:.8f}\t"
                    f"{float(threshold):.8f}\t{error_type}\n"
                )
    else:
        for idx in range(sample_count):
            true_label = int(y_true[idx])
            pred_label = int(y_pred_binary[idx])
            image_path = sample_paths[idx]
            if true_label == 0 and pred_label == 1:
                real_as_fake.append(image_path)
                all_failed.append(image_path)
            elif true_label == 1 and pred_label == 0:
                fake_as_real.append(image_path)
                all_failed.append(image_path)

    real_as_fake_path = os.path.join(output_dir, f"{prefix}_misclassified_real_as_fake.txt")
    fake_as_real_path = os.path.join(output_dir, f"{prefix}_misclassified_fake_as_real.txt")
    all_failed_path = os.path.join(output_dir, f"{prefix}_misclassified_all.txt")
    summary_path = os.path.join(output_dir, f"{prefix}_misclassified_summary.json")

    _write_text_lines(real_as_fake_path, real_as_fake)
    _write_text_lines(fake_as_real_path, fake_as_real)
    _write_text_lines(all_failed_path, all_failed)

    summary = {
        "threshold": float(threshold),
        "num_samples": sample_count,
        "num_misclassified_total": int(len(all_failed)),
        "num_real_as_fake": int(len(real_as_fake)),
        "num_fake_as_real": int(len(fake_as_real)),
        "real_as_fake_list": real_as_fake_path,
        "fake_as_real_list": fake_as_real_path,
        "all_failed_list": all_failed_path,
        "predictions_tsv": prediction_tsv_path if save_all_predictions else "",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["summary_json"] = summary_path
    return summary


def test(
    model,
    loader,
    gpu_id,
    threshold=0.5,
    errors_output_dir="",
    errors_prefix="test",
    save_all_predictions=False,
):
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

    if str(errors_output_dir).strip():
        sample_paths = list(getattr(loader.dataset, "total_list", []))
        summary = export_error_reports(
            output_dir=errors_output_dir,
            report_prefix=errors_prefix,
            sample_paths=sample_paths,
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
            save_all_predictions=bool(save_all_predictions),
        )
        print(f"[Info] Misclassified reports saved to: {errors_output_dir}")
        print(
            f"[Info] Misclassified: total={summary['num_misclassified_total']} | "
            f"real->fake={summary['num_real_as_fake']} | fake->real={summary['num_fake_as_real']}"
        )
        print(f"[Info] Summary file: {summary['summary_json']}")

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
    parser.add_argument("--errors_output_dir", type=str, default="", help="Directory to save misclassified sample lists.")
    parser.add_argument("--errors_prefix", type=str, default="test", help="Filename prefix for misclassified reports.")
    parser.add_argument("--save_all_predictions", action="store_true", help="Also save per-sample prediction TSV.")

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

    test(
        model,
        loader,
        gpu_id=[opt.gpu],
        threshold=float(opt.threshold),
        errors_output_dir=opt.errors_output_dir,
        errors_prefix=opt.errors_prefix,
        save_all_predictions=bool(opt.save_all_predictions),
    )
