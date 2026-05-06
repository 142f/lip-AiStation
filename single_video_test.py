import argparse
import json
import os
import shutil
from types import SimpleNamespace

import torch

import utils
from data.datasets import AVLip
from models import build_model
from test import aggregate_video_score, custom_collate_fn, get_sorted_image_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess one video/audio pair and run single-video inference."
    )
    parser.add_argument("--video_file", type=str, required=True, help="Input mp4 file.")
    parser.add_argument("--audio_file", type=str, required=True, help="Input wav file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="./temp/single_video",
        help="Working root used when --preprocessed_dir is not set.",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="",
        help="Exact directory for preprocessed image windows.",
    )
    parser.add_argument("--temp_dir", type=str, default="./temp")
    parser.add_argument(
        "--label",
        type=str,
        default="0_real",
        choices=["0_real", "1_fake", "real", "fake"],
        help="Dummy label folder for preprocessing. It does not affect fake probability.",
    )
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--agg_method",
        type=str,
        default="top3_mean",
        choices=["mean", "max", "top3_mean", "top5_mean", "median"],
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Delete this video's preprocessed folder before writing new images.",
    )
    parser.add_argument(
        "--result_only",
        action="store_true",
        help="Print only the final Chinese label.",
    )
    return parser.parse_args()


def load_model(ckpt_path, arch, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(arch)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_ema" in checkpoint:
        state_dict = checkpoint["model_ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[7:]
            elif name.startswith("_orig_mod."):
                name = name[10:]
        clean_state_dict[name] = value

    msg = model.load_state_dict(clean_state_dict, strict=False)
    if msg.missing_keys:
        print(f"[Warning] missing checkpoint keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[Warning] unexpected checkpoint keys: {len(msg.unexpected_keys)}")

    model.to(device)
    model.eval()
    return model


def build_dataset_options(work_root, dataset_name, video_output_dir):
    empty_real_dir = os.path.join(work_root, "_empty_real")
    empty_fake_dir = os.path.join(work_root, "_empty_fake")
    os.makedirs(empty_real_dir, exist_ok=True)
    os.makedirs(empty_fake_dir, exist_ok=True)

    if dataset_name == "1_fake":
        real_dir = empty_real_dir
        fake_dir = video_output_dir
    else:
        real_dir = video_output_dir
        fake_dir = empty_fake_dir

    return SimpleNamespace(
        real_list_path=real_dir,
        fake_list_path=fake_dir,
        data_label="test",
        num_classes=2,
        fix_backbone=False,
        fix_encoder=False,
        name="single_video_test",
    )


def run_inference(model, loader, device, agg_method):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    frame_probs = []
    with torch.no_grad():
        for img, raw_crops, _label in loader:
            img = img.to(device, non_blocking=True)
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            img_tens = img.sub(mean).div(std)

            crops_tens = []
            for scale_list in raw_crops:
                processed_scale = []
                for crop_batch in scale_list:
                    processed_scale.append(crop_batch.to(device, non_blocking=True))
                crops_tens.append(processed_scale)

            features = model.get_features(img_tens)
            logits = model(crops_tens, features)[0]
            probs = torch.softmax(logits, dim=1)[:, 1]
            frame_probs.extend(probs.flatten().cpu().tolist())

    if not frame_probs:
        raise RuntimeError("No preprocessed image windows were found for inference.")

    video_prob = aggregate_video_score(frame_probs, agg_method)
    return video_prob, frame_probs


def main():
    args = parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_file))[0]
    dataset_name = args.label

    work_root = os.path.abspath(args.output_root)
    video_output_dir = os.path.abspath(args.preprocessed_dir) if args.preprocessed_dir else ""
    if video_output_dir:
        output_label_dir = os.path.dirname(video_output_dir)
        work_root = os.path.dirname(output_label_dir) or work_root
    else:
        output_label_dir = os.path.join(work_root, dataset_name)
        video_output_dir = os.path.join(output_label_dir, video_name)

    if args.clean_output and os.path.isdir(video_output_dir):
        shutil.rmtree(video_output_dir)

    os.makedirs(work_root, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    if not os.path.isdir(video_output_dir) or not os.listdir(video_output_dir):
        from preprocess import process_video_file

        preprocess_args = SimpleNamespace(
            n_extract=10,
            window_len=5,
            temp_dir=args.temp_dir,
            output_video_dir=video_output_dir,
        )
        saved_count, video_output_dir = process_video_file(
            args.video_file,
            args.audio_file,
            output_label_dir,
            preprocess_args,
        )
        if saved_count <= 0:
            raise RuntimeError(f"Preprocess produced no image windows: {video_output_dir}")

    utils.get_list = get_sorted_image_list
    dataset_opt = build_dataset_options(work_root, dataset_name, video_output_dir)
    dataset = AVLip(dataset_opt)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, args.arch, device)
    fake_prob, frame_probs = run_inference(model, loader, device, args.agg_method)
    pred_label = "fake" if fake_prob >= args.threshold else "real"

    result = {
        "video_file": os.path.abspath(args.video_file),
        "audio_file": os.path.abspath(args.audio_file),
        "preprocessed_dir": os.path.abspath(video_output_dir),
        "num_windows": int(len(frame_probs)),
        "agg_method": args.agg_method,
        "threshold": float(args.threshold),
        "fake_probability": float(fake_prob),
        "prediction": pred_label,
        "prediction_zh": "\u5047" if pred_label == "fake" else "\u771f",
        "frame_fake_probabilities": [float(x) for x in frame_probs],
    }

    if args.result_only:
        print(result["prediction_zh"])
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
