import argparse
import json
import os
import re
import shutil
from pathlib import Path


LABEL_DIR_NAMES = {"0_real", "1_fake", "real", "fake"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRAME_FILE_RE = re.compile(r"^(?:group|frame)_\d+$", re.IGNORECASE)
LAV_REAL_PAIR_RE = re.compile(r"^(?P<pair>.+?_pair_.+?)_seg\d+_\d+$", re.IGNORECASE)
LAV_FAKE_RE = re.compile(r"^(?P<video>.+?)_seg\d+_\d+$", re.IGNORECASE)
TRAILING_FRAME_RE = re.compile(r"^(?P<video>.+)_\d+$")


def normalize_root(path_text):
    return Path(str(path_text).strip().strip('"').strip("'")).resolve()


def split_path_parts(path):
    normalized = os.path.normpath(str(path))
    drive, tail = os.path.splitdrive(normalized)
    parts = [part for part in tail.split(os.sep) if part]
    if drive:
        parts.insert(0, drive)
    return parts


def label_dir_index(parts):
    for idx in range(len(parts) - 2, -1, -1):
        if parts[idx].lower() in LABEL_DIR_NAMES:
            return idx
    return None


def make_video_key(parts, label_idx, sample_name):
    if label_idx is None:
        return os.path.normpath(sample_name)

    key_parts = parts[: label_idx + 1]
    if key_parts and key_parts[0].endswith(":"):
        return os.path.normpath(os.path.join(key_parts[0] + os.sep, *key_parts[1:], sample_name))
    return os.path.normpath(os.path.join(*key_parts, sample_name))


def extract_video_key(frame_path):
    parts = split_path_parts(frame_path)
    label_idx = label_dir_index(parts)
    stem = Path(frame_path).stem

    if FRAME_FILE_RE.match(stem) and len(parts) >= 2:
        return make_video_key(parts, label_idx, parts[-2])

    real_pair_match = LAV_REAL_PAIR_RE.match(stem)
    if real_pair_match:
        return make_video_key(parts, label_idx, real_pair_match.group("pair"))

    fake_match = LAV_FAKE_RE.match(stem)
    if fake_match:
        return make_video_key(parts, label_idx, fake_match.group("video"))

    trailing_match = TRAILING_FRAME_RE.match(stem)
    if trailing_match:
        return make_video_key(parts, label_idx, trailing_match.group("video"))

    return os.path.normpath(str(frame_path))


def collect_image_groups(input_root):
    groups = {}
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = extract_video_key(path)
        groups.setdefault(key, []).append(path)

    for files in groups.values():
        files.sort(key=lambda p: str(p).lower())
    return groups


def copy_kept_groups(input_root, output_root, kept_groups, overwrite):
    if output_root == input_root:
        raise ValueError("output_root must be different from input_root")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)

    copied = 0
    for files in kept_groups.values():
        for source in files:
            relative = source.relative_to(input_root)
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied


def build_summary(groups, kept_groups, min_frames, output_root):
    counts = [len(files) for files in groups.values()]
    kept_counts = [len(files) for files in kept_groups.values()]
    dropped = {key: files for key, files in groups.items() if key not in kept_groups}
    dropped_counts = [len(files) for files in dropped.values()]

    return {
        "min_frames": int(min_frames),
        "input_videos": len(groups),
        "input_images": int(sum(counts)),
        "kept_videos": len(kept_groups),
        "kept_images": int(sum(kept_counts)),
        "dropped_videos": len(dropped),
        "dropped_images": int(sum(dropped_counts)),
        "input_min_frames": int(min(counts)) if counts else 0,
        "input_max_frames": int(max(counts)) if counts else 0,
        "input_mean_frames": float(sum(counts) / len(counts)) if counts else 0.0,
        "output_root": str(output_root),
        "dropped_groups": [
            {"video_key": key, "frame_count": len(files)}
            for key, files in sorted(dropped.items(), key=lambda item: item[0])
        ],
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_root", default=r"E:\data\LAV-DF-test-mb")
    parser.add_argument("--output_root", default=r"E:\data\LAV-DF-test-mb-min10")
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_root = normalize_root(args.input_root)
    output_root = normalize_root(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if args.min_frames < 1:
        raise ValueError("--min_frames must be >= 1")

    groups = collect_image_groups(input_root)
    kept_groups = {
        key: files
        for key, files in groups.items()
        if len(files) >= args.min_frames
    }

    if not args.dry_run:
        copy_kept_groups(input_root, output_root, kept_groups, args.overwrite)

    summary = build_summary(groups, kept_groups, args.min_frames, output_root)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "min_video_frames_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_path"] = str(summary_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
