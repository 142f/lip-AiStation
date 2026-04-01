import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NAME_PATTERN = re.compile(r"^(?P<metadata_id>.+)_(?P<frame_index>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract original metadata/video ids from preprocessed image names."
    )
    parser.add_argument(
        "--dataset-root",
        default=r"E:\data\test-pro\test",
        help="Dataset root containing folders like 0_real and 1_fake.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("data") / "test_pro_test_id_export"),
        help="Directory used to save csv/json/txt outputs.",
    )
    return parser.parse_args()


def discover_split_dirs(dataset_root: Path):
    preferred = ["0_real", "1_fake", "real", "fake"]
    existing = []

    for name in preferred:
        candidate = dataset_root / name
        if candidate.is_dir():
            existing.append(candidate)

    if existing:
        return existing

    return sorted([path for path in dataset_root.iterdir() if path.is_dir()])


def parse_image_name(file_path: Path):
    match = NAME_PATTERN.match(file_path.stem)
    if not match:
        return None

    return {
        "image_name": file_path.name,
        "metadata_id": match.group("metadata_id"),
        "frame_index": int(match.group("frame_index")),
        "full_path": str(file_path),
    }


def collect_records(split_dir: Path):
    records = []
    skipped = []

    for file_path in sorted(split_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parsed = parse_image_name(file_path)
        if parsed is None:
            skipped.append(file_path.name)
            continue

        parsed["split"] = split_dir.name
        records.append(parsed)

    return records, skipped


def build_summary(records_by_split, skipped_by_split):
    summary = {
        "splits": {},
        "shared_metadata_ids": [],
        "unique_to_split": {},
    }

    split_to_id_set = {}

    for split_name, records in records_by_split.items():
        id_counter = Counter(record["metadata_id"] for record in records)
        frame_counter = defaultdict(list)

        for record in records:
            frame_counter[record["metadata_id"]].append(record["frame_index"])

        split_to_id_set[split_name] = set(id_counter)
        summary["splits"][split_name] = {
            "image_count": len(records),
            "unique_metadata_count": len(id_counter),
            "metadata_ids": sorted(id_counter),
            "images_per_metadata": dict(sorted(id_counter.items(), key=lambda item: item[0])),
            "frame_indices_per_metadata": {
                metadata_id: sorted(indices)
                for metadata_id, indices in sorted(frame_counter.items(), key=lambda item: item[0])
            },
            "skipped_files": skipped_by_split.get(split_name, []),
        }

    if len(split_to_id_set) >= 2:
        split_names = sorted(split_to_id_set)
        shared = set.intersection(*(split_to_id_set[name] for name in split_names))
        summary["shared_metadata_ids"] = sorted(shared)
        for split_name in split_names:
            other_sets = [split_to_id_set[name] for name in split_names if name != split_name]
            others_union = set().union(*other_sets) if other_sets else set()
            summary["unique_to_split"][split_name] = sorted(split_to_id_set[split_name] - others_union)

    return summary


def write_csv(records_by_split, output_dir: Path):
    output_path = output_dir / "image_to_metadata.csv"
    fieldnames = ["split", "image_name", "metadata_id", "frame_index", "full_path"]

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in sorted(records_by_split):
            for record in records_by_split[split_name]:
                writer.writerow({key: record[key] for key in fieldnames})

    return output_path


def write_id_lists(summary, output_dir: Path):
    output_paths = []
    for split_name, split_info in summary["splits"].items():
        output_path = output_dir / f"{split_name}_metadata_ids.txt"
        output_path.write_text("\n".join(split_info["metadata_ids"]) + "\n", encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def write_summary(summary, output_dir: Path):
    output_path = output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_console_summary(summary):
    for split_name, split_info in summary["splits"].items():
        print(f"[{split_name}]")
        print(f"  image_count: {split_info['image_count']}")
        print(f"  unique_metadata_count: {split_info['unique_metadata_count']}")
        sample = split_info["metadata_ids"][:10]
        print(f"  first_10_metadata_ids: {sample}")

    if summary["shared_metadata_ids"]:
        print(f"shared_metadata_count: {len(summary['shared_metadata_ids'])}")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    split_dirs = discover_split_dirs(dataset_root)
    if not split_dirs:
        raise FileNotFoundError(f"No split folders found under: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = {}
    skipped_by_split = {}

    for split_dir in split_dirs:
        records, skipped = collect_records(split_dir)
        records_by_split[split_dir.name] = records
        skipped_by_split[split_dir.name] = skipped

    summary = build_summary(records_by_split, skipped_by_split)
    csv_path = write_csv(records_by_split, output_dir)
    write_id_lists(summary, output_dir)
    summary_path = write_summary(summary, output_dir)

    print_console_summary(summary)
    print(f"csv_saved_to: {csv_path}")
    print(f"summary_saved_to: {summary_path}")


if __name__ == "__main__":
    main()
