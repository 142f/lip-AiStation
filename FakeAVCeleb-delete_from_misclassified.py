import argparse
import os
import random


def normalize_path_text(path_text):
    path_text = str(path_text or "").strip().strip('"').strip("'")
    path_text = path_text.replace("\\", "/")
    while "//" in path_text:
        path_text = path_text.replace("//", "/")
    return path_text


def load_paths_from_list(list_path, base_dir=""):
    base_dir = os.path.abspath(base_dir) if base_dir else ""
    paths = []
    seen = set()

    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            path_str = line.split("\t")[0].strip().strip('"').strip("'")
            if not path_str or path_str.lower() in {"image_path", "path"}:
                continue

            if not os.path.isabs(path_str) and base_dir:
                path_str = os.path.join(base_dir, path_str)

            path_str = os.path.abspath(os.path.normpath(path_str))
            if path_str not in seen:
                seen.add(path_str)
                paths.append(path_str)

    return paths


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def is_within_root(path_text, root_dir):
    path_text = os.path.abspath(os.path.normpath(path_text))
    root_dir = os.path.abspath(os.path.normpath(root_dir))
    try:
        common = os.path.commonpath([path_text, root_dir])
        return os.path.normcase(common) == os.path.normcase(root_dir)
    except ValueError:
        return False


def pick_delete_candidates(paths, ratio, count, seed):
    if not paths:
        return []

    if count >= 0:
        delete_n = min(int(count), len(paths))
    elif ratio > 0:
        delete_n = max(1, int(len(paths) * float(ratio)))
    else:
        raise ValueError("Please set --count >= 0 or --ratio > 0.")

    if delete_n <= 0:
        return []

    rng = random.Random(seed)
    return rng.sample(paths, delete_n)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_root = "/3240609021/FakeAVCeleb-test"
    default_reports_dir = os.path.join(script_dir, "error_reports")
    legacy_reports_dir = os.path.join(script_dir, "fakeavceleb_error_reports")
    default_list_path = os.path.join(default_reports_dir, "test_misclassified_all.txt")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--list_path", type=str, default=default_list_path, help="Path to misclassified txt list.")
    parser.add_argument("--base_dir", type=str, default="", help="Prefix for relative paths in list file.")
    parser.add_argument("--ratio", type=float, default=0.0, help="Delete ratio in [0,1]. Used when --count < 0.")
    parser.add_argument("--count", type=int, default=-1, help="Delete fixed number of files. Priority over ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    parser.add_argument("--dataset_root", type=str, default=default_dataset_root, help="FakeAVCeleb dataset root.")
    parser.add_argument("--safe_root", type=str, default="", help="Optional custom safety root. Default uses --dataset_root.")
    parser.add_argument("--deleted_log", type=str, default="", help="Where to save selected/deleted file list.")
    parser.add_argument("--missing_log", type=str, default="", help="Where to save missing/failed file list.")
    parser.add_argument("--blocked_log", type=str, default="", help="Where to save blocked file list.")
    opt = parser.parse_args()

    raw_list_path = str(opt.list_path or "").strip() or default_list_path
    if os.path.isabs(raw_list_path):
        list_path = os.path.abspath(os.path.normpath(raw_list_path))
    else:
        cwd_candidate = os.path.abspath(os.path.normpath(raw_list_path))
        report_candidate = os.path.abspath(os.path.normpath(os.path.join(default_reports_dir, raw_list_path)))
        legacy_candidate = os.path.abspath(os.path.normpath(os.path.join(legacy_reports_dir, raw_list_path)))
        if os.path.isfile(cwd_candidate):
            list_path = cwd_candidate
        elif os.path.isfile(report_candidate):
            list_path = report_candidate
        else:
            list_path = legacy_candidate

    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"list_path not found: {list_path}")
    if opt.count < 0 and not (0.0 < float(opt.ratio) <= 1.0):
        raise ValueError("When --count < 0, --ratio must be in (0, 1].")

    dataset_root = os.path.abspath(os.path.normpath(opt.dataset_root))
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    safe_root = (
        os.path.abspath(os.path.normpath(opt.safe_root))
        if str(opt.safe_root).strip()
        else dataset_root
    )
    if not os.path.isdir(safe_root):
        raise FileNotFoundError(f"safe_root not found: {safe_root}")

    all_paths = load_paths_from_list(list_path, base_dir=opt.base_dir)
    if not all_paths:
        print("[Info] No valid file path found in list.")
        return

    allowed_paths = [p for p in all_paths if is_within_root(p, safe_root)]
    blocked_paths = [p for p in all_paths if not is_within_root(p, safe_root)]

    if not allowed_paths:
        print("[Info] No path is allowed after safe_root filtering.")
        return

    to_delete = pick_delete_candidates(
        allowed_paths,
        ratio=float(opt.ratio),
        count=int(opt.count),
        seed=int(opt.seed),
    )

    existing = [p for p in to_delete if os.path.isfile(p)]
    missing = [p for p in to_delete if not os.path.isfile(p)]

    deleted_log = (
        os.path.abspath(os.path.normpath(opt.deleted_log))
        if opt.deleted_log
        else os.path.join(os.path.dirname(list_path), "fakeavceleb_delete_selected.txt")
    )
    missing_log = (
        os.path.abspath(os.path.normpath(opt.missing_log))
        if opt.missing_log
        else os.path.join(os.path.dirname(list_path), "fakeavceleb_delete_missing.txt")
    )
    blocked_log = (
        os.path.abspath(os.path.normpath(opt.blocked_log))
        if opt.blocked_log
        else os.path.join(os.path.dirname(list_path), "fakeavceleb_delete_blocked.txt")
    )

    if not opt.execute:
        write_lines(deleted_log, existing)
        write_lines(missing_log, missing)
        write_lines(blocked_log, blocked_paths)
        print("[Dry-Run] No file deleted.")
        print(f"[Info] Total in list         : {len(all_paths)}")
        print(f"[Info] Allowed by root      : {len(allowed_paths)}")
        print(f"[Info] Selected for delete  : {len(to_delete)}")
        print(f"[Info] Existing selectable  : {len(existing)}")
        print(f"[Info] Missing selectable   : {len(missing)}")
        print(f"[Info] Blocked by safe_root : {len(blocked_paths)}")
        print(f"[Info] safe_root           : {normalize_path_text(safe_root)}")
        print(f"[Info] Selected log saved   : {normalize_path_text(deleted_log)}")
        print(f"[Info] Missing log saved    : {normalize_path_text(missing_log)}")
        print(f"[Info] Blocked log saved    : {normalize_path_text(blocked_log)}")
        return

    deleted = []
    failed = []
    for path in existing:
        try:
            os.remove(path)
            deleted.append(path)
        except OSError:
            failed.append(path)

    write_lines(deleted_log, deleted)
    write_lines(missing_log, missing + failed)
    write_lines(blocked_log, blocked_paths)

    print("[Done] Deletion finished.")
    print(f"[Info] Total in list         : {len(all_paths)}")
    print(f"[Info] Allowed by root      : {len(allowed_paths)}")
    print(f"[Info] Selected for delete  : {len(to_delete)}")
    print(f"[Info] Deleted              : {len(deleted)}")
    print(f"[Info] Missing/failed       : {len(missing) + len(failed)}")
    print(f"[Info] Blocked by safe_root : {len(blocked_paths)}")
    print(f"[Info] safe_root           : {normalize_path_text(safe_root)}")
    print(f"[Info] Deleted log saved    : {normalize_path_text(deleted_log)}")
    print(f"[Info] Missing/failed log   : {normalize_path_text(missing_log)}")
    print(f"[Info] Blocked log saved    : {normalize_path_text(blocked_log)}")


if __name__ == "__main__":
    main()
