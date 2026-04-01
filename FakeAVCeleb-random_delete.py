import argparse
import os
import random


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def normalize_path_text(path_text):
    path_text = str(path_text or "").strip().strip('"').strip("'")
    path_text = path_text.replace("\\", "/")
    while "//" in path_text:
        path_text = path_text.replace("//", "/")
    return path_text


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTS


def list_images_recursive(root_dir):
    image_paths = []
    for current_root, _, files in os.walk(root_dir):
        for filename in files:
            if is_image_file(filename):
                image_paths.append(os.path.abspath(os.path.join(current_root, filename)))
    image_paths.sort()
    return image_paths


def pick_candidates(paths, delete_count, seed):
    if delete_count <= 0 or not paths:
        return []
    delete_count = min(int(delete_count), len(paths))
    rng = random.Random(seed)
    return rng.sample(paths, delete_count)


def delete_files(paths):
    deleted = []
    failed = []
    for path in paths:
        try:
            os.remove(path)
            deleted.append(path)
        except OSError:
            failed.append(path)
    return deleted, failed


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_root = "/3240609021/FakeAVCeleb-test"
    default_output_dir = os.path.join(script_dir, "fakeavceleb_error_reports")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", type=str, default=default_dataset_root, help="FakeAVCeleb dataset root.")
    parser.add_argument("--real_count", type=int, default=0, help="Number of real images to delete.")
    parser.add_argument("--fake_count", type=int, default=0, help="Number of fake images to delete.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save logs.")
    opt = parser.parse_args()

    if int(opt.real_count) < 0 or int(opt.fake_count) < 0:
        raise ValueError("--real_count and --fake_count must be >= 0.")
    if int(opt.real_count) == 0 and int(opt.fake_count) == 0:
        raise ValueError("At least one of --real_count or --fake_count must be > 0.")

    dataset_root = os.path.abspath(os.path.normpath(opt.dataset_root))
    real_dir = os.path.join(dataset_root, "real")
    fake_dir = os.path.join(dataset_root, "fake")

    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"real dir not found: {real_dir}")
    if not os.path.isdir(fake_dir):
        raise FileNotFoundError(f"fake dir not found: {fake_dir}")

    real_paths = list_images_recursive(real_dir)
    fake_paths = list_images_recursive(fake_dir)

    real_selected = pick_candidates(real_paths, int(opt.real_count), int(opt.seed))
    fake_selected = pick_candidates(fake_paths, int(opt.fake_count), int(opt.seed) + 1)

    output_dir = os.path.abspath(os.path.normpath(opt.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    selected_real_log = os.path.join(output_dir, "fakeavceleb_random_delete_selected_real.txt")
    selected_fake_log = os.path.join(output_dir, "fakeavceleb_random_delete_selected_fake.txt")
    deleted_real_log = os.path.join(output_dir, "fakeavceleb_random_delete_deleted_real.txt")
    deleted_fake_log = os.path.join(output_dir, "fakeavceleb_random_delete_deleted_fake.txt")
    failed_real_log = os.path.join(output_dir, "fakeavceleb_random_delete_failed_real.txt")
    failed_fake_log = os.path.join(output_dir, "fakeavceleb_random_delete_failed_fake.txt")

    write_lines(selected_real_log, real_selected)
    write_lines(selected_fake_log, fake_selected)

    print(f"[Info] Dataset root         : {normalize_path_text(dataset_root)}")
    print(f"[Info] Real dir             : {normalize_path_text(real_dir)}")
    print(f"[Info] Fake dir             : {normalize_path_text(fake_dir)}")
    print(f"[Info] Total real images    : {len(real_paths)}")
    print(f"[Info] Total fake images    : {len(fake_paths)}")
    print(f"[Info] Selected real delete : {len(real_selected)}")
    print(f"[Info] Selected fake delete : {len(fake_selected)}")
    print(f"[Info] Selected real log    : {normalize_path_text(selected_real_log)}")
    print(f"[Info] Selected fake log    : {normalize_path_text(selected_fake_log)}")

    if not opt.execute:
        print("[Dry-Run] No file deleted.")
        return

    deleted_real, failed_real = delete_files(real_selected)
    deleted_fake, failed_fake = delete_files(fake_selected)

    write_lines(deleted_real_log, deleted_real)
    write_lines(deleted_fake_log, deleted_fake)
    write_lines(failed_real_log, failed_real)
    write_lines(failed_fake_log, failed_fake)

    print("[Done] Random deletion finished.")
    print(f"[Info] Deleted real         : {len(deleted_real)}")
    print(f"[Info] Deleted fake         : {len(deleted_fake)}")
    print(f"[Info] Failed real          : {len(failed_real)}")
    print(f"[Info] Failed fake          : {len(failed_fake)}")
    print(f"[Info] Deleted real log     : {normalize_path_text(deleted_real_log)}")
    print(f"[Info] Deleted fake log     : {normalize_path_text(deleted_fake_log)}")
    print(f"[Info] Failed real log      : {normalize_path_text(failed_real_log)}")
    print(f"[Info] Failed fake log      : {normalize_path_text(failed_fake_log)}")


if __name__ == "__main__":
    main()
