r"""
LAV-DF file selection + InsightFace preprocessing.

要求:
1. 文件选择逻辑与 preprocess-LAV-DF.py 保持一致
2. 预处理方法使用 preprocess-FakeAVCeleb-Insightface.py 的 InsightFace 流水线

输出结构:
OUTPUT_ROOT/
  TARGET_SPLIT/
    0_real/
      <sample_name>/
        frame_000000.jpg
        ...
        metadata.json
    1_fake/
      <sample_name>/
        frame_000000.jpg
        ...
        metadata.json

同时输出 selection_manifest.json，便于核对“选择文件是否一致”。
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import json
import math
import random
import shutil
import tempfile

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from tqdm import tqdm


# =====================================================================
# 可选: 注入 nvidia pip 运行时 DLL 目录
# =====================================================================
_DLL_DIR_HANDLES = []


def setup_nvidia_dll_path():
    try:
        import nvidia
    except Exception:
        return

    nvidia_root = os.path.dirname(nvidia.__file__)
    candidates = [
        os.path.join(nvidia_root, "cuda_runtime", "bin"),
        os.path.join(nvidia_root, "cuda_nvrtc", "bin"),
        os.path.join(nvidia_root, "cublas", "bin"),
        os.path.join(nvidia_root, "cudnn", "bin"),
        os.path.join(nvidia_root, "cufft", "bin"),
        os.path.join(nvidia_root, "curand", "bin"),
        os.path.join(nvidia_root, "nvjitlink", "bin"),
    ]

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []

    for dll_dir in candidates:
        if not os.path.isdir(dll_dir):
            continue

        if dll_dir not in path_parts:
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            path_parts.append(dll_dir)

        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))
            except Exception:
                pass


setup_nvidia_dll_path()


# =====================================================================
# 参数配置
# =====================================================================
N_EXTRACT = 10
WINDOW_LEN = 5
IMAGE_SIZE = 500
# 总预算约 8000 帧，可按需调整 real/fake 比例
MAX_REAL_IMAGES = 3000
MAX_FAKE_IMAGES = 5000
MAX_VIDEO = None
TARGET_SPLIT = "test"
RANDOM_SEED = 42

DATASET_ROOT = r"E:\data\LAV-DF"
OUTPUT_ROOT = r"E:\data\LAV-DF-Insightface"
METADATA_FILE = ""

INSIGHTFACE_MODEL = "buffalo_l"
INSIGHTFACE_DET_SIZE = (640, 640)
FRAME_SKIP = 1
# None 表示不对单视频做帧数截断，保证入选视频按完整片段提取
MAX_FRAMES_PER_VIDEO = None
FACE_DET_THRESHOLD = 0.2
SAVE_FORMAT = ".jpg"
VERBOSE = False

# True: 每个进入处理流程的任务必须完整提取；若预算不足则整段跳过
REQUIRE_FULL_TASK_WITHIN_BUDGET = True
# True: 逐帧保存（即使该帧未检测到有效人脸）
SAVE_ALL_FRAMES = True
# True: 运行前清空 TARGET_SPLIT 输出目录，避免历史结果干扰预算
CLEAN_SPLIT_OUTPUT = False


# =====================================================================
# 工具函数
# =====================================================================
def resolve_dataset_and_metadata(root_dir, metadata_path=""):
    root_dir = os.path.abspath(root_dir)
    candidates = []

    if metadata_path:
        explicit_meta = os.path.abspath(metadata_path)
        candidates.append((os.path.dirname(explicit_meta), explicit_meta))

    candidates.append((root_dir, os.path.join(root_dir, "metadata.min.json")))

    nested_root = os.path.join(root_dir, "LAV-DF")
    candidates.append((nested_root, os.path.join(nested_root, "metadata.min.json")))

    checked = []
    for data_root, meta_path in candidates:
        checked.append(meta_path)
        if os.path.isfile(meta_path):
            return data_root, meta_path

    checked_msg = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(f"metadata.min.json not found. Checked:\n{checked_msg}")


def safe_imwrite(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = SAVE_FORMAT

        ok, im_buf_arr = cv2.imencode(ext, img)
        if not ok:
            return False

        with open(file_path, "wb") as f:
            f.write(im_buf_arr.tobytes())
        return True
    except Exception:
        return False


class SafeVideoCapture:
    """兼容中文路径的视频读取器"""

    def __init__(self, path):
        self.path = path
        self.temp_path = None
        self.cap = None

        if not path.isascii():
            fd, self.temp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            shutil.copy2(path, self.temp_path)
            self.cap = cv2.VideoCapture(self.temp_path)
        else:
            self.cap = cv2.VideoCapture(path)

    def isOpened(self):
        return self.cap.isOpened() if self.cap else False

    def get(self, prop_id):
        return self.cap.get(prop_id) if self.cap else 0

    def read(self):
        return self.cap.read() if self.cap else (False, None)

    def release(self):
        if self.cap:
            self.cap.release()
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except Exception:
                pass


def count_existing_images(save_dir):
    if not os.path.isdir(save_dir):
        return 0

    count = 0
    for _, _, files in os.walk(save_dir):
        for f in files:
            if f.lower().endswith(SAVE_FORMAT):
                count += 1
    return count


# =====================================================================
# 选择逻辑 (严格参考 preprocess-LAV-DF.py)
# =====================================================================
def build_selection_tasks(meta, target_split):
    meta_dict = {item["file"]: item for item in meta}

    real_list = [
        item for item in meta
        if item.get("n_fakes") == 0 and item.get("split") == target_split
    ]
    fake_list = [
        item for item in meta
        if item.get("n_fakes") > 0 and item.get("split") == target_split
    ]

    random.seed(RANDOM_SEED)
    random.shuffle(real_list)
    random.shuffle(fake_list)

    if MAX_VIDEO is not None:
        real_list = real_list[:MAX_VIDEO]
        fake_list = fake_list[:MAX_VIDEO]

    fake_related_tasks = []

    for item in fake_list:
        fake_periods = item.get("fake_periods", [])
        fake_name = os.path.splitext(os.path.basename(item["file"]))[0]
        item_duration = float(item.get("duration", 0.0) or 0.0)

        for seg_id, period in enumerate(fake_periods):
            fake_related_tasks.append({
                "task_type": "fake_period",
                "label_dir": "1_fake",
                "label_val": 1,
                "source_file": item["file"],
                "sample_name": f"{fake_name}_seg{seg_id}",
                "period": period,
                "source_duration": item_duration,
                "pair_from": None,
            })

            try:
                t0, t1 = float(period[0]), float(period[1])
            except Exception:
                # 与原脚本一致：无效 period 的 fake 任务会被创建，执行阶段自然跳过
                continue

            original_key = item.get("original")
            if original_key is None or original_key not in meta_dict:
                continue

            real_item = meta_dict[original_key]
            real_file = real_item.get("file")
            real_duration = float(real_item.get("duration", 0.0) or 0.0)
            if not real_file:
                continue

            r0 = t0 / max(item_duration, 1e-8) * real_duration
            r1 = t1 / max(item_duration, 1e-8) * real_duration

            real_name = os.path.splitext(os.path.basename(real_file))[0]
            fake_related_tasks.append({
                "task_type": "real_pair",
                "label_dir": "0_real",
                "label_val": 0,
                "source_file": real_file,
                "sample_name": f"{real_name}_pair_{fake_name}_seg{seg_id}",
                "period": [r0, r1],
                "source_duration": real_duration,
                "pair_from": item["file"],
            })

    # 与原脚本一致: 额外准备 full real videos 用于补足 real 配额
    full_real_tasks = []
    for item in real_list:
        duration = float(item.get("duration", 0.0) or 0.0)

        name = os.path.splitext(os.path.basename(item["file"]))[0]
        full_real_tasks.append({
            "task_type": "real_full",
            "label_dir": "0_real",
            "label_val": 0,
            "source_file": item["file"],
            "sample_name": name,
            "period": [0.0, duration],
            "source_duration": duration,
            "pair_from": None,
        })

    stats = {
        "real_video_candidates": len(real_list),
        "fake_video_candidates": len(fake_list),
        "fake_related_tasks": len(fake_related_tasks),
        "full_real_tasks": len(full_real_tasks),
    }
    return fake_related_tasks, full_real_tasks, stats


def write_selection_manifest(output_root, data_root, metadata_path, fake_related_tasks, full_real_tasks, stats):
    manifest = {
        "dataset_root": data_root,
        "metadata_path": metadata_path,
        "target_split": TARGET_SPLIT,
        "random_seed": RANDOM_SEED,
        "max_video": MAX_VIDEO,
        "max_real_images": MAX_REAL_IMAGES,
        "max_fake_images": MAX_FAKE_IMAGES,
        "stats": stats,
        "fake_related_tasks": fake_related_tasks,
        "full_real_tasks": full_real_tasks,
    }

    path = os.path.join(output_root, TARGET_SPLIT, "selection_manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


# =====================================================================
# InsightFace 初始化与单任务处理
# =====================================================================
global_app = None


def init_worker():
    global global_app

    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)

    try:
        ort.preload_dlls()
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] preload_dlls failed: {e}")

    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
    except Exception:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    global_app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    global_app.prepare(ctx_id=ctx_id, det_size=INSIGHTFACE_DET_SIZE)

    device_name = "GPU (CUDA)" if ctx_id == 0 else "CPU"
    print(f"[InsightFace] 初始化完毕，运行设备: {device_name}")


def process_task(task, data_root, split_root, remaining_budget):
    """
    处理单个任务（fake period / real pair / real full）。

    返回:
        ok: bool
        num_saved: int
        status: str
    """
    if remaining_budget <= 0:
        return False, 0, "budget_exhausted"

    period = task.get("period", None)
    try:
        t0 = float(period[0])
        t1 = float(period[1])
    except Exception:
        return False, 0, "invalid_period"

    source_file = task["source_file"]
    video_path = os.path.join(data_root, source_file)
    if not os.path.exists(video_path):
        return False, 0, "source_missing"

    out_dir = os.path.join(split_root, task["label_dir"], task["sample_name"])
    json_path = os.path.join(out_dir, "metadata.json")

    if os.path.exists(json_path):
        existing = count_existing_images(out_dir)
        return True, existing, "skipped_existing"

    os.makedirs(out_dir, exist_ok=True)

    cap = SafeVideoCapture(video_path)
    if not cap.isOpened():
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, "open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, "invalid_video_info"

    start_frame = max(0, int(math.floor(t0 * fps)))
    end_frame = min(total_frames, int(math.ceil(t1 * fps)))

    if end_frame <= start_frame:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, "empty_segment"

    segment_target_frames = max(0, (end_frame - start_frame + FRAME_SKIP - 1) // FRAME_SKIP)
    if REQUIRE_FULL_TASK_WITHIN_BUDGET and segment_target_frames > remaining_budget:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, "segment_exceeds_budget"

    metadata = {
        "video_info": {
            "video_name": os.path.basename(source_file),
            "source_file": source_file,
            "label": task["label_val"],
            "fps": round(float(fps), 4),
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
        "task_info": {
            "task_type": task["task_type"],
            "sample_name": task["sample_name"],
            "pair_from": task["pair_from"],
        },
        "segment_info": {
            "start_sec": t0,
            "end_sec": t1,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "frames": [],
    }

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= end_frame:
            break

        if frame_idx < start_frame:
            frame_idx += 1
            continue

        if MAX_FRAMES_PER_VIDEO is not None and saved_count >= MAX_FRAMES_PER_VIDEO:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        face_payload = None
        faces = global_app.get(frame)
        if faces:
            valid_faces = [
                f for f in faces
                if getattr(f, "det_score", 0) >= FACE_DET_THRESHOLD
            ]
            if valid_faces:
                best_face = max(
                    valid_faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )

                kps_106_obj = getattr(best_face, "landmark_2d_106", None)
                if kps_106_obj is not None:
                    kps_5 = best_face.kps.tolist()
                    kps_106 = kps_106_obj.tolist()
                    bbox = best_face.bbox.tolist()
                    det_score = float(best_face.det_score)
                    face_payload = {
                        "bbox": [round(v, 2) for v in bbox],
                        "det_score": round(det_score, 4),
                        "kps_5": [[round(x, 2), round(y, 2)] for x, y in kps_5],
                        "kps_106": [[round(x, 2), round(y, 2)] for x, y in kps_106],
                    }

        if not SAVE_ALL_FRAMES and face_payload is None:
            frame_idx += 1
            continue

        img_name = f"frame_{frame_idx:06d}{SAVE_FORMAT}"
        img_path = os.path.join(out_dir, img_name)

        if not safe_imwrite(img_path, frame):
            frame_idx += 1
            continue

        metadata["frames"].append({
            "file_path": img_name,
            "frame_idx": frame_idx,
            "face": face_payload,
        })
        saved_count += 1
        frame_idx += 1

    cap.release()

    if metadata["frames"]:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True, saved_count, "ok"
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)
            return False, 0, "json_write_failed"

    shutil.rmtree(out_dir, ignore_errors=True)
    return False, 0, "no_valid_frames"


# =====================================================================
# 主流程
# =====================================================================
def run():
    data_root, meta_path = resolve_dataset_and_metadata(DATASET_ROOT, METADATA_FILE)

    split_root = os.path.join(OUTPUT_ROOT, TARGET_SPLIT)
    if CLEAN_SPLIT_OUTPUT and os.path.isdir(split_root):
        shutil.rmtree(split_root, ignore_errors=True)

    real_out_dir = os.path.join(split_root, "0_real")
    fake_out_dir = os.path.join(split_root, "1_fake")
    os.makedirs(real_out_dir, exist_ok=True)
    os.makedirs(fake_out_dir, exist_ok=True)

    print("=" * 72)
    print("LAV-DF 选择逻辑 + InsightFace 预处理")
    print(f"[配置] data_root:   {data_root}")
    print(f"[配置] metadata:    {meta_path}")
    print(f"[配置] output_root: {split_root}")
    print(f"[配置] split:       {TARGET_SPLIT}")
    print("=" * 72)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fake_related_tasks, full_real_tasks, stats = build_selection_tasks(meta, TARGET_SPLIT)
    manifest_path = write_selection_manifest(
        OUTPUT_ROOT,
        data_root,
        meta_path,
        fake_related_tasks,
        full_real_tasks,
        stats,
    )

    print(f"[选择] fake_related_tasks: {len(fake_related_tasks)}")
    print(f"[选择] full_real_tasks:    {len(full_real_tasks)}")
    print(f"[选择] manifest:           {manifest_path}")

    saved_real = count_existing_images(real_out_dir)
    saved_fake = count_existing_images(fake_out_dir)
    print(f"[已存在] real frames: {saved_real}/{MAX_REAL_IMAGES}")
    print(f"[已存在] fake frames: {saved_fake}/{MAX_FAKE_IMAGES}")

    init_worker()

    success = 0
    fail = 0
    skipped_budget = 0

    print("\n[阶段1] 处理 fake_period + real_pair ...")
    for task in tqdm(fake_related_tasks, desc="fake/pair tasks"):
        if task["label_val"] == 1:
            remaining = MAX_FAKE_IMAGES - saved_fake
        else:
            remaining = MAX_REAL_IMAGES - saved_real

        if remaining <= 0:
            continue

        ok, num_saved, status = process_task(task, data_root, split_root, remaining)
        if ok:
            success += 1
            if task["label_val"] == 1:
                saved_fake += num_saved
            else:
                saved_real += num_saved
        else:
            if status in {"budget_exhausted", "segment_exceeds_budget"}:
                skipped_budget += 1
            else:
                fail += 1

    print("\n[阶段2] 处理 full_real (仅用于补足 real 配额) ...")
    for task in tqdm(full_real_tasks, desc="full real tasks"):
        remaining = MAX_REAL_IMAGES - saved_real
        if remaining <= 0:
            break

        ok, num_saved, status = process_task(task, data_root, split_root, remaining)
        if ok:
            success += 1
            saved_real += num_saved
        else:
            if status in {"budget_exhausted", "segment_exceeds_budget"}:
                skipped_budget += 1
            else:
                fail += 1

    print("\n" + "=" * 72)
    print("处理完成")
    print(f"[结果] success_tasks: {success}")
    print(f"[结果] fail_tasks:    {fail}")
    print(f"[结果] skip_budget:   {skipped_budget}")
    print(f"[结果] real frames:   {saved_real}/{MAX_REAL_IMAGES}")
    print(f"[结果] fake frames:   {saved_fake}/{MAX_FAKE_IMAGES}")
    print(f"[输出] {split_root}")
    print("=" * 72)


if __name__ == "__main__":
    run()
