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
from models.offline_paths import insightface_root

import json
import math
import random
import shutil
import tempfile
import threading
import queue
import time
import warnings

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
# 以下常量来自原始 preprocess-LAV-DF.py，当前版本改用 FRAME_SKIP/segment 来控制采样。
# 保留仅供参考，但不再参与运行（避免误导维护者）。
# N_EXTRACT = 10
# WINDOW_LEN = 5
# IMAGE_SIZE = 500
# 公平评测默认：real/fake 预算对称，避免类别预算偏置。
# 若目标是“每次运行最终各 2000 帧”，请保持 CLEAN_SPLIT_OUTPUT=True。
MAX_REAL_IMAGES = 2000
MAX_FAKE_IMAGES = 2000
MAX_VIDEO = None
TARGET_SPLIT = "test"
RANDOM_SEED = 42

DATASET_ROOT = r"E:\data\LAV-DF"
OUTPUT_ROOT = r"E:\data\LAV-DF-test-pre"
METADATA_FILE = ""

INSIGHTFACE_MODEL = "buffalo_l"
INSIGHTFACE_DET_SIZE = (640, 640)
# 仅加载当前脚本需要的模块（检测 + 106 点），可显著提升单帧吞吐；输出标签与选帧逻辑不变。
INSIGHTFACE_ALLOWED_MODULES = ("detection", "landmark_2d_106")
FRAME_SKIP = 1
# None 表示不对单视频做帧数截断，保证入选视频按完整片段提取
MAX_FRAMES_PER_VIDEO = None
FACE_DET_THRESHOLD = 0.2
SAVE_FORMAT = ".jpg"
VERBOSE = False
# 屏蔽 InsightFace 依赖链上的 FutureWarning，避免频繁刷屏影响观感。
SUPPRESS_INSIGHTFACE_FUTURE_WARNINGS = True

# True: 每个进入处理流程的任务必须完整提取；若预算不足则整段跳过
REQUIRE_FULL_TASK_WITHIN_BUDGET = True
# True: 逐帧保存（即使该帧未检测到有效人脸）
SAVE_ALL_FRAMES = True
# True: 运行前清空 TARGET_SPLIT 输出目录，避免历史结果干扰预算
# 设为 True 可保证本次运行按预算从 0 开始计数（例如各 2000 帧）。
CLEAN_SPLIT_OUTPUT = True
# True: 忽略已有结果并重建对应 sample 目录
FORCE_REPROCESS = False
# 连续跳过阈值：当连续 N 个任务都没有新增帧时，认为后面大概率也不会再产出，提前退出（防止扫完全部任务）
CONSECUTIVE_SKIP_THRESHOLD = 1000
# True: 视觉任务时忽略“音频伪造但视频真实”(RF)，仅保留 RR/FR/FF 三种情况
EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL = True

# 公平评测模式：
# 1) 取消“短片段优先”以避免预算下的长度偏置；
# 2) 为每个 sample 施加最小帧数门槛（默认 32），不足时扩窗，仍不足则跳过。
FAIR_EVAL_MODE = True
SORT_FAKE_TASKS_BY_DURATION = False
MIN_SAVED_FRAMES_PER_SAMPLE = 32
EXPAND_SHORT_SEGMENTS_FOR_MIN_FRAMES = True
SKIP_IF_MIN_FRAMES_UNREACHABLE = True

# 流水线队列容量：reader 解码队列和 writer 写盘队列。
READ_QUEUE_MAXSIZE = 16
WRITE_QUEUE_MAXSIZE = 32
_QUEUE_SENTINEL = object()
FPS_POSTFIX_REFRESH_SEC = 0.5


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
    """
    零拷贝（Zero-Copy）图像落盘函数。

    修改原因：原代码 `im_buf_arr.tobytes()` 会在内存中申请与图像等大的新连续空间，
    这在多线程高频调用时会引发灾难性的 Python GC（垃圾回收）停顿。
    改用 memoryview 暴露底层 numpy C array 的 buffer 接口，直接与操作系统的 IO API 交互，
    实现无感落盘，极大降低 CPU 开销。
    """
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = SAVE_FORMAT

        # im_buf_arr 是一个一维的 numpy 数组，包含编码后的二进制数据
        ok, im_buf_arr = cv2.imencode(ext, img)
        if not ok:
            return False

        with open(file_path, "wb") as f:
            # 性能核心：通过 memoryview 直接投递内存指针，拒绝 Python 层的硬拷贝
            f.write(memoryview(im_buf_arr))
        return True
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] safe_imwrite failed: {e}")
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

    def set(self, prop_id, value):
        return self.cap.set(prop_id, value) if self.cap else False

    def release(self):
        if self.cap:
            self.cap.release()
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except Exception:
                pass


def count_existing_images(save_dir):
    """Count image files in a sample directory using single-level scandir."""
    if not os.path.isdir(save_dir):
        return 0

    suffix = SAVE_FORMAT.lower()
    try:
        return sum(
            1
            for entry in os.scandir(save_dir)
            if entry.is_file() and entry.name.lower().endswith(suffix)
        )
    except OSError:
        return 0


def count_existing_images_recursive(root_dir):
    """Count image files recursively under a label root directory."""
    if not os.path.isdir(root_dir):
        return 0

    suffix = SAVE_FORMAT.lower()
    total = 0
    try:
        for cur_root, _, file_names in os.walk(root_dir):
            total += sum(1 for name in file_names if name.lower().endswith(suffix))
        return total
    except OSError:
        return 0


def is_valid_processed_output(save_dir, json_path):
    """判断现有输出是否完整，避免残缺目录被误判为可跳过。"""
    if not (os.path.isdir(save_dir) and os.path.isfile(json_path)):
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    frames = meta.get("frames", [])
    if not frames:
        return False

    for item in frames:
        file_name = item.get("file_path")
        if not file_name:
            return False
        frame_path = os.path.join(save_dir, file_name)
        if not os.path.isfile(frame_path):
            return False

    return True


def format_budget(saved, limit):
    if limit is None:
        return f"{saved}/unlimited"
    return f"{saved}/{limit}"


def remaining_budget(saved, limit):
    if limit is None:
        return float("inf")
    return limit - saved


class FrameRateMeter:
    """Track write throughput using actually written frames."""

    def __init__(self, refresh_interval=0.5, smooth_alpha=0.25):
        self.start_time = time.perf_counter()
        self.last_update_time = self.start_time
        self.last_postfix_time = 0.0
        self.last_frames = 0
        self.total_frames = 0
        self.ema_fps = 0.0
        self.refresh_interval = refresh_interval
        self.smooth_alpha = smooth_alpha

    def add_frames(self, n_frames):
        if n_frames > 0:
            self.total_frames += int(n_frames)

    def _snapshot(self):
        now = time.perf_counter()
        elapsed = max(1e-9, now - self.start_time)
        avg_fps = self.total_frames / elapsed

        delta_t = max(1e-9, now - self.last_update_time)
        delta_frames = self.total_frames - self.last_frames
        inst_fps = max(0.0, delta_frames / delta_t)

        if self.ema_fps <= 0:
            self.ema_fps = inst_fps
        else:
            a = min(max(float(self.smooth_alpha), 0.0), 1.0)
            self.ema_fps = a * inst_fps + (1.0 - a) * self.ema_fps

        self.last_update_time = now
        self.last_frames = self.total_frames
        return now, self.ema_fps, avg_fps

    def update_tqdm(self, pbar, saved_real, saved_fake, remaining_real, remaining_fake, force=False):
        now = time.perf_counter()
        if (not force) and (now - self.last_postfix_time < self.refresh_interval):
            return

        _, inst_fps, avg_fps = self._snapshot()
        total_saved = int(saved_real) + int(saved_fake)
        rem_real = "unlimited" if remaining_real is None else str(max(0, int(remaining_real)))
        rem_fake = "unlimited" if remaining_fake is None else str(max(0, int(remaining_fake)))
        pbar.set_postfix_str(
            f"write_fps={inst_fps:.2f} avg_fps={avg_fps:.2f} written_run={self.total_frames} total_saved={total_saved} rem_real={rem_real} rem_fake={rem_fake}",
            refresh=False,
        )
        self.last_postfix_time = now


def sampled_frames_between(start_frame, end_frame, frame_skip):
    span = max(0, int(end_frame) - int(start_frame))
    if span <= 0:
        return 0
    return (span + frame_skip - 1) // frame_skip


def expand_segment_for_min_frames(start_frame, end_frame, total_frames, min_frames, frame_skip):
    """Symmetrically expand segment to satisfy minimum sampled frame count when possible."""
    if min_frames <= 0:
        return start_frame, end_frame

    min_span = (min_frames - 1) * frame_skip + 1
    cur_span = end_frame - start_frame
    if cur_span >= min_span:
        return start_frame, end_frame

    need = min_span - cur_span
    left = need // 2
    right = need - left

    new_start = max(0, start_frame - left)
    new_end = min(total_frames, end_frame + right)

    cur_span = new_end - new_start
    if cur_span < min_span and new_start > 0:
        grow_left = min(min_span - cur_span, new_start)
        new_start -= grow_left
        cur_span = new_end - new_start

    if cur_span < min_span and new_end < total_frames:
        grow_right = min(min_span - cur_span, total_frames - new_end)
        new_end += grow_right

    return new_start, new_end


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _lavdf_type(item):
    """RR: real/real, RF: real video + fake audio, FR: fake video + real audio, FF: fake/fake."""
    modify_video = _to_bool(item.get("modify_video", False))
    modify_audio = _to_bool(item.get("modify_audio", False))

    if (not modify_video) and (not modify_audio):
        return "RR"
    if (not modify_video) and modify_audio:
        return "RF"
    if modify_video and (not modify_audio):
        return "FR"
    return "FF"


def _build_face_payload(face):
    """
    从 InsightFace 人脸对象构造 JSON-safe 的 face 字段。
    总是返回字典（若部分关键点缺失则填 None），便于下游统一处理。
    """
    try:
        bbox = face.bbox.tolist()
    except Exception:
        bbox = [0, 0, 0, 0]

    det_score = float(getattr(face, "det_score", 0.0))

    payload = {
        "bbox": [round(v, 2) for v in bbox],
        "det_score": round(det_score, 4),
        "kps_5": None,
        "kps_106": None,
    }

    kps_5_obj = getattr(face, "kps", None)
    if kps_5_obj is not None:
        try:
            kps_5 = kps_5_obj.tolist()
            payload["kps_5"] = [[round(x, 2), round(y, 2)] for x, y in kps_5]
        except Exception:
            payload["kps_5"] = None

    kps_106_obj = getattr(face, "landmark_2d_106", None)
    if kps_106_obj is not None:
        try:
            kps_106 = kps_106_obj.tolist()
            payload["kps_106"] = [[round(x, 2), round(y, 2)] for x, y in kps_106]
        except Exception:
            payload["kps_106"] = None

    return payload


# =====================================================================
# 选择逻辑 (严格参考 preprocess-LAV-DF.py)
# =====================================================================
def build_selection_tasks(meta, target_split):
    meta_dict = {item["file"]: item for item in meta}

    split_items = [item for item in meta if item.get("split") == target_split]
    type_counts_all = {"RR": 0, "RF": 0, "FR": 0, "FF": 0}
    type_counts_used = {"RR": 0, "RF": 0, "FR": 0, "FF": 0}
    excluded_audio_only_fake = 0

    real_list = []
    fake_list = []

    for item in split_items:
        item_type = _lavdf_type(item)
        type_counts_all[item_type] += 1

        # 视觉任务：忽略 RF（视频真实，仅音频伪造）
        if EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL and item_type == "RF":
            excluded_audio_only_fake += 1
            continue

        # 视觉二分类：只有视频被修改(FR/FF)才计入 fake
        if item_type in {"FR", "FF"}:
            fake_list.append(item)
        else:
            real_list.append(item)

        type_counts_used[item_type] += 1

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
        "label_policy": "visual",
        "exclude_audio_only_fake_in_visual": bool(EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL),
        "excluded_audio_only_fake_count": int(excluded_audio_only_fake),
        "type_counts_all_split": type_counts_all,
        "type_counts_used_after_filter": type_counts_used,
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


def export_processed_clip_list(split_root):
    """Export final processed samples into a unified clip index for downstream benchmarking."""
    rows = []
    label_map = {"0_real": 0, "1_fake": 1}

    for label_dir, label_val in label_map.items():
        root_dir = os.path.join(split_root, label_dir)
        if not os.path.isdir(root_dir):
            continue

        for sample_name in os.listdir(root_dir):
            sample_dir = os.path.join(root_dir, sample_name)
            if not os.path.isdir(sample_dir):
                continue

            json_path = os.path.join(sample_dir, "metadata.json")
            if not os.path.isfile(json_path):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            frames = meta.get("frames", [])
            if not frames:
                continue

            task_info = meta.get("task_info", {})
            seg_info = meta.get("segment_info", {})
            video_info = meta.get("video_info", {})

            rows.append({
                "sample_name": sample_name,
                "label": label_val,
                "label_dir": label_dir,
                "source_file": video_info.get("source_file", ""),
                "task_type": task_info.get("task_type", ""),
                "pair_from": task_info.get("pair_from", ""),
                "start_sec": seg_info.get("start_sec", ""),
                "end_sec": seg_info.get("end_sec", ""),
                "num_frames": len(frames),
                "fps": video_info.get("fps", ""),
            })

    rows.sort(key=lambda x: (x["label"], x["sample_name"]))

    out_path = os.path.join(split_root, "processed_clip_list.tsv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sample_name\tlabel\tlabel_dir\tsource_file\ttask_type\tpair_from\tstart_sec\tend_sec\tnum_frames\tfps\n")
        for row in rows:
            f.write(
                f"{row['sample_name']}\t{row['label']}\t{row['label_dir']}\t{row['source_file']}\t"
                f"{row['task_type']}\t{row['pair_from']}\t{row['start_sec']}\t{row['end_sec']}\t"
                f"{row['num_frames']}\t{row['fps']}\n"
            )

    return out_path, len(rows)


# =====================================================================
# InsightFace 初始化与单任务处理
# =====================================================================
global_app = None


def init_worker():
    """
    全局 InsightFace 工作流初始化。

    修改原因：打破 ORT (ONNXRuntime) 的保守默认黑盒配置。
    通过 Provider 字典注入底层算子优化选项：
    1. 启用 cuDNN 穷举搜索最优卷积核 (EXHAUSTIVE)。
    2. 开启内存池策略 (kSameAsRequested) 避免显存碎片化。
    这些改动不会影响浮点计算结果，但能拉升 10%~20% 的 GPU 吞吐量。
    """
    global global_app

    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)

    if SUPPRESS_INSIGHTFACE_FUTURE_WARNINGS:
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"`rcond` parameter will change.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"`estimate` is deprecated.*",
        )

    try:
        ort.preload_dlls()
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] preload_dlls failed: {e}")

    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            # 构造深度优化配置的 Providers 列表
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",  # 减少显存动态伸缩开销
                "cudnn_conv_algo_search": "EXHAUSTIVE",       # 首次推理时寻找最优 cuDNN 算法
                "do_copy_in_default_stream": True,              # 优化 CPU/GPU 内存拷贝流
            }
            providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
            ctx_id = 0
        else:
            # CPU 兜底配置：限制内部线程池创建，避免与当前的多线程 Python 抢夺 CPU 核心
            cpu_options = {
                "arena_extend_strategy": "kSameAsRequested"
            }
            providers = [("CPUExecutionProvider", cpu_options)]
            ctx_id = -1
    except Exception:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    face_analysis_kwargs = {
        "name": INSIGHTFACE_MODEL,
        "root": insightface_root(),
        "providers": providers,
    }
    if INSIGHTFACE_ALLOWED_MODULES:
        face_analysis_kwargs["allowed_modules"] = list(INSIGHTFACE_ALLOWED_MODULES)

    global_app = FaceAnalysis(**face_analysis_kwargs)
    global_app.prepare(ctx_id=ctx_id, det_size=INSIGHTFACE_DET_SIZE)

    device_name = "GPU (CUDA)" if ctx_id == 0 else "CPU"
    print(f"[InsightFace] 初始化完毕，已注入底层硬件加速参数，运行设备: {device_name}")


def _frame_reader_thread(cap, start_frame, end_frame, frame_skip, read_queue, stop_event):
    """
    带有智能跳帧（Smart Frame Skipping）与生命周期管理的视频解码线程。

    修改原因：拦截因 FRAME_SKIP > 1 导致的冗余 H.264/YUV->BGR 解码开销。
    利用 cap.grab() 仅解析容器报文推进指针，仅在真正需要时执行 retrieve() 像素解码。
    """
    frame_idx = start_frame
    try:
        while frame_idx < end_frame and not stop_event.is_set():
            # 命中采样周期，执行完整解码
            if (frame_idx - start_frame) % frame_skip == 0:
                ret, frame = cap.read()  # read() 本质是 grab() + retrieve()
                if not ret:
                    break

                while not stop_event.is_set():
                    try:
                        read_queue.put((frame_idx, frame), timeout=1.0)
                        break
                    except queue.Full:
                        continue
            else:
                # 未命中采样周期，仅抓取下一帧的网络抽象层单元(NALU)，跳过像素级解压
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1
    except Exception as e:
        print(f"[ERROR] 视频解码线程异常: {e}")
    finally:
        while True:
            try:
                read_queue.put(_QUEUE_SENTINEL, timeout=2.0)
                break
            except queue.Full:
                if stop_event.is_set():
                    continue
                continue


def _frame_writer_thread(write_queue, out_dir, metadata_frames):
    """Write frames to disk and append corresponding metadata entries."""
    while True:
        item = write_queue.get()
        if item is _QUEUE_SENTINEL:
            break

        try:
            img_name, frame_idx, frame, face_payload = item
            img_path = os.path.join(out_dir, img_name)
            if safe_imwrite(img_path, frame):
                metadata_frames.append({
                    "file_path": img_name,
                    "frame_idx": frame_idx,
                    "face": face_payload,
                })
        except Exception as e:
            if VERBOSE:
                print(f"[WARN] writer thread failed for frame item: {e}")


def process_task(task, data_root, split_root, budget_remaining):
    """
    处理单个任务（fake period / real pair / real full）。

    返回:
        ok: bool
        budget_delta: int
        written_frames: int
        status: str
    """
    if budget_remaining <= 0:
        return False, 0, 0, "budget_exhausted"

    period = task.get("period", None)
    try:
        t0 = float(period[0])
        t1 = float(period[1])
    except Exception:
        return False, 0, 0, "invalid_period"

    source_file = task["source_file"]
    video_path = source_file if os.path.isabs(source_file) else os.path.join(data_root, source_file)
    if not os.path.exists(video_path):
        return False, 0, 0, "source_missing"

    out_dir = os.path.join(split_root, task["label_dir"], task["sample_name"])
    json_path = os.path.join(out_dir, "metadata.json")

    # 记录旧产物数量，避免 FORCE_REPROCESS 时导致主流程计数双重累计。
    old_count = 0
    if os.path.isdir(out_dir):
        old_count = count_existing_images(out_dir)

    if (not FORCE_REPROCESS) and is_valid_processed_output(out_dir, json_path):
        return True, 0, 0, "skipped_existing"

    # 移除旧目录（若存在），准备重建
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)

    cap = SafeVideoCapture(video_path)
    if not cap.isOpened():
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, 0, "open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, 0, "invalid_video_info"

    start_frame = max(0, int(math.floor(t0 * fps)))
    end_frame = min(total_frames, int(math.ceil(t1 * fps)))

    if end_frame <= start_frame:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, 0, "empty_segment"

    # 公平评测约束：短片段优先扩窗到最小帧数，仍不足则跳过。
    if FAIR_EVAL_MODE and MIN_SAVED_FRAMES_PER_SAMPLE > 0:
        seg_frames = sampled_frames_between(start_frame, end_frame, FRAME_SKIP)
        if seg_frames < MIN_SAVED_FRAMES_PER_SAMPLE and EXPAND_SHORT_SEGMENTS_FOR_MIN_FRAMES:
            start_frame, end_frame = expand_segment_for_min_frames(
                start_frame,
                end_frame,
                total_frames,
                MIN_SAVED_FRAMES_PER_SAMPLE,
                FRAME_SKIP,
            )

        seg_frames = sampled_frames_between(start_frame, end_frame, FRAME_SKIP)
        if seg_frames < MIN_SAVED_FRAMES_PER_SAMPLE and SKIP_IF_MIN_FRAMES_UNREACHABLE:
            cap.release()
            shutil.rmtree(out_dir, ignore_errors=True)
            return False, 0, 0, "segment_too_short_for_min_frames"

    segment_target_frames = max(0, (end_frame - start_frame + FRAME_SKIP - 1) // FRAME_SKIP)
    if REQUIRE_FULL_TASK_WITHIN_BUDGET and segment_target_frames > budget_remaining:
        cap.release()
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, 0, "segment_exceeds_budget"

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

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    read_queue = queue.Queue(maxsize=READ_QUEUE_MAXSIZE)
    write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
    stop_event = threading.Event()

    reader_thread = threading.Thread(
        target=_frame_reader_thread,
        args=(cap, start_frame, end_frame, FRAME_SKIP, read_queue, stop_event),
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=_frame_writer_thread,
        args=(write_queue, out_dir, metadata["frames"]),
        daemon=True,
    )
    reader_thread.start()
    writer_thread.start()

    queued_count = 0

    # Main thread: consume decoded frames, run face inference, and enqueue write jobs.
    while True:
        item = read_queue.get()
        if item is _QUEUE_SENTINEL:
            break

        frame_idx, frame = item

        # Keep draining the queue even after reaching frame limit to avoid blocking reader thread.
        if MAX_FRAMES_PER_VIDEO is not None and queued_count >= MAX_FRAMES_PER_VIDEO:
            continue

        face_payload = None
        try:
            faces = global_app.get(frame)
            if faces:
                best_face = None
                best_area = -1.0
                for f in faces:
                    if getattr(f, "det_score", 0) < FACE_DET_THRESHOLD:
                        continue
                    area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    if area > best_area:
                        best_area = area
                        best_face = f

                if best_face is not None:
                    face_payload = _build_face_payload(best_face)
        except Exception:
            # 检测异常时继续处理（若 SAVE_ALL_FRAMES=True 则仍保存帧）
            face_payload = None

        if not SAVE_ALL_FRAMES and face_payload is None:
            continue

        img_name = f"frame_{frame_idx:06d}{SAVE_FORMAT}"
        write_queue.put((img_name, frame_idx, frame, face_payload))
        queued_count += 1

    stop_event.set()
    write_queue.put(_QUEUE_SENTINEL)
    writer_thread.join()
    reader_thread.join()
    cap.release()

    saved_count = len(metadata["frames"])

    if FAIR_EVAL_MODE and MIN_SAVED_FRAMES_PER_SAMPLE > 0 and saved_count < MIN_SAVED_FRAMES_PER_SAMPLE:
        shutil.rmtree(out_dir, ignore_errors=True)
        return False, 0, 0, "saved_frames_below_min"

    if metadata["frames"]:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            # 返回净增量，便于 run() 层正确维护已保存计数（避免双重计数）
            return True, max(0, saved_count - old_count), saved_count, "ok"
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)
            return False, 0, 0, "json_write_failed"

    shutil.rmtree(out_dir, ignore_errors=True)
    return False, 0, 0, "no_valid_frames"


# =====================================================================
# 主流程
# =====================================================================
def run():
    def _remaining_for_postfix(saved, limit):
        rem = remaining_budget(saved, limit)
        if math.isinf(rem):
            return None
        return max(0, int(rem))

    data_root, meta_path = resolve_dataset_and_metadata(DATASET_ROOT, METADATA_FILE)

    if FAIR_EVAL_MODE and (MAX_REAL_IMAGES is not None and MAX_FAKE_IMAGES is not None):
        if MAX_REAL_IMAGES != MAX_FAKE_IMAGES:
            raise ValueError("FAIR_EVAL_MODE requires symmetric class budgets: MAX_REAL_IMAGES must equal MAX_FAKE_IMAGES")

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
    print(f"[配置] 视觉任务过滤 RF(audio-only fake): {EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL}")
    print(f"[配置] fair_eval_mode: {FAIR_EVAL_MODE}")
    print(f"[配置] min_saved_frames_per_sample: {MIN_SAVED_FRAMES_PER_SAMPLE}")
    print(f"[配置] expand_short_segments: {EXPAND_SHORT_SEGMENTS_FOR_MIN_FRAMES}")
    print("=" * 72)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fake_related_tasks, full_real_tasks, stats = build_selection_tasks(meta, TARGET_SPLIT)

    # 预过滤：移除无效 period 的任务（避免执行大量无效任务）
    def _task_has_valid_period(task):
        p = task.get("period")
        if not p or len(p) < 2:
            return False
        try:
            t0 = float(p[0]); t1 = float(p[1])
            return t1 > t0
        except Exception:
            return False

    fake_related_tasks = [t for t in fake_related_tasks if _task_has_valid_period(t)]

    if SORT_FAKE_TASKS_BY_DURATION:
        # 可选策略：按片段长度排序。公平评测模式下建议关闭，避免短段偏置。
        try:
            fake_related_tasks.sort(key=lambda t: float(t.get("period", [0, float("inf")])[1]) - float(t.get("period", [0, 0])[0]))
        except Exception:
            pass

    stats["fake_related_tasks"] = len(fake_related_tasks)

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
    print(f"[选择] 类型计数(全量):      {stats.get('type_counts_all_split', {})}")
    print(f"[选择] 类型计数(入选):      {stats.get('type_counts_used_after_filter', {})}")
    print(f"[选择] 跳过 RF 数量:        {stats.get('excluded_audio_only_fake_count', 0)}")
    print(f"[选择] manifest:           {manifest_path}")

    saved_real = count_existing_images_recursive(real_out_dir)
    saved_fake = count_existing_images_recursive(fake_out_dir)
    print(f"[已存在] real frames: {format_budget(saved_real, MAX_REAL_IMAGES)}")
    print(f"[已存在] fake frames: {format_budget(saved_fake, MAX_FAKE_IMAGES)}")
    if not CLEAN_SPLIT_OUTPUT and (saved_real > 0 or saved_fake > 0):
        print("[提示] CLEAN_SPLIT_OUTPUT=False，当前预算基于历史结果继续累计，而非从 0 开始")

    rem_real_init = _remaining_for_postfix(saved_real, MAX_REAL_IMAGES)
    rem_fake_init = _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES)
    if rem_real_init is None or rem_fake_init is None:
        frame_budget_total = None
        print("[进度] 帧预算含 unlimited，使用无上限帧进度条")
    else:
        frame_budget_total = int(rem_real_init + rem_fake_init)
        print(
            f"[进度] 帧预算倒计时: total={frame_budget_total} "
            f"(real={rem_real_init}, fake={rem_fake_init})"
        )

    init_worker()

    success = 0
    fail = 0
    skipped_budget = 0
    skipped_min_frames = 0
    skipped_existing = 0
    perf_meter = FrameRateMeter(refresh_interval=FPS_POSTFIX_REFRESH_SEC)
    run_written_frames = 0

    with tqdm(total=frame_budget_total, desc="frame budget", unit="frame") as frame_pbar:
        perf_meter.update_tqdm(
            frame_pbar,
            saved_real,
            saved_fake,
            _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
            _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
            force=True,
        )

        print("\n[阶段1] 处理 fake_period + real_pair ...")
        # 连续未新增帧计数器，用于提前退出（防止遍历所有任务）
        consecutive_no_new = 0
        for task in fake_related_tasks:
            # 若两个配额均耗尽，则直接终止遍历，避免扫描剩余大量任务
            if remaining_budget(saved_fake, MAX_FAKE_IMAGES) <= 0 and remaining_budget(saved_real, MAX_REAL_IMAGES) <= 0:
                break

            if task["label_val"] == 1:
                remaining = remaining_budget(saved_fake, MAX_FAKE_IMAGES)
            else:
                remaining = remaining_budget(saved_real, MAX_REAL_IMAGES)

            if remaining <= 0:
                consecutive_no_new += 1
                # 若连续大量任务都无法消耗预算，则提前退出
                if consecutive_no_new >= CONSECUTIVE_SKIP_THRESHOLD:
                    print(f"\n[INFO] 连续 {CONSECUTIVE_SKIP_THRESHOLD} 个任务未新增帧，提前退出 fake/pair 阶段")
                    break
                perf_meter.update_tqdm(
                    frame_pbar,
                    saved_real,
                    saved_fake,
                    _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
                    _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
                )
                continue

            ok, num_saved, written_frames, status = process_task(task, data_root, split_root, remaining)
            perf_meter.add_frames(written_frames)
            run_written_frames += written_frames

            if ok:
                success += 1
                if status == "skipped_existing":
                    skipped_existing += 1
                    consecutive_no_new += 1
                else:
                    if task["label_val"] == 1:
                        saved_fake += num_saved
                    else:
                        saved_real += num_saved

                    if num_saved > 0:
                        frame_pbar.update(num_saved)
                        consecutive_no_new = 0
                    else:
                        consecutive_no_new += 1
            else:
                if status in {"budget_exhausted", "segment_exceeds_budget"}:
                    skipped_budget += 1
                    consecutive_no_new += 1
                elif status in {"segment_too_short_for_min_frames", "saved_frames_below_min"}:
                    skipped_min_frames += 1
                    consecutive_no_new += 1
                else:
                    fail += 1
                    consecutive_no_new += 1

            perf_meter.update_tqdm(
                frame_pbar,
                saved_real,
                saved_fake,
                _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
                _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
            )

        perf_meter.update_tqdm(
            frame_pbar,
            saved_real,
            saved_fake,
            _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
            _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
            force=True,
        )

        print("\n[阶段2] 处理 full_real (仅用于补足 real 配额) ...")
        for task in full_real_tasks:
            remaining = remaining_budget(saved_real, MAX_REAL_IMAGES)
            if remaining <= 0:
                break

            ok, num_saved, written_frames, status = process_task(task, data_root, split_root, remaining)
            perf_meter.add_frames(written_frames)
            run_written_frames += written_frames

            if ok:
                success += 1
                if status == "skipped_existing":
                    skipped_existing += 1
                else:
                    saved_real += num_saved
                    if num_saved > 0:
                        frame_pbar.update(num_saved)
            else:
                if status in {"budget_exhausted", "segment_exceeds_budget"}:
                    skipped_budget += 1
                elif status in {"segment_too_short_for_min_frames", "saved_frames_below_min"}:
                    skipped_min_frames += 1
                else:
                    fail += 1

            perf_meter.update_tqdm(
                frame_pbar,
                saved_real,
                saved_fake,
                _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
                _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
            )

        perf_meter.update_tqdm(
            frame_pbar,
            saved_real,
            saved_fake,
            _remaining_for_postfix(saved_real, MAX_REAL_IMAGES),
            _remaining_for_postfix(saved_fake, MAX_FAKE_IMAGES),
            force=True,
        )

    clip_list_path, clip_rows = export_processed_clip_list(split_root)
    run_elapsed = max(1e-9, time.perf_counter() - perf_meter.start_time)
    avg_write_fps = run_written_frames / run_elapsed

    print("\n" + "=" * 72)
    print("处理完成")
    if VERBOSE:
        print(f"[结果] success_tasks: {success}")
        print(f"[结果] fail_tasks:    {fail}")
        print(f"[结果] skip_existing: {skipped_existing}")
        print(f"[结果] skip_budget:   {skipped_budget}")
        print(f"[结果] skip_min_frames: {skipped_min_frames}")
    print(f"[结果] real frames:   {format_budget(saved_real, MAX_REAL_IMAGES)}")
    print(f"[结果] fake frames:   {format_budget(saved_fake, MAX_FAKE_IMAGES)}")
    print(f"[性能] written_frames_this_run: {run_written_frames}")
    print(f"[性能] avg_write_fps: {avg_write_fps:.2f} frame/s")
    print(f"[输出] clip_list: {clip_list_path} (rows={clip_rows})")
    print(f"[输出] {split_root}")
    print("=" * 72)


if __name__ == "__main__":
    run()
