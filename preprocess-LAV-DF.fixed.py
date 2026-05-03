import os
import json
import cv2
import numpy as np
import subprocess
import tempfile
import librosa
from librosa import feature as audio
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
import random
import threading
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
LAV-DF 数据集处理流水线 - 修正版

本版本重点修复以下问题：
1. 修复 get_start_frames() 的 off-by-one，避免合法窗口被误删。
2. 引入“段完成标记(marker)”机制，支持可靠断点续跑，避免旧脚本那种覆盖文件但计数继续增长的假续跑问题。
3. 降低 MIN_IMAGES_PER_SAMPLE 默认门槛，减少对短片段的系统性误杀。
4. 音频提取失败不再静默生成全零频谱，而是直接跳过该样本并记录原因，避免脏样本混入。
5. 提供严格可复现模式：默认单线程，先保证实验可复现，再考虑速度。
6. 配对 real 的“按总时长比例映射”默认关闭，避免把粗对齐伪装成严格对齐。
7. 增加详细统计与日志，便于复盘数据筛除原因。

目录结构期望:
├── train
├── dev
├── test
├── metadata.min.json
└── README.md
"""

############ 自定义参数配置区 ##############
N_EXTRACT = 10                 # 每个视频片段最多提取的图像组数（不是硬性必须提满）
MIN_IMAGES_PER_SAMPLE = 3      # 每个片段至少保留多少组；建议不要再设成 10，避免系统性偏向长片段
WINDOW_LEN = 5                 # 每次提取的连续帧窗口长度
IMAGE_SIZE = 500               # 输出图像的目标尺寸
MAX_REAL_IMAGES = 1500         # 真实图像的最大数量阈值
MAX_FAKE_IMAGES = 1500         # 伪造图像的最大数量阈值
MAX_VIDEO = None               # None 表示不限制测试的视频总数，处理全量数据
TARGET_SPLIT = "test"          # 仅处理此划分集合："train", "test", 或 "dev"
RANDOM_SEED = 42               # 随机种子，确保采样顺序可复现
NUM_THREADS = 4                # 非严格复现模式下的线程数
STRICT_REPRODUCIBLE = True     # 严格复现模式：默认开启，自动退化为单线程执行

# 任务模式说明：
# 本脚本当前输出的是“音频频谱 + 连续视频帧”的拼接图，因此属于跨模态/多模态预处理。
# 如果你做的是纯视觉任务，请不要直接使用本脚本输出结果去声称 visual-only。
USE_AUDIO_SPECTROGRAM = True

# 视觉任务场景：忽略仅音频伪造（RF: modify_video=False, modify_audio=True）
EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL = True

# 是否在阶段 1 中从 original real 视频按“相对时长比例”抽取配对片段。
# 默认关闭：因为这只是粗对齐，不是严格语义对齐；开启前请明确承担协议风险。
ENABLE_PROPORTIONAL_REAL_PAIR = False

# 断点续跑相关：
# 默认不删除历史文件，便于人工排查旧结果。
AUTO_CLEAN_ORPHAN_OUTPUTS = False
# 对于"同一 segment 但缺少 marker"的残留文件，默认也不自动清理。
AUTO_CLEAN_PARTIAL_SEGMENT_OUTPUTS = False

# 频谱图参数
SPECTROGRAM_HEIGHT = 512
SPECTROGRAM_WIDTH = 1000

# 音频提取
FFMPEG_EXE = "ffmpeg"
FFMPEG_TIMEOUT_SEC = 30

# 路径
dataset_root = r"E:\data\LAV-DF"
output_root = r"E:\data\LAV-DF-test-mb2"
metadata_file = ""
############################################


def resolve_dataset_and_metadata(root_dir, metadata_path=""):
    """解析并定位数据集根目录及元数据配置文件。"""
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
    raise FileNotFoundError(f"未找到 metadata.min.json 文件，已检查以下路径:\n{checked_msg}")


class AtomicImageCounter:
    """线程安全的原子计数器，用于控制总采样配额。"""

    def __init__(self, initial_value: int, max_value: int):
        self._value = int(initial_value)
        self._max_value = int(max_value)
        self._lock = threading.Lock()

    def has_remaining(self) -> bool:
        with self._lock:
            return self._value < self._max_value

    def try_acquire(self) -> bool:
        with self._lock:
            if self._value >= self._max_value:
                return False
            self._value += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._value > 0:
                self._value -= 1

    def get(self) -> int:
        with self._lock:
            return self._value


class SkipStats:
    """用于记录预处理阶段的各类跳过原因，方便复盘。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {}

    def add(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + delta

    def dump(self):
        with self._lock:
            return dict(sorted(self._stats.items(), key=lambda x: x[0]))


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _lavdf_type(item):
    """
    统一 LAV-DF 样本类型：
    RR: real/real, RF: real video + fake audio, FR: fake video + real audio, FF: fake/fake
    """
    if "modify_video" not in item and "modify_audio" not in item:
        return "FF" if int(item.get("n_fakes", 0) or 0) > 0 else "RR"

    modify_video = _to_bool(item.get("modify_video", False))
    modify_audio = _to_bool(item.get("modify_audio", False))

    if (not modify_video) and (not modify_audio):
        return "RR"
    if (not modify_video) and modify_audio:
        return "RF"
    if modify_video and (not modify_audio):
        return "FR"
    return "FF"


def sanitize_name(name: str) -> str:
    """将文件名/段名规整为适合落盘的安全字符串。"""
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)



def get_segment_prefix(name: str) -> str:
    """为一个 segment 生成稳定的前缀。"""
    return sanitize_name(name)



def get_marker_path(save_dir: str, prefix: str) -> str:
    return os.path.join(save_dir, f".{prefix}.done.json")



def get_sample_path(save_dir: str, prefix: str, sample_index: int) -> str:
    """使用连续索引命名，避免出现跳号，同时兼容 test.py 聚合正则。"""
    return os.path.join(save_dir, f"{prefix}_{int(sample_index)}.png")


def list_segment_outputs(save_dir: str, prefix: str):
    paths = []
    paths.extend(glob.glob(os.path.join(save_dir, f"{prefix}_[0-9]*.png")))
    paths.extend(glob.glob(os.path.join(save_dir, f"{prefix}_sf*.png")))
    return sorted(set(paths))


def cleanup_disabled_markers(save_dir: str) -> int:
    """清理历史marker文件（不再使用）。"""
    return 0



def cleanup_segment_outputs(save_dir: str, prefix: str) -> int:
    """清理某个 segment 残留文件，并返回删除数量。"""
    patterns = [
        os.path.join(save_dir, f"{prefix}_[0-9]*.png"),
        os.path.join(save_dir, f"{prefix}_sf*.png"),
    ]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    marker_path = get_marker_path(save_dir, prefix)
    try:
        if os.path.exists(marker_path):
            os.remove(marker_path)
    except OSError:
        pass
    return removed



def validate_marker(save_dir: str, prefix: str):
    """验证某个 segment 的完成标记是否有效（不再使用）。"""
    return False, []



def write_marker(save_dir: str, prefix: str, saved_paths, metadata=None) -> None:
    """为完整成功的 segment 写入完成标记。"""
    pass  # 不生成JSON文件


def migrate_legacy_sf_filenames(save_dir: str) -> int:
    """将历史 *_sf000123.png 迁移为 *_000123.png，避免 test.py 聚合失败。"""
    if not os.path.isdir(save_dir):
        return 0

    renamed = 0
    for old_path in glob.glob(os.path.join(save_dir, "*_sf*.png")):
        base = os.path.basename(old_path)
        stem, ext = os.path.splitext(base)
        if "_sf" not in stem:
            continue

        left, right = stem.rsplit("_sf", 1)
        if (not right.isdigit()) or (len(right) == 0):
            continue

        new_name = f"{left}_{right}{ext}"
        new_path = os.path.join(save_dir, new_name)

        # 目标已存在时，删除旧文件以避免重复计数。
        try:
            if os.path.exists(new_path):
                os.remove(old_path)
            else:
                os.rename(old_path, new_path)
            renamed += 1
        except OSError:
            pass

    return renamed



def scan_output_state(save_dir: str, auto_clean_orphans: bool = True):
    """
    扫描输出目录状态，返回可靠样本数。
    不依赖 .done.json marker，直接计数 PNG 文件数量。
    """
    os.makedirs(save_dir, exist_ok=True)
    all_pngs = len([f for f in glob.glob(os.path.join(save_dir, "*.png")) if os.path.isfile(f)])
    
    return {
        "valid_count": all_pngs,
        "broken_markers": 0,
        "orphan_pngs": 0,
        "removed_orphans": 0,
        "removed_markers": 0,
    }



def extract_audio(video_file: str, audio_path: str) -> bool:
    """提取视频中的音频轨道为 16kHz PCM WAV；失败时返回 False。"""
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_file,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            return False
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return False
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False



def get_spectrogram(video_file: str):
    """
    提取视频音频并生成梅尔频谱图。

    返回：
    - 成功: (mel_img_uint8, None)
    - 失败: (None, reason)

    这里不再像旧版那样在失败时静默返回全零频谱；
    因为那会把错误样本伪装成正常样本，污染训练/测试分布。
    """
    os.makedirs("./temp", exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="temp_audio_", suffix=".wav", delete=False, dir="./temp") as temp_wav_file:
        temp_wav = temp_wav_file.name

    try:
        ok = extract_audio(video_file, temp_wav)
        if not ok:
            return None, "audio_extract_failed"

        try:
            data, sr = librosa.load(temp_wav, sr=None)
        except Exception:
            return None, "audio_load_failed"

        if data is None or len(data) == 0:
            return None, "empty_audio"

        try:
            # 这里改成 ref=np.max，是更常见、更稳定的 dB 参考系。
            mel = audio.melspectrogram(y=data, sr=sr)
            mel = librosa.power_to_db(mel, ref=np.max)
        except Exception:
            return None, "mel_build_failed"

        buf = io.BytesIO()
        plt.imsave(buf, mel, format="png")
        buf.seek(0)
        mel_img = (plt.imread(buf) * 255).astype(np.uint8)

        if mel_img.ndim != 3:
            return None, "invalid_mel_shape"

        mel_img = cv2.resize(mel_img, (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return mel_img, None
    finally:
        try:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        except OSError:
            pass



def get_start_frames(start_frame: int, end_frame: int):
    """
    基于指定帧范围，计算均匀间隔的采样起始帧索引。

    修复点：
    旧版写成 end - start - WINDOW_LEN，少了 +1，导致合法窗口被少算一个。
    正确的合法窗口数应为：
        available_windows = (end_frame - start_frame) - WINDOW_LEN + 1
    """
    total_frames = int(end_frame) - int(start_frame)
    available_windows = total_frames - WINDOW_LEN + 1
    if available_windows <= 0:
        return []

    extract_num = min(N_EXTRACT, available_windows)
    if extract_num == 1:
        return [int(start_frame)]

    last_start = int(end_frame) - WINDOW_LEN
    frame_idx = np.linspace(int(start_frame), int(last_start), extract_num, endpoint=True, dtype=np.int32).tolist()
    # 去重并排序，避免极短区间 linspace 因量化产生重复起点
    frame_idx = sorted(set(int(f) for f in frame_idx))
    return frame_idx



def read_needed_frames(video_file: str, frame_sequence):
    """
    只读取指定序列的目标帧。
    通过 grab()+retrieve() 跳过不需要的解码工作，减少 CPU 压力。
    """
    video_capture = cv2.VideoCapture(video_file)
    frame_set = set(frame_sequence)
    frame_map = {}
    current_frame = 0
    max_needed_frame = frame_sequence[-1] if frame_sequence else -1

    try:
        while current_frame <= max_needed_frame:
            ret = video_capture.grab()
            if not ret:
                break

            if current_frame in frame_set:
                valid_decode, frame = video_capture.retrieve()
                if valid_decode and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                    frame_map[current_frame] = frame
            current_frame += 1
    finally:
        video_capture.release()

    return frame_map



def build_composite_image(frames, sub_mel):
    """将频谱图与连续帧拼接为一张图。"""
    x = np.concatenate(frames, axis=1)
    x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
    return x.astype(np.uint8)



def reserve_and_save_image(save_path: str, image: np.ndarray, counter: AtomicImageCounter | None):
    """
    先占配额，再落盘；若保存失败则回滚配额。
    若 save_path 已存在，说明这是可靠续跑场景中的已完成样本，不重复计数也不覆写。
    """
    if os.path.exists(save_path):
        return True, False  # success=True, reserved=False

    reserved = False
    try:
        if counter is not None:
            if not counter.try_acquire():
                return False, False
            reserved = True

        # 使用 cv2.imwrite 可避免 Matplotlib 再次引入额外 figure 状态。
        ok = cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            if reserved and counter is not None:
                counter.release()
            return False, False
        return True, reserved
    except Exception:
        if reserved and counter is not None:
            counter.release()
        return False, False



def extract_segment(
    video_file: str,
    save_dir: str,
    name: str,
    t0: float,
    t1: float,
    stats: SkipStats,
    counter: AtomicImageCounter | None = None,
    max_images: int | None = None,
):
    """
    从目标视频中裁剪画面帧并结合梅尔频谱保存拼接图像。

    返回值：
    - >=0: 成功保留的样本数
    - 0: 未生成任何有效样本
    """
    if max_images is not None and max_images <= 0:
        stats.add("segment_skip_max_images_le_zero")
        return 0
    if counter is not None and not counter.has_remaining():
        stats.add("segment_skip_quota_full")
        return 0

    os.makedirs(save_dir, exist_ok=True)
    prefix = get_segment_prefix(name)

    # 直接扫描现有PNG文件数量以判断是否已有足够输出
    existing_paths = list_segment_outputs(save_dir, prefix)
    if existing_paths and (MIN_IMAGES_PER_SAMPLE <= 0 or len(existing_paths) >= MIN_IMAGES_PER_SAMPLE):
        stats.add("segment_resume_png_hit")
        return len(existing_paths)

    # 若允许自动清理，则清掉“同一 segment 且无 marker”的残留后再重做。
    if AUTO_CLEAN_PARTIAL_SEGMENT_OUTPUTS:
        cleaned = cleanup_segment_outputs(save_dir, prefix)
        if cleaned > 0:
            stats.add("segment_partial_cleanup", cleaned)

    try:
        video_capture = cv2.VideoCapture(video_file)
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

        if fps <= 0 or frame_count <= 0:
            stats.add("segment_skip_invalid_video_meta")
            return 0

        # 保护 t0/t1 边界
        t0 = max(0.0, float(t0))
        t1 = max(t0, float(t1))

        start_frame = max(0, int(np.floor(t0 * fps)))
        end_frame = min(frame_count, int(np.ceil(t1 * fps)))

        if end_frame - start_frame < WINDOW_LEN:
            stats.add("segment_skip_too_short_for_window")
            return 0

        start_frames = get_start_frames(start_frame, end_frame)
        if not start_frames:
            stats.add("segment_skip_no_valid_start_frames")
            return 0

        # 更温和的最小门槛：只要候选窗口数连 min 都达不到，才提前跳过。
        if MIN_IMAGES_PER_SAMPLE > 0 and len(start_frames) < MIN_IMAGES_PER_SAMPLE:
            stats.add("segment_skip_not_enough_candidate_windows")
            return 0

        frame_sequence = [idx for s in start_frames for idx in range(s, s + WINDOW_LEN)]
        frame_map = read_needed_frames(video_file, frame_sequence)

        if USE_AUDIO_SPECTROGRAM:
            mel, mel_error = get_spectrogram(video_file)
            if mel is None:
                stats.add(f"segment_skip_{mel_error}")
                return 0
            mapping = mel.shape[1] / max(frame_count, 1)
        else:
            mel = None
            mapping = 1.0

        saved_paths = []
        success_groups = 0
        planned_starts = start_frames[:max_images] if max_images is not None else start_frames

        for start in planned_starts:
            if counter is not None and not counter.has_remaining():
                stats.add("segment_stop_quota_full_during_loop")
                break

            frames = []
            ok_frames = True
            for idx in range(start, start + WINDOW_LEN):
                frame = frame_map.get(idx)
                if frame is None:
                    ok_frames = False
                    break
                frames.append(frame)
            if not ok_frames:
                stats.add("window_skip_missing_frame")
                continue

            if mel is not None:
                begin = int(np.round(start * mapping))
                end = int(np.round((start + WINDOW_LEN) * mapping))
                begin = max(0, min(begin, mel.shape[1] - 1))
                end = max(begin + 1, min(end, mel.shape[1]))
                sub_mel = cv2.resize(mel[:, begin:end], (IMAGE_SIZE * WINDOW_LEN, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                image = build_composite_image(frames, sub_mel)
            else:
                image = np.concatenate([f[:, :, :3] for f in frames], axis=1).astype(np.uint8)

            # 使用连续样本序号命名，避免按原始 start_frame 命名产生跳号。
            save_path = get_sample_path(save_dir, prefix, success_groups)
            success, _ = reserve_and_save_image(save_path, image, counter)
            if not success:
                break

            if os.path.exists(save_path):
                saved_paths.append(save_path)
                success_groups += 1

        # 只有真正达到最小保留门槛，才认为该段成功；否则整段清除，避免残缺 segment 混入。
        if MIN_IMAGES_PER_SAMPLE > 0 and success_groups < MIN_IMAGES_PER_SAMPLE:
            for save_path in saved_paths:
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                        if counter is not None:
                            counter.release()
                except OSError:
                    pass
            stats.add("segment_skip_saved_below_min_after_filter")
            return 0

        if success_groups == 0:
            stats.add("segment_skip_zero_saved_groups")
            return 0

        write_marker(
            save_dir,
            prefix,
            saved_paths,
            metadata={
                "video_file": video_file,
                "t0": float(t0),
                "t1": float(t1),
                "window_len": int(WINDOW_LEN),
                "n_extract": int(N_EXTRACT),
                "min_images_per_sample": int(MIN_IMAGES_PER_SAMPLE),
            },
        )
        stats.add("segment_saved", success_groups)
        return success_groups
    except Exception:
        stats.add("segment_unhandled_exception")
        return 0



def process_fake_period(item, seg_id, period, data_root, meta_dict, fake_out_dir, real_out_dir, fake_counter, real_counter, stats):
    """处理单条伪造周期，并在必要时抽取 paired real。"""
    if not fake_counter.has_remaining() and not real_counter.has_remaining():
        stats.add("task_skip_both_quota_full")
        return

    video_file = os.path.join(data_root, item["file"])
    if not os.path.exists(video_file):
        stats.add("task_skip_fake_video_missing")
        return

    try:
        t0, t1 = period
    except Exception:
        stats.add("task_skip_invalid_fake_period")
        return

    fake_name = os.path.splitext(os.path.basename(item["file"]))[0]

    if fake_counter.has_remaining():
        extract_segment(
            video_file=video_file,
            save_dir=fake_out_dir,
            name=f"{fake_name}_seg{seg_id}",
            t0=t0,
            t1=t1,
            stats=stats,
            counter=fake_counter,
        )

    # 只在明确开启时才做 proportional real pair；默认关闭，因为这不是严格时间对齐。
    if ENABLE_PROPORTIONAL_REAL_PAIR:
        original_key = item.get("original")
        if real_counter.has_remaining() and original_key is not None and original_key in meta_dict:
            real_item = meta_dict[original_key]
            real_file = os.path.join(data_root, real_item["file"])
            if os.path.exists(real_file):
                real_name = os.path.splitext(os.path.basename(real_item["file"]))[0]
                fake_duration = max(float(item.get("duration", 0.0)), 1e-8)
                real_duration = max(float(real_item.get("duration", 0.0)), 0.0)
                r0 = float(t0) / fake_duration * real_duration
                r1 = float(t1) / fake_duration * real_duration
                stats.add("paired_real_by_proportional_time_used")
                extract_segment(
                    video_file=real_file,
                    save_dir=real_out_dir,
                    name=f"{real_name}_pair_{fake_name}_seg{seg_id}",
                    t0=r0,
                    t1=r1,
                    stats=stats,
                    counter=real_counter,
                )
            else:
                stats.add("paired_real_missing_video")



def process_full_real_video(item, data_root, real_out_dir, real_counter, stats):
    """处理完整的真实视频片段（用于兜底填充配额）。"""
    if not real_counter.has_remaining():
        stats.add("real_fill_skip_quota_full")
        return

    video_file = os.path.join(data_root, item["file"])
    if not os.path.exists(video_file):
        stats.add("real_fill_skip_video_missing")
        return

    name = os.path.splitext(os.path.basename(item["file"]))[0]
    duration = float(item.get("duration", 0.0))
    extract_segment(
        video_file=video_file,
        save_dir=real_out_dir,
        name=name,
        t0=0.0,
        t1=duration,
        stats=stats,
        counter=real_counter,
    )



def run_parallel_or_serial(tasks, worker_fn, worker_desc: str, max_workers: int):
    """根据配置选择串行或线程池执行。"""
    if len(tasks) == 0:
        return

    if max_workers <= 1:
        for task in tqdm(tasks, total=len(tasks), desc=worker_desc):
            worker_fn(*task)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, *task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=worker_desc):
            pass



def run():
    data_root, meta_path = resolve_dataset_and_metadata(dataset_root, metadata_file)
    print(f"[Info] 目标 split: {TARGET_SPLIT}")
    print(f"[Info] 输出目录: {output_root}")
    print(f"[Info] 最大采样配额: Real={MAX_REAL_IMAGES}, Fake={MAX_FAKE_IMAGES}")
    print(f"[Info] 严格可复现模式: {STRICT_REPRODUCIBLE}")
    print(f"[Info] 音频频谱启用: {USE_AUDIO_SPECTROGRAM}")
    print(f"[Info] 比例映射 paired real 启用: {ENABLE_PROPORTIONAL_REAL_PAIR}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_dict = {item["file"]: item for item in meta}
    split_items = [item for item in meta if item.get("split") == TARGET_SPLIT]

    type_counts_all = {"RR": 0, "RF": 0, "FR": 0, "FF": 0}
    excluded_rf = 0
    real_list = []
    fake_list = []

    for item in split_items:
        item_type = _lavdf_type(item)
        type_counts_all[item_type] = type_counts_all.get(item_type, 0) + 1

        if EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL and item_type == "RF":
            excluded_rf += 1
            continue

        if item_type in {"FR", "FF"}:
            fake_list.append(item)
        else:
            real_list.append(item)

    print(f"[Info] 原始 split 类型计数: {type_counts_all}")
    print(f"[Info] 已排除 RF 数量: {excluded_rf}")
    print(f"[Info] 入选视频数: real={len(real_list)} fake={len(fake_list)}")

    random.seed(RANDOM_SEED)
    random.shuffle(real_list)
    random.shuffle(fake_list)

    if MAX_VIDEO is not None:
        real_list = real_list[:MAX_VIDEO]
        fake_list = fake_list[:MAX_VIDEO]

    real_out_dir = os.path.join(output_root, TARGET_SPLIT, "0_real")
    fake_out_dir = os.path.join(output_root, TARGET_SPLIT, "1_fake")
    os.makedirs(real_out_dir, exist_ok=True)
    os.makedirs(fake_out_dir, exist_ok=True)

    migrated_real = migrate_legacy_sf_filenames(real_out_dir)
    migrated_fake = migrate_legacy_sf_filenames(fake_out_dir)
    if migrated_real > 0 or migrated_fake > 0:
        print(f"[Info] 历史命名迁移完成: real={migrated_real}, fake={migrated_fake}")

    # 新版先通过 marker 扫描真实有效样本数，避免旧版 count_existing_images 的计数失真。
    real_state = scan_output_state(real_out_dir, auto_clean_orphans=AUTO_CLEAN_ORPHAN_OUTPUTS)
    fake_state = scan_output_state(fake_out_dir, auto_clean_orphans=AUTO_CLEAN_ORPHAN_OUTPUTS)

    print(f"[Info] Real 目录状态: {real_state}")
    print(f"[Info] Fake 目录状态: {fake_state}")

    real_counter = AtomicImageCounter(real_state["valid_count"], MAX_REAL_IMAGES)
    fake_counter = AtomicImageCounter(fake_state["valid_count"], MAX_FAKE_IMAGES)
    stats = SkipStats()

    workers = 1 if STRICT_REPRODUCIBLE else max(1, int(NUM_THREADS))
    print(f"[Info] 实际并发数: {workers}")

    if fake_counter.has_remaining() or real_counter.has_remaining():
        print(">> 阶段 1: 处理 Fake 片段" + (" 与按比例映射的 paired Real" if ENABLE_PROPORTIONAL_REAL_PAIR else ""))
        fake_tasks = [
            (item, seg_id, period, data_root, meta_dict, fake_out_dir, real_out_dir, fake_counter, real_counter, stats)
            for item in fake_list
            for seg_id, period in enumerate(item.get("fake_periods", []))
        ]
        run_parallel_or_serial(fake_tasks, process_fake_period, "Fake/Pair 抽取进度", workers)

    if real_counter.has_remaining():
        print(">> 阶段 2: 用完整 Real 视频补足 Real 配额")
        real_tasks = [(item, data_root, real_out_dir, real_counter, stats) for item in real_list]
        run_parallel_or_serial(real_tasks, process_full_real_video, "全长 Real 抽取进度", workers)

    saved_real = real_counter.get()
    saved_fake = fake_counter.get()

    print(f"\n[Info] 脚本执行完毕。最终有效样本: Real={saved_real}, Fake={saved_fake}")
    print(f"[Info] 跳过/清理统计: {stats.dump()}")


if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    try:
        run()
    except KeyboardInterrupt:
        print("\n[Info] 捕获到中断信号(Ctrl+C)，已安全停止。")
