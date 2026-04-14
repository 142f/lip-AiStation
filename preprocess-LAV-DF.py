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
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
LAV-DF 数据集处理流水线 - 性能优化增强版
目录结构期望:
├── train
├── dev
├── test
├── metadata.min.json
└── README.md
"""

############ 自定义参数配置区 ##############
N_EXTRACT = 10      # 每个视频片段最多提取的图像组数
WINDOW_LEN = 5      # 每次提取的连续帧窗口长度
IMAGE_SIZE = 500    # 输出图像的目标尺寸
MAX_REAL_IMAGES = 2000   # 真实图像的最大数量阈值
MAX_FAKE_IMAGES = 2000   # 伪造图像的最大数量阈值
MAX_VIDEO = None    # None 表示不限制测试的视频总数，处理全量数据
TARGET_SPLIT = "test"  # 仅处理此划分集合："train", "test", 或 "dev"
RANDOM_SEED = 42    # 随机种子，确保每次采样的可复现性
NUM_THREADS = 4     # 线程池并发数量，请根据 CPU 核心数调节
# 视觉任务场景：忽略仅音频伪造（RF: modify_video=False, modify_audio=True）
EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL = True

dataset_root = r"E:\data\LAV-DF"          # LAV-DF 数据集根目录
output_root = r"E:\data\LAV-DF-test-mb"    # 固定输出路径
metadata_file = ""                         # 可选：显式指定 metadata 文件路径
FFMPEG_EXE = "ffmpeg"                      # ffmpeg 环境变量指令
FFMPEG_TIMEOUT_SEC = 15                    # 预防僵尸进程的超时时长限制
############################################

def resolve_dataset_and_metadata(root_dir, metadata_path=""):
    """解析并定位数据集根目录及元数据配置文件"""
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

class AtomicImageCounter:
    """线程安全的原子计数器，用于精准控制多线程下的总采样配额"""
    def __init__(self, initial_value, max_value):
        self._value = int(initial_value)
        self._max_value = int(max_value)
        self._lock = threading.Lock()

    def has_remaining(self):
        with self._lock:
            return self._value < self._max_value

    def try_acquire(self):
        """尝试获取一个配额，如果未满则返回 True 并累加，满则返回 False"""
        with self._lock:
            if self._value >= self._max_value:
                return False
            self._value += 1
            return True

    def release(self):
        """释放占用的配额（当图像生成中途报错时回退配额）"""
        with self._lock:
            if self._value > 0:
                self._value -= 1

    def get(self):
        with self._lock:
            return self._value

def extract_audio(video_file, audio_path):
    """提取视频中的音频轨道为 16kHz 的 pcm_s16le 格式，捕获超时异常防范主进程卡死"""
    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error", "-i", video_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                       check=False, timeout=FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        # 抛弃超时进程，拦截底层死锁隐患
        pass

def get_spectrogram(video_file):
    """
    性能优化点：提取音频并生成梅尔频谱图。
    通过 io.BytesIO 拦截 Matplotlib 的磁盘写出动作，实现 100% 内存渲染操作，
    彻底消除该瓶颈环节的磁盘 IOPS 磨损。
    """
    # 仍保留 wav 落盘，因为 librosa/ffmpeg 在某些系统上对管道流存在兼容性边界
    # 但改用 NamedTemporaryFile 自行管理上下文，封堵文件描述符泄漏风险
    with tempfile.NamedTemporaryFile(prefix="temp_audio_", suffix=".wav", delete=False, dir="./temp") as temp_wav_file:
        temp_wav = temp_wav_file.name

    try:
        extract_audio(video_file, temp_wav)
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
            return np.zeros((512, 1000, 4), dtype=np.uint8)

        data, sr = librosa.load(temp_wav, sr=None)
        if len(data) == 0:
             return np.zeros((512, 1000, 4), dtype=np.uint8)
             
        mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
        
        # [核心优化] 使用 BytesIO 将文件读写替换为内存字节流，结果像素级等价
        buf = io.BytesIO()
        plt.imsave(buf, mel, format='png')  # Matplotlib 引擎对 ndarray 隐式应用 viridis 颜色映射
        buf.seek(0)                         # 重置指针到文件头
        mel_img = plt.imread(buf) * 255     # imread 读取为 0-1 范围的 float32，转为 0-255 标量
        return mel_img.astype(np.uint8)
    finally:
        # 严格的清理机制，无视 IO 异常
        try:
            if os.path.exists(temp_wav): os.remove(temp_wav)
        except OSError:
            pass

def get_start_frames(start_frame, end_frame):
    """基于指定的帧范围，计算均匀间隔的采样起始帧索引"""
    valid_num = end_frame - start_frame - WINDOW_LEN
    if valid_num < 1:
        return []
    extract_num = min(N_EXTRACT, valid_num)
    if extract_num == 1:
        return [start_frame]
    frame_idx = np.linspace(start_frame, end_frame - WINDOW_LEN - 1, extract_num, endpoint=True, dtype=np.int32).tolist()
    return sorted(set([int(f) for f in frame_idx]))

def read_needed_frames(video_file, frame_sequence):
    """
    性能优化点：读取指定序列的目标帧。
    通过 grab() + retrieve() 组合跳过不需要的帧的像素解码操作，节约海量 CPU 周期。
    引入 try...finally 保证硬件资源的释放。
    """
    video_capture = cv2.VideoCapture(video_file)
    frame_set = set(frame_sequence)
    frame_map = {}
    current_frame = 0
    max_needed_frame = frame_sequence[-1] if frame_sequence else -1
    
    try:
        while current_frame <= max_needed_frame:
            # 仅移动视频流指针，不执行耗时的载荷（BGR转换）解码
            ret = video_capture.grab()
            if not ret: break
            
            # 若是目标帧，才真正调用 retrieve 拉取像素数据
            if current_frame in frame_set:
                valid_decode, frame = video_capture.retrieve()
                if valid_decode:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    frame_map[current_frame] = frame
            current_frame += 1
    finally:
        # 前置拦截内存泄露：无论异常与否，释放底层 C++ 句柄
        video_capture.release()
        
    return frame_map

def count_existing_images(save_dir):
    """统计当前目录中已经存在的 PNG 文件数，支持断点续传"""
    if not os.path.isdir(save_dir): return 0
    return sum(1 for _, _, files in os.walk(save_dir) for f in files if f.lower().endswith(".png"))

def extract_segment(video_file, save_dir, name, t0, t1, max_images=None, counter=None):
    """从目标视频中裁剪画面帧并结合梅尔频谱保存拼接图像"""
    if max_images is not None and max_images <= 0:
        return 0
    if counter is not None and not counter.has_remaining():
        return 0
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        video_capture = cv2.VideoCapture(video_file)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

        if fps <= 0 or frame_count <= 0: return 0

        start_frame = max(0, int(np.floor(t0 * fps)))
        end_frame = min(frame_count, int(np.ceil(t1 * fps)))
        start_frames = get_start_frames(start_frame, end_frame)
        if not start_frames: return 0

        frame_sequence = [i for num in start_frames for i in range(num, num + WINDOW_LEN)]
        frame_map = read_needed_frames(video_file, frame_sequence)
        
        mel = get_spectrogram(video_file)
        # 保护边界：防止除以0或处理异常短的文件
        mapping = mel.shape[1] / max(frame_count, 1)

        group = 0
        for start in start_frames:
            if max_images is not None and group >= max_images:
                break
            if counter is not None and not counter.has_remaining():
                break

            frames = []
            ok = True
            for idx in range(start, start + WINDOW_LEN):
                if idx not in frame_map:
                    ok = False
                    break
                frames.append(frame_map[idx])
            if not ok: continue

            reserved = False
            try:
                begin = np.round(start * mapping)
                end = np.round((start + WINDOW_LEN) * mapping)
                begin = max(0, min(int(begin), mel.shape[1] - 1))
                end = max(begin + 1, min(int(end), mel.shape[1]))

                # 取出频谱片段并重采样以对齐画面宽度
                sub_mel = cv2.resize(mel[:, begin:end], (IMAGE_SIZE * WINDOW_LEN, IMAGE_SIZE))
                x = np.concatenate(frames, axis=1)
                # 拼接：上方为截取的频谱特征，下方为连续图像序列（舍弃透明度通道）
                x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                # 并发执行的配额竞争预占
                if counter is not None:
                    if not counter.try_acquire():
                        break
                    reserved = True
                
                save_path = os.path.join(save_dir, f"{name}_{group}.png")
                plt.imsave(save_path, x.astype(np.uint8))
                
                group += 1
            except Exception as e:
                # 若保存中断，必须撤销原子计数器的预占数，防止数据少采
                if reserved and counter is not None:
                    counter.release()
                continue

        return group
    except Exception as e:
        return 0

def process_fake_period(item, seg_id, period, data_root, meta_dict, fake_out_dir, real_out_dir, fake_counter, real_counter):
    """多线程分发单元：处理单条伪造周期并尝试匹配生成真实对抗对 (Pair)"""
    if not fake_counter.has_remaining() and not real_counter.has_remaining():
        return

    video_file = os.path.join(data_root, item["file"])
    if not os.path.exists(video_file):
        return

    try:
        t0, t1 = period
    except Exception:
        return

    fake_name = os.path.splitext(os.path.basename(item["file"]))[0]

    # 1. 从伪造视频中提取序列
    if fake_counter.has_remaining():
        extract_segment(video_file, fake_out_dir, f"{fake_name}_seg{seg_id}", t0, t1, counter=fake_counter)

    # 2. 如果条件允许，从原始来源(Real)视频提取完全对齐的一致性切片，提供网络对比学习的基础
    if real_counter.has_remaining() and item["original"] is not None and item["original"] in meta_dict:
        real_item = meta_dict[item["original"]]
        real_file = os.path.join(data_root, real_item["file"])
        if os.path.exists(real_file):
            real_name = os.path.splitext(os.path.basename(real_item["file"]))[0]
            # 计算原始视频的时间映射（相对比例对齐）
            r0 = t0 / max(item["duration"], 1e-8) * real_item["duration"]
            r1 = t1 / max(item["duration"], 1e-8) * real_item["duration"]
            extract_segment(real_file, real_out_dir, f"{real_name}_pair_{fake_name}_seg{seg_id}", r0, r1, counter=real_counter)

def process_full_real_video(item, data_root, real_out_dir, real_counter):
    """多线程分发单元：处理完整的真实视频片段（用于兜底填充配额）"""
    if not real_counter.has_remaining():
        return

    video_file = os.path.join(data_root, item["file"])
    if not os.path.exists(video_file):
        return

    name = os.path.splitext(os.path.basename(item["file"]))[0]
    extract_segment(video_file, real_out_dir, name, 0.0, item["duration"], counter=real_counter)

def run():
    """入口逻辑架构封装"""
    data_root, meta_path = resolve_dataset_and_metadata(dataset_root, metadata_file)
    print(f"[Info] 设置启动... 目标存入 {TARGET_SPLIT} 文件夹. 最大处理：真实 {MAX_REAL_IMAGES} 张 / 伪造 {MAX_FAKE_IMAGES} 张.")

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

    print(f"[Info] 类型计数(原始 split): {type_counts_all}")
    print(f"[Info] RF 过滤开关: {EXCLUDE_AUDIO_ONLY_FAKE_IN_VISUAL}, 已排除 RF 数量: {excluded_rf}")
    print(f"[Info] 入选视频数: real={len(real_list)} fake={len(fake_list)}")

    # 固定随机种子并重排序列以保障随机采样多样性
    random.seed(RANDOM_SEED)
    random.shuffle(real_list)
    random.shuffle(fake_list)

    if MAX_VIDEO is not None:
        real_list = real_list[:MAX_VIDEO]
        fake_list = fake_list[:MAX_VIDEO]

    real_out_dir = os.path.join(output_root, TARGET_SPLIT, "0_real")
    fake_out_dir = os.path.join(output_root, TARGET_SPLIT, "1_fake")
    
    saved_real = count_existing_images(real_out_dir)
    saved_fake = count_existing_images(fake_out_dir)
    real_counter = AtomicImageCounter(saved_real, MAX_REAL_IMAGES)
    fake_counter = AtomicImageCounter(saved_fake, MAX_FAKE_IMAGES)

    print(f"[Info] 发现已有 Real 数据：{saved_real}/{MAX_REAL_IMAGES}")
    print(f"[Info] 发现已有 Fake 数据：{saved_fake}/{MAX_FAKE_IMAGES}")
    print(f"[Info] 并发线程数激活: {NUM_THREADS}")

    if fake_counter.has_remaining() or real_counter.has_remaining():
        print(">> 阶段 1: 扫描 Fake 列表及其配对的 Real 切片 ...")
        fake_tasks = [
            (item, seg_id, period)
            for item in fake_list
            for seg_id, period in enumerate(item.get("fake_periods", []))
        ]
        if fake_tasks:
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = [
                    executor.submit(
                        process_fake_period,
                        item, seg_id, period, data_root, meta_dict,
                        fake_out_dir, real_out_dir, fake_counter, real_counter
                    )
                    for item, seg_id, period in fake_tasks
                ]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Fake/Pair 抽取进度"):
                    pass

    # 若前期提取配对切片仍未塞满配额，执行兜底提取
    if real_counter.has_remaining():
        print(">> 阶段 2: 填充未达标的 Real 配额 (全长采样) ...")
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(process_full_real_video, item, data_root, real_out_dir, real_counter)
                for item in real_list
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="全长 Real 抽取进度"):
                pass

    saved_real = real_counter.get()
    saved_fake = fake_counter.get()

    print(f"\n[Info] 脚本执行完毕。系统最终落盘: {saved_real} 张 Real 对抗图, {saved_fake} 张 Fake 对抗图.")

if __name__ == "__main__":
    # 初始化前置依赖目录
    os.makedirs(output_root, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    try:
        run()
    except KeyboardInterrupt:
        print("\n[Info] 系统捕获到中断信号(Ctrl+C)，安全停止服务。")