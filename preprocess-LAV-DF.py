import os
import json
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio
from pathlib import Path
import subprocess
import shutil
import random


"""
LAV-DF test preprocessing

目标：
1. 从 test 目录读取 mp4
2. 按原参考代码方式均匀抽取窗口帧
3. 从 mp4 中临时提取音频，不保存 wav
4. 生成 mel 频谱图
5. 按原参考代码风格拼接 mel + 5帧图
6. 按整视频标签保存到:
   - 0_real
   - 1_fake

本版本额外改动：
1. 只处理 100 个 real + 100 个 fake
2. 每个视频输出 10 张图，最终约 2000 张
3. 不再使用 imageio_ffmpeg.get_ffmpeg_exe()，避免卡死
4. 优先使用你手动指定的 FFMPEG_EXE；若为空，则使用系统 PATH 中的 ffmpeg
"""


############ Custom parameter ##############
N_EXTRACT = 10          # 每个视频输出 10 张图
WINDOW_LEN = 5          # 每个窗口 5 帧

TARGET_REAL_VIDEOS = 100
TARGET_FAKE_VIDEOS = 100
RANDOM_SEED = 42
SHUFFLE_VIDEO = True

DATASET_ROOT = Path(r"E:\data\LAV-DF\LAV-DF")
VIDEO_ROOT = DATASET_ROOT / "test"
METADATA_PATH = DATASET_ROOT / "metadata.min.json"
OUTPUT_ROOT = Path(r"E:\data\LAV-DF-test")
TEMP_ROOT = Path("./temp")

SAMPLE_RATE = 16000

# 强烈建议你手动写死 ffmpeg.exe 路径，最稳
# 例如：FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
# 如果你已经把 ffmpeg 加到系统 PATH，也可以设为 None
FFMPEG_EXE = None

# 是否清空旧输出目录
CLEAN_OUTPUT = False
############################################


def load_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    meta_map = {}
    for item in items:
        key = item["file"].replace("\\", "/")
        meta_map[key] = item
    return meta_map


def get_video_label(meta_item):
    """
    严格按整视频标签处理，不做窗口级标签重判
    original == null -> real
    original != null -> fake
    """
    original = meta_item.get("original", None)
    if original in (None, "", "null"):
        return 0, "0_real"
    else:
        return 1, "1_fake"


def validate_ffmpeg(ffmpeg_path):
    """
    检查 ffmpeg 是否能正常执行
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_ffmpeg_exe():
    """
    优先级：
    1. 手动指定的 FFMPEG_EXE
    2. 环境变量 FFMPEG_EXE / IMAGEIO_FFMPEG_EXE
    3. 系统 PATH 中的 ffmpeg
    4. 常见 Windows 安装路径
    """
    candidates = []

    def add_candidate(path_like):
        if not path_like:
            return
        # 兼容用户把路径写成带引号的字符串
        candidate = str(path_like).strip().strip('"').strip("'")
        if candidate:
            candidates.append(candidate)

    if FFMPEG_EXE:
        add_candidate(FFMPEG_EXE)

    env_ffmpeg = os.environ.get("FFMPEG_EXE", None)
    if env_ffmpeg:
        add_candidate(env_ffmpeg)

    env_imageio_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE", None)
    if env_imageio_ffmpeg:
        add_candidate(env_imageio_ffmpeg)

    path_ffmpeg_exe = shutil.which("ffmpeg.exe")
    if path_ffmpeg_exe:
        add_candidate(path_ffmpeg_exe)

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        add_candidate(path_ffmpeg)

    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        user_profile = os.environ.get("USERPROFILE", "")
        local_app_data = os.environ.get("LOCALAPPDATA", "")

        windows_candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(program_files, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(program_files_x86, "ffmpeg", "bin", "ffmpeg.exe"),
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            os.path.join(user_profile, "scoop", "shims", "ffmpeg.exe"),
            os.path.join(user_profile, "scoop", "apps", "ffmpeg", "current", "bin", "ffmpeg.exe"),
        ]

        for exe_path in windows_candidates:
            add_candidate(exe_path)

        # winget 安装通常位于该目录下的版本号子目录中
        winget_pkg_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg_root.exists():
            for exe_path in winget_pkg_root.glob("*ffmpeg*/*/ffmpeg.exe"):
                add_candidate(str(exe_path))

    # 去重并校验
    checked = set()
    for exe in candidates:
        if exe in checked:
            continue
        checked.add(exe)

        if Path(exe).exists() or exe.lower() == "ffmpeg" or "ffmpeg" in exe.lower():
            if validate_ffmpeg(exe):
                return exe

    raise RuntimeError(
        "Cannot find a valid ffmpeg executable.\n"
        "可选修复方式:\n"
        "1) 在脚本顶部设置 FFMPEG_EXE = r'C:\\path\\to\\ffmpeg.exe'\n"
        "2) 在当前终端执行: $env:FFMPEG_EXE = 'C:\\path\\to\\ffmpeg.exe'\n"
        "3) 安装并加入 PATH (Windows): winget install Gyan.FFmpeg\n"
        "然后重开终端再运行。"
    )


def extract_audio_waveform_from_video(video_file, ffmpeg_exe, sample_rate=16000):
    """
    从 mp4 中直接提取单通道音频到内存，不保存 wav 文件
    """
    cmd = [
        ffmpeg_exe,
        "-i", str(video_file),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        "-loglevel", "error",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg audio extraction failed: {video_file}\n{err}")

    waveform = np.frombuffer(result.stdout, dtype=np.float32)

    if waveform.size == 0:
        raise RuntimeError(f"No audio extracted from video: {video_file}")

    return waveform


def get_spectrogram_from_video(video_file, temp_mel_path, ffmpeg_exe):
    """
    严格复刻原代码的 mel 生成/保存/再读取逻辑
    原代码:
        data, sr = librosa.load(audio_file)
        mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
        plt.imsave("./temp/mel.png", mel)
        mel = plt.imread("./temp/mel.png") * 255
    """
    data = extract_audio_waveform_from_video(video_file, ffmpeg_exe, SAMPLE_RATE)
    sr = SAMPLE_RATE

    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(str(temp_mel_path), mel)

    mel_img = plt.imread(str(temp_mel_path)) * 255
    mel_img = mel_img.astype(np.uint8)
    return mel_img


def prepare_output_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "0_real").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "1_fake").mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT:
        for folder_name in ["0_real", "1_fake"]:
            folder = OUTPUT_ROOT / folder_name
            for file in folder.glob("*.png"):
                try:
                    file.unlink()
                except Exception:
                    pass


def build_candidate_lists(metadata_map):
    """
    从 test 目录中建立 real / fake 候选视频列表
    """
    video_list = sorted([p for p in VIDEO_ROOT.iterdir() if p.suffix.lower() == ".mp4"])

    real_videos = []
    fake_videos = []

    for video_path in video_list:
        rel_key = f"test/{video_path.name}".replace("\\", "/")
        if rel_key not in metadata_map:
            continue

        meta_item = metadata_map[rel_key]
        label_id, _ = get_video_label(meta_item)

        if label_id == 0:
            real_videos.append(video_path)
        else:
            fake_videos.append(video_path)

    if SHUFFLE_VIDEO:
        random.seed(RANDOM_SEED)
        random.shuffle(real_videos)
        random.shuffle(fake_videos)

    return real_videos, fake_videos


def process_one_video(video_path, dataset_name, temp_mel_path, ffmpeg_exe):
    """
    处理单个视频
    返回:
        success(bool), saved_count(int), error_message(str or None)
    """
    try:
        # load video
        video_capture = cv2.VideoCapture(str(video_path))
        if not video_capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            video_capture.release()
            raise RuntimeError(f"Invalid frame count: {video_path}")

        # select N_EXTRACT starting points from frames
        max_start = frame_count - WINDOW_LEN - 1
        if max_start < 0:
            max_start = 0

        if max_start == 0:
            frame_idx = [0] * N_EXTRACT
        else:
            # 这里不能用 np.uint8，否则帧数 >255 时会溢出
            frame_idx = np.linspace(
                0,
                max_start,
                N_EXTRACT,
                endpoint=True
            ).astype(np.int32).tolist()

        frame_idx.sort()

        # selected frames
        frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]

        frame_list = []
        current_frame = 0
        last_needed = frame_sequence[-1] if len(frame_sequence) > 0 else 0

        while current_frame <= last_needed:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Error in reading frame {video_path.name}: {current_frame}")
                break

            if current_frame in frame_sequence:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame = cv2.resize(frame, (500, 500))
                frame_list.append(frame)

            current_frame += 1

        video_capture.release()

        # 如果因为短视频或读取问题导致帧不足，补齐
        needed_len = len(frame_sequence)
        if len(frame_list) < needed_len:
            if len(frame_list) == 0:
                pad_frame = np.zeros((500, 500, 4), dtype=np.uint8)
            else:
                pad_frame = frame_list[-1].copy()

            while len(frame_list) < needed_len:
                frame_list.append(pad_frame.copy())

        # load audio from mp4 and create mel spectrogram
        name = video_path.stem
        group = 0

        mel = get_spectrogram_from_video(video_path, temp_mel_path, ffmpeg_exe)

        if mel.shape[1] <= 0:
            raise RuntimeError(f"Invalid mel spectrogram width: {video_path}")

        mapping = mel.shape[1] / frame_count
        saved_count = 0

        for i in range(len(frame_list)):
            idx = i % WINDOW_LEN
            if idx == 0:
                try:
                    begin = int(np.round(frame_sequence[i] * mapping))
                    end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))

                    if end <= begin:
                        end = begin + 1

                    # 防止越界
                    begin = max(0, min(begin, mel.shape[1] - 1))
                    end = max(begin + 1, min(end, mel.shape[1]))

                    sub_mel = cv2.resize(
                        mel[:, begin:end],
                        (500 * WINDOW_LEN, 500)
                    )

                    x = np.concatenate(frame_list[i: i + WINDOW_LEN], axis=1)
                    x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                    save_path = OUTPUT_ROOT / dataset_name / f"{name}_{group}.png"
                    plt.imsave(str(save_path), x)
                    group += 1
                    saved_count += 1

                except ValueError:
                    print(f"ValueError: {name}")
                    continue

        if saved_count == 0:
            raise RuntimeError(f"No image saved for video: {video_path}")

        return True, saved_count, None

    except Exception as e:
        return False, 0, str(e)


def run():
    prepare_output_dirs()

    ffmpeg_exe = resolve_ffmpeg_exe()
    print(f"Using ffmpeg: {ffmpeg_exe}")

    metadata_map = load_metadata(METADATA_PATH)

    real_candidates, fake_candidates = build_candidate_lists(metadata_map)

    print(f"Found real candidate videos: {len(real_candidates)}")
    print(f"Found fake candidate videos: {len(fake_candidates)}")

    if len(real_candidates) < TARGET_REAL_VIDEOS:
        raise RuntimeError(
            f"Not enough real videos. Need {TARGET_REAL_VIDEOS}, but only found {len(real_candidates)}"
        )
    if len(fake_candidates) < TARGET_FAKE_VIDEOS:
        raise RuntimeError(
            f"Not enough fake videos. Need {TARGET_FAKE_VIDEOS}, but only found {len(fake_candidates)}"
        )

    temp_mel_path = TEMP_ROOT / "mel.png"

    success_real_videos = 0
    success_fake_videos = 0
    saved_real_images = 0
    saved_fake_images = 0
    failed_list = []

    print("Start processing real videos...")
    real_pbar = tqdm(real_candidates, total=len(real_candidates))
    for video_path in real_pbar:
        if success_real_videos >= TARGET_REAL_VIDEOS:
            break

        ok, saved_cnt, err = process_one_video(
            video_path=video_path,
            dataset_name="0_real",
            temp_mel_path=temp_mel_path,
            ffmpeg_exe=ffmpeg_exe
        )

        if ok:
            success_real_videos += 1
            saved_real_images += saved_cnt
            real_pbar.set_description(
                f"real videos={success_real_videos}/{TARGET_REAL_VIDEOS}, images={saved_real_images}"
            )
        else:
            failed_list.append((video_path.name, err))

    print("Start processing fake videos...")
    fake_pbar = tqdm(fake_candidates, total=len(fake_candidates))
    for video_path in fake_pbar:
        if success_fake_videos >= TARGET_FAKE_VIDEOS:
            break

        ok, saved_cnt, err = process_one_video(
            video_path=video_path,
            dataset_name="1_fake",
            temp_mel_path=temp_mel_path,
            ffmpeg_exe=ffmpeg_exe
        )

        if ok:
            success_fake_videos += 1
            saved_fake_images += saved_cnt
            fake_pbar.set_description(
                f"fake videos={success_fake_videos}/{TARGET_FAKE_VIDEOS}, images={saved_fake_images}"
            )
        else:
            failed_list.append((video_path.name, err))

    total_videos = success_real_videos + success_fake_videos
    total_images = saved_real_images + saved_fake_images

    print("\nFinished.")
    print(f"Processed real videos : {success_real_videos}")
    print(f"Processed fake videos : {success_fake_videos}")
    print(f"Processed total videos: {total_videos}")
    print(f"Saved real images     : {saved_real_images}")
    print(f"Saved fake images     : {saved_fake_images}")
    print(f"Saved total images    : {total_images}")

    if success_real_videos < TARGET_REAL_VIDEOS:
        print(
            f"[WARN] real successful videos less than target: {success_real_videos}/{TARGET_REAL_VIDEOS}"
        )
    if success_fake_videos < TARGET_FAKE_VIDEOS:
        print(
            f"[WARN] fake successful videos less than target: {success_fake_videos}/{TARGET_FAKE_VIDEOS}"
        )

    if failed_list:
        print(f"\nFailed videos: {len(failed_list)}")
        for item in failed_list[:20]:
            print("[ERROR]", item[0], "->", item[1])


if __name__ == "__main__":
    run()