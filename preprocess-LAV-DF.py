import os
import json
import cv2
import numpy as np
import subprocess
import tempfile
import librosa
from librosa import feature as audio
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
LAV-DF
├── train
├── dev
├── test
├── metadata.min.json
└── README.md
"""

############ Custom parameter ##############
N_EXTRACT = 10      # max number of extracted images from one segment
WINDOW_LEN = 5      # frames of each window
IMAGE_SIZE = 500
MAX_REAL_IMAGES = 2000   # maximum number of real images
MAX_FAKE_IMAGES = 5000   # maximum number of fake images
MAX_VIDEO = None    # None means use all videos
TARGET_SPLIT = "test"  # Only process this split: "train", "test", or "dev"
RANDOM_SEED = 42    # Random seed for shuffling
NUM_THREADS = 4     # default number of threads

dataset_root = r"E:\data\LAV-DF"          # LAV-DF dataset root
output_root = r"E:\data\LAV-DF-window4"    # fixed output directory
metadata_file = ""                         # optional explicit metadata path
FFMPEG_EXE = "ffmpeg"
FFMPEG_TIMEOUT_SEC = 15
############################################

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

class AtomicImageCounter:
    def __init__(self, initial_value, max_value):
        self._value = int(initial_value)
        self._max_value = int(max_value)
        self._lock = threading.Lock()

    def has_remaining(self):
        with self._lock:
            return self._value < self._max_value

    def try_acquire(self):
        with self._lock:
            if self._value >= self._max_value:
                return False
            self._value += 1
            return True

    def release(self):
        with self._lock:
            if self._value > 0:
                self._value -= 1

    def get(self):
        with self._lock:
            return self._value

def extract_audio(video_file, audio_path):
    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error", "-i", video_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=FFMPEG_TIMEOUT_SEC)

def get_spectrogram(video_file):
    os.makedirs("./temp", exist_ok=True)
    temp_fd_wav, temp_wav = tempfile.mkstemp(prefix="temp_audio_", suffix=".wav", dir="./temp")
    temp_fd_img, temp_img = tempfile.mkstemp(prefix="temp_mel_", suffix=".png", dir="./temp")
    os.close(temp_fd_wav)
    os.close(temp_fd_img)
    try:
        extract_audio(video_file, temp_wav)
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
            return np.zeros((512, 1000, 4), dtype=np.uint8)

        data, sr = librosa.load(temp_wav, sr=None)
        if len(data) == 0:
             return np.zeros((512, 1000, 4), dtype=np.uint8)
             
        mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
        plt.imsave(temp_img, mel)
        mel_img = plt.imread(temp_img) * 255
        return mel_img.astype(np.uint8)
    finally:
        try:
            if os.path.exists(temp_wav): os.remove(temp_wav)
            if os.path.exists(temp_img): os.remove(temp_img)
        except OSError:
            pass

def get_start_frames(start_frame, end_frame):
    valid_num = end_frame - start_frame - WINDOW_LEN
    if valid_num < 1:
        return []
    extract_num = min(N_EXTRACT, valid_num)
    if extract_num == 1:
        return [start_frame]
    frame_idx = np.linspace(start_frame, end_frame - WINDOW_LEN - 1, extract_num, endpoint=True, dtype=np.int32).tolist()
    return sorted(set([int(f) for f in frame_idx]))

def read_needed_frames(video_file, frame_sequence):
    video_capture = cv2.VideoCapture(video_file)
    frame_set = set(frame_sequence)
    frame_map = {}
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret: break
        if current_frame in frame_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frame_map[current_frame] = frame
        current_frame += 1
    video_capture.release()
    return frame_map

def count_existing_images(save_dir):
    if not os.path.isdir(save_dir): return 0
    return sum(1 for _, _, files in os.walk(save_dir) for f in files if f.lower().endswith(".png"))

def extract_segment(video_file, save_dir, name, t0, t1, max_images=None, counter=None):
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

                sub_mel = cv2.resize(mel[:, begin:end], (IMAGE_SIZE * WINDOW_LEN, IMAGE_SIZE))
                x = np.concatenate(frames, axis=1)
                x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                if counter is not None:
                    if not counter.try_acquire():
                        break
                    reserved = True
                
                save_path = os.path.join(save_dir, f"{name}_{group}.png")
                plt.imsave(save_path, x.astype(np.uint8))
                
                group += 1
            except Exception as e:
                if reserved and counter is not None:
                    counter.release()
                continue

        return group
    except Exception as e:
        return 0

def process_fake_period(item, seg_id, period, data_root, meta_dict, fake_out_dir, real_out_dir, fake_counter, real_counter):
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

    # 1. Extract from Fake video
    if fake_counter.has_remaining():
        extract_segment(video_file, fake_out_dir, f"{fake_name}_seg{seg_id}", t0, t1, counter=fake_counter)

    # 2. Extract corresponding accurate paired segments from Original Real video
    if real_counter.has_remaining() and item["original"] is not None and item["original"] in meta_dict:
        real_item = meta_dict[item["original"]]
        real_file = os.path.join(data_root, real_item["file"])
        if os.path.exists(real_file):
            real_name = os.path.splitext(os.path.basename(real_item["file"]))[0]
            r0 = t0 / max(item["duration"], 1e-8) * real_item["duration"]
            r1 = t1 / max(item["duration"], 1e-8) * real_item["duration"]
            extract_segment(real_file, real_out_dir, f"{real_name}_pair_{fake_name}_seg{seg_id}", r0, r1, counter=real_counter)

def process_full_real_video(item, data_root, real_out_dir, real_counter):
    if not real_counter.has_remaining():
        return

    video_file = os.path.join(data_root, item["file"])
    if not os.path.exists(video_file):
        return

    name = os.path.splitext(os.path.basename(item["file"]))[0]
    extract_segment(video_file, real_out_dir, name, 0.0, item["duration"], counter=real_counter)

def run():
    data_root, meta_path = resolve_dataset_and_metadata(dataset_root, metadata_file)
    print(f"[Info] Setting Up... Saving to {TARGET_SPLIT} folder. Max {MAX_REAL_IMAGES} real / {MAX_FAKE_IMAGES} fake images.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_dict = {item["file"]: item for item in meta}
    real_list = [item for item in meta if item["n_fakes"] == 0 and item.get("split") == TARGET_SPLIT]
    fake_list = [item for item in meta if item["n_fakes"] > 0 and item.get("split") == TARGET_SPLIT]

    # Set random seed and shuffle lists
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

    print(f"[Info] existing real images: {saved_real}/{MAX_REAL_IMAGES}")
    print(f"[Info] existing fake images: {saved_fake}/{MAX_FAKE_IMAGES}")
    print(f"[Info] using {NUM_THREADS} threads")

    if fake_counter.has_remaining() or real_counter.has_remaining():
        print("Handling 1_fake (only fake periods) AND extracting their real pairs first...")
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
                        item,
                        seg_id,
                        period,
                        data_root,
                        meta_dict,
                        fake_out_dir,
                        real_out_dir,
                        fake_counter,
                        real_counter,
                    )
                    for item, seg_id, period in fake_tasks
                ]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="fake/pair segments"):
                    pass

    # Fill remaining Real images if limit is not reached yet
    if real_counter.has_remaining():
        print("Handling 0_real (full real videos) to fill remaining quota...")
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(process_full_real_video, item, data_root, real_out_dir, real_counter)
                for item in real_list
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="full real videos"):
                pass

    saved_real = real_counter.get()
    saved_fake = fake_counter.get()

    print(f"\n[Info] Finished. Extracted {saved_real} real images and {saved_fake} fake images.")

if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    try:
        run()
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
