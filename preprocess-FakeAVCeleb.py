import os
import cv2
import tempfile
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from librosa import feature as audio_feature
from moviepy import VideoFileClip
import matplotlib.pyplot as plt
import tempfile

r"""
FakeAVCeleb preprocessing script
改动重点：
1. 从 mp4 中自动提取临时 wav
2. 用临时 wav 生成 mel 频谱图
3. 再与视频帧拼接成 PNG
4. 处理完成后默认删除临时 wav
"""

############ Custom parameters ##############
DATASET_ROOT = r"E:\data\FakeAVCeleb\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
OUTPUT_ROOT = r"E:\data\FakeAVCeleb-test"
METADATA_CSV = os.path.join(DATASET_ROOT, "meta_data.csv")

N_EXTRACT = 10       # number of temporal windows per video
WINDOW_LEN = 5       # frames per window
IMG_SIZE = 500       # resize each frame to IMG_SIZE x IMG_SIZE
AUDIO_SR = 16000     # unified audio sampling rate
OVERWRITE = False    # skip already processed videos when False

# mel parameters
MEL_N_FFT = 400
MEL_HOP_LENGTH = 160
MEL_N_MELS = 128

# temp wav settings
TEMP_AUDIO_DIR = os.path.join(OUTPUT_ROOT, "_temp_audio")
DELETE_TEMP_AUDIO = True
#############################################


# 50 predefined source IDs
SOURCE_SPLIT = {
    "African": {
        "men": ["id00478", "id00781", "id00987", "id01170", "id01179"],
        "women": ["id02301", "id02586", "id04055", "id04736", "id04939"],
    },
    "Asian (East)": {
        "men": ["id00056", "id00126", "id01683", "id02553", "id04726"],
        "women": ["id00579", "id02807", "id06054", "id06060", "id06065"],
    },
    "Asian (South)": {
        "men": ["id00745", "id00816", "id07163", "id07210", "id07463"],
        "women": ["id00461", "id04070", "id04490", "id04564", "id06437"],
    },
    "Caucasian (American)": {
        "men": ["id00179", "id00243", "id01096", "id01201", "id03668"],
        "women": ["id00100", "id00398", "id00431", "id01091", "id01217"],
    },
    "Caucasian (European)": {
        "men": ["id00266", "id00946", "id00999", "id01058", "id01126"],
        "women": ["id00270", "id00325", "id00495", "id00633", "id00823"],
    },
}

TYPE_TO_SHORT = {
    "RealVideo-RealAudio": "RR",
    "RealVideo-FakeAudio": "RF",
    "FakeVideo-RealAudio": "FR",
    "FakeVideo-FakeAudio": "FF",
}

TYPE_ORDER = [
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize FakeAVCeleb metadata columns.

    常见情况：
    - 'path' 实际上是 filename
    - 最后一列 unnamed 是 folder 路径
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "folder" not in df.columns:
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed_cols:
            df.rename(columns={unnamed_cols[-1]: "folder"}, inplace=True)
        else:
            df["folder"] = ""

    if "filename" not in df.columns and "path" in df.columns:
        df.rename(columns={"path": "filename"}, inplace=True)

    required = ["source", "method", "category", "type", "race", "gender", "filename", "folder"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    for c in required:
        df[c] = df[c].astype(str).str.strip()

    return df


def resolve_video_path(dataset_root: str, folder: str, filename: str) -> str:
    """
    Resolve absolute path robustly across slightly different folder prefixes.
    """
    candidates = []
    folder_norm = str(folder).replace("/", os.sep).replace("\\", os.sep).strip(os.sep)
    filename = str(filename).strip()

    if folder_norm:
        candidates.append(os.path.join(dataset_root, folder_norm, filename))

        if folder_norm.startswith(f"FakeAVCeleb{os.sep}"):
            candidates.append(os.path.join(dataset_root, folder_norm.split(os.sep, 1)[1], filename))

        if folder_norm.startswith("FakeAVCeleb/"):
            candidates.append(os.path.join(dataset_root, folder_norm[len("FakeAVCeleb/"):], filename))

    candidates.append(os.path.join(dataset_root, filename))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Cannot resolve video path for folder={folder!r}, filename={filename!r}")


def extract_audio_to_temp_wav(video_path: str, sr: int = AUDIO_SR) -> str:
    """
    从 mp4 中自动提取音频，保存为临时 wav。
    统一为：
    - mono
    - 16kHz
    """
    ensure_dir(TEMP_AUDIO_DIR)

    fd, temp_wav = tempfile.mkstemp(suffix=".wav", dir=TEMP_AUDIO_DIR)
    os.close(fd)

    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            raise ValueError(f"No audio stream found in video: {video_path}")

        clip.audio.write_audiofile(
            temp_wav,
            fps=sr,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],   # force mono
            logger=None
        )
    finally:
        try:
            if clip.audio is not None:
                clip.audio.close()
        except Exception:
            pass
        clip.close()

    return temp_wav



def get_mel_image_from_wav(wav_path: str, sr: int = AUDIO_SR) -> np.ndarray:
    """
    按你原始参考代码的方式生成 mel 图：
    1. librosa.load
    2. power_to_db(..., ref=np.min)
    3. plt.imsave 临时 mel.png
    4. plt.imread 再读回来
    """
    data, sr = librosa.load(wav_path, sr=sr, mono=True)

    if data is None or len(data) == 0:
        raise ValueError(f"Empty audio after extraction: {wav_path}")

    mel = librosa.power_to_db(
        audio_feature.melspectrogram(y=data, sr=sr),
        ref=np.min
    )

    fd, temp_png = tempfile.mkstemp(suffix=".png", dir=TEMP_AUDIO_DIR)
    os.close(fd)

    try:
        plt.imsave(temp_png, mel)
        mel_img = plt.imread(temp_png) * 255
        mel_img = mel_img.astype(np.uint8)

        # 有些情况下读出来是 RGBA，这里保留前3通道
        if mel_img.ndim == 3 and mel_img.shape[2] >= 3:
            mel_img = mel_img[:, :, :3]
        else:
            # 防御性处理：如果是灰度，就补成3通道
            mel_img = np.stack([mel_img] * 3, axis=-1)

    finally:
        safe_remove(temp_png)

    return mel_img


def get_mel_image_from_video(video_path: str, sr: int = AUDIO_SR) -> np.ndarray:
    """
    mp4 -> 临时 wav -> mel 图
    """
    temp_wav = extract_audio_to_temp_wav(video_path, sr=sr)
    try:
        mel = get_mel_image_from_wav(temp_wav, sr=sr)
    finally:
        if DELETE_TEMP_AUDIO:
            safe_remove(temp_wav)
    return mel

def select_frame_sequence(frame_count: int, n_extract: int, window_len: int):
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")

    if frame_count <= window_len:
        starts = [0] * n_extract
    else:
        starts = np.linspace(
            0,
            max(frame_count - window_len - 1, 0),
            n_extract,
            endpoint=True,
            dtype=np.int32,
        ).tolist()

    starts.sort()
    seq = [i for num in starts for i in range(num, num + window_len)]
    return starts, seq


def build_sample_plan(metadata_csv: str, dataset_root: str) -> pd.DataFrame:
    """
    Build 200-video plan:
    - RR: unique one
    - RF: unique one
    - FR: sorted first
    - FF: sorted first
    """
    df = pd.read_csv(metadata_csv)
    df = normalize_metadata(df)

    records = []
    sample_idx = 1

    for race, gender_map in SOURCE_SPLIT.items():
        for gender, source_ids in gender_map.items():
            for source_id in source_ids:
                for original_type in TYPE_ORDER:
                    rows = df[
                        (df["source"] == source_id)
                        & (df["race"] == race)
                        & (df["gender"] == gender)
                        & (df["type"] == original_type)
                    ].copy()

                    if rows.empty:
                        raise ValueError(
                            f"No metadata row found for source={source_id}, race={race}, "
                            f"gender={gender}, type={original_type}"
                        )

                    rows["_sort_key"] = rows["filename"].astype(str) + "|" + rows["folder"].astype(str)
                    rows = rows.sort_values("_sort_key").reset_index(drop=True)
                    chosen = rows.iloc[0]

                    video_path = resolve_video_path(dataset_root, chosen["folder"], chosen["filename"])
                    short_type = TYPE_TO_SHORT[original_type]
                    label = "real" if original_type == "RealVideo-RealAudio" else "fake"
                    sample_name = f"{race}_{gender}_{source_id}_{short_type}".replace(" ", "_")
                    output_frame_dir = os.path.join(OUTPUT_ROOT, label, sample_name)

                    records.append(
                        {
                            "sample_id": f"S{sample_idx:04d}",
                            "label": label,
                            "original_type": original_type,
                            "type_short": short_type,
                            "audio_label": "fake" if original_type in {"RealVideo-FakeAudio", "FakeVideo-FakeAudio"} else "real",
                            "video_label": "fake" if original_type in {"FakeVideo-RealAudio", "FakeVideo-FakeAudio"} else "real",
                            "race": race,
                            "gender": gender,
                            "source_id": source_id,
                            "method": chosen["method"],
                            "category": chosen["category"],
                            "selected_rule": "unique" if original_type in {"RealVideo-RealAudio", "RealVideo-FakeAudio"} else "sorted_first",
                            "video_path": video_path,
                            "output_frame_dir": output_frame_dir,
                            "filename": chosen["filename"],
                            "folder": chosen["folder"],
                        }
                    )
                    sample_idx += 1

    return pd.DataFrame(records)


def process_one_video(video_path: str, save_dir: str):
    """
    核心逻辑不变：
    - 取视频帧
    - 提取音频生成 mel
    - mel + 5帧 拼接
    - 保存 group_xxx.png
    """
    ensure_dir(save_dir)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame_sequence = select_frame_sequence(frame_count, N_EXTRACT, WINDOW_LEN)

    frame_list = []
    current_frame = 0
    last_needed = frame_sequence[-1] if frame_sequence else -1

    while current_frame <= last_needed:
        ret, frame = video_capture.read()
        if not ret:
            break

        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_list.append(frame)

        current_frame += 1

    video_capture.release()

    if len(frame_list) < WINDOW_LEN:
        raise RuntimeError(f"Not enough frames extracted from {video_path}")

    # ===== 这里是真正的“mp4 -> 临时wav -> mel” =====
    mel = get_mel_image_from_video(video_path)

    mapping = mel.shape[1] / max(frame_count, 1)

    saved_files = []
    group = 0

    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        if idx != 0:
            continue

        try:
            begin = int(np.round(frame_sequence[i] * mapping))
            end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))

            begin = max(0, min(begin, mel.shape[1] - 1))
            end = max(begin + 1, min(end, mel.shape[1]))

            sub_mel = cv2.resize(mel[:, begin:end], (IMG_SIZE * WINDOW_LEN, IMG_SIZE))

            x = np.concatenate(frame_list[i:i + WINDOW_LEN], axis=1)
            x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

            out_path = os.path.join(save_dir, f"group_{group:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
            saved_files.append(out_path)
            group += 1

        except Exception as e:
            print(f"[WARN] Failed to build window for {video_path}: {e}")
            continue

    return saved_files


def run():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(os.path.join(OUTPUT_ROOT, "real"))
    ensure_dir(os.path.join(OUTPUT_ROOT, "fake"))
    ensure_dir(TEMP_AUDIO_DIR)

    manifest = build_sample_plan(METADATA_CSV, DATASET_ROOT)

    manifest_path = os.path.join(OUTPUT_ROOT, "test_manifest.csv")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Manifest saved to: {manifest_path}")
    print(f"[INFO] Planned samples: {len(manifest)}")
    print("[INFO] Label counts:", manifest["label"].value_counts().to_dict())
    print("[INFO] Type counts:", manifest["original_type"].value_counts().to_dict())

    results = []
    for row in tqdm(manifest.to_dict(orient="records"), total=len(manifest), desc="Processing FakeAVCeleb"):
        save_dir = row["output_frame_dir"]

        if os.path.isdir(save_dir) and (not OVERWRITE):
            existing_png = [f for f in os.listdir(save_dir) if f.lower().endswith(".png")]
            if existing_png:
                results.append({**row, "status": "skipped_existing", "num_png": len(existing_png)})
                continue

        try:
            saved_files = process_one_video(row["video_path"], save_dir)
            results.append({**row, "status": "ok", "num_png": len(saved_files)})
        except Exception as e:
            results.append({**row, "status": f"error: {e}", "num_png": 0})
            print(f"[ERROR] {row['video_path']} -> {e}")

    result_df = pd.DataFrame(results)
    result_csv = os.path.join(OUTPUT_ROOT, "test_manifest_with_status.csv")
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Processing summary saved to: {result_csv}")

    summary = (
        result_df.groupby(["label", "original_type", "race"])
        .agg(
            videos=("sample_id", "count"),
            ok=("status", lambda s: int(sum(str(x) in {"ok", "skipped_existing"} for x in s))),
            pngs=("num_png", "sum"),
        )
        .reset_index()
    )
    summary_path = os.path.join(OUTPUT_ROOT, "summary_counts.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    run()