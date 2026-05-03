import os
import argparse
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio


"""
Structure of the AVLips dataset:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

############ Custom parameter ##############
N_EXTRACT = 10   # number of extracted images from video
WINDOW_LEN = 5   # frames of each window
MAX_SAMPLE = 100 

audio_root = "./AVLips/wav"
video_root = "./AVLips"
output_root = "./datasets/AVLips"
############################################

labels = [(0, "0_real"), (1, "1_fake")]

def get_spectrogram(audio_file, temp_dir):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    mel_path = os.path.join(temp_dir, "mel.png")
    plt.imsave(mel_path, mel)
    return mel_path


def infer_label_name(video_file, audio_file, label):
    if label in {"0_real", "real"}:
        return "0_real"
    if label in {"1_fake", "fake"}:
        return "1_fake"

    path_parts = []
    for path in (video_file, audio_file):
        normalized = os.path.normpath(str(path))
        path_parts.extend(part.lower() for part in normalized.split(os.sep))

    if "0_real" in path_parts or "real" in path_parts:
        return "0_real"
    if "1_fake" in path_parts or "fake" in path_parts:
        return "1_fake"

    raise ValueError("Single-file mode requires --label 0_real or --label 1_fake when label cannot be inferred.")


def process_video_file(video_path, audio_path, output_label_dir, args):
    video_name = os.path.basename(video_path)
    name = os.path.splitext(video_name)[0]
    video_output_dir = str(getattr(args, "output_video_dir", "") or "").strip()
    if not video_output_dir:
        video_output_dir = os.path.join(output_label_dir, name)

    if not os.path.isfile(video_path):
        print(f"Skip missing video: {video_path}")
        return 0, video_output_dir
    if not os.path.isfile(audio_path):
        print(f"Skip missing audio: {audio_path}")
        return 0, video_output_dir

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Skip unreadable video: {video_path}")
        return 0, video_output_dir

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= args.window_len:
        print(f"Skip short video {video_name}: frame_count={frame_count}")
        video_capture.release()
        return 0, video_output_dir

    frame_idx = np.linspace(
        0,
        frame_count - args.window_len - 1,
        args.n_extract,
        endpoint=True,
        dtype=int,
    ).tolist()
    frame_idx.sort()
    frame_sequence = [
        i for num in frame_idx for i in range(num, num + args.window_len)
    ]
    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error in reading frame {video_name}: {current_frame}")
            break
        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (500, 500)))
        current_frame += 1
    video_capture.release()

    if len(frame_list) < args.window_len:
        print(
            f"Skip {video_name}: only read {len(frame_list)} selected frame(s), "
            f"need at least {args.window_len}."
        )
        return 0, video_output_dir

    os.makedirs(video_output_dir, exist_ok=True)

    group = 0
    mel_path = get_spectrogram(audio_path, args.temp_dir)
    mel = plt.imread(mel_path) * 255
    mel = mel.astype(np.uint8)
    mapping = mel.shape[1] / frame_count
    for frame_pos in range(len(frame_list)):
        idx = frame_pos % args.window_len
        if idx == 0:
            try:
                begin = np.round(frame_sequence[frame_pos] * mapping)
                end = np.round((frame_sequence[frame_pos] + args.window_len) * mapping)
                sub_mel = cv2.resize(
                    (mel[:, int(begin) : int(end)]), (500 * args.window_len, 500)
                )
                x = np.concatenate(frame_list[frame_pos : frame_pos + args.window_len], axis=1)
                x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
                plt.imsave(
                    os.path.join(video_output_dir, f"{name}_{group}.png"), x
                )
                group = group + 1
            except ValueError:
                print(f"ValueError: {name}")
                continue

    return group, video_output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess AVLips videos into image windows."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=video_root,
        help="Video dataset root containing 0_real and 1_fake.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default=audio_root,
        help="Audio dataset root containing 0_real and 1_fake wav files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=output_root,
        help="Output root. Images are saved as <output_root>/<label>/<video_name>/<video_name>_<idx>.png.",
    )
    parser.add_argument(
        "--video_file",
        type=str,
        default="",
        help="Optional single mp4 file to process. If set, --audio_file is required.",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="",
        help="Optional single wav file paired with --video_file.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        choices=["", "0_real", "1_fake", "real", "fake"],
        help="Label folder for single-file mode. Auto-inferred from path when omitted.",
    )
    parser.add_argument("--n_extract", type=int, default=N_EXTRACT)
    parser.add_argument("--window_len", type=int, default=WINDOW_LEN)
    parser.add_argument(
        "--max_sample",
        type=int,
        default=MAX_SAMPLE,
        help="Maximum number of label folders to process. Use -1 for all labels.",
    )
    parser.add_argument("--temp_dir", type=str, default="./temp")
    return parser.parse_args()


def run(args):
    if args.video_file or args.audio_file:
        if not args.video_file or not args.audio_file:
            raise ValueError("--video_file and --audio_file must be provided together.")
        dataset_name = infer_label_name(args.video_file, args.audio_file, args.label)
        output_label_dir = os.path.join(args.output_root, dataset_name)
        os.makedirs(output_label_dir, exist_ok=True)
        print(f"Handling single file as {dataset_name}...")
        count, video_output_dir = process_video_file(args.video_file, args.audio_file, output_label_dir, args)
        print(f"Saved {count} image window(s) to {video_output_dir}")
        return

    label_count = 0
    for label, dataset_name in labels:
        output_label_dir = os.path.join(args.output_root, dataset_name)
        os.makedirs(output_label_dir, exist_ok=True)

        if args.max_sample >= 0 and label_count == args.max_sample:
            break
        root = os.path.join(args.video_root, dataset_name)
        if not os.path.isdir(root):
            print(f"Skip missing video folder: {root}")
            label_count += 1
            continue
        video_list = os.listdir(root)
        print(f"Handling {dataset_name}...")
        for j in tqdm(range(len(video_list))):
            v = video_list[j]
            video_path = os.path.join(root, v)
            if not os.path.isfile(video_path):
                continue
            name = os.path.splitext(v)[0]
            a = os.path.join(args.audio_root, dataset_name, f"{name}.wav")
            process_video_file(video_path, a, output_label_dir, args)
        label_count += 1


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    run(args)
