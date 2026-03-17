import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# Keep preprocessing stable on environments where numba JIT warm-up can be slow.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except Exception:
    pass


MANIFEST_FIELDS = [
    "split",
    "video_id",
    "video_name",
    "video_rel_path",
    "video_label",
    "window_label",
    "window_source",
    "frame_start",
    "frame_end_exclusive",
    "time_start_sec",
    "time_end_sec",
    "overlap_ratio",
    "modify_audio",
    "modify_video",
    "policy_tag",
    "n_fakes",
    "status",
    "output_path",
]

FAILED_FIELDS = [
    "split",
    "video_id",
    "video_name",
    "video_rel_path",
    "video_label",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="LAV-DF window-level preprocessing using fake_periods relabeling.",
    )

    parser.add_argument("--dataset_root", type=str, default=r"E:\\data\\LAV-DF\\LAV-DF")
    parser.add_argument("--metadata_path", type=str, default="", help="Default: <dataset_root>/metadata.min.json")
    parser.add_argument("--output_root", type=str, default=r"E:\\data\\LAV-DF-window")
    parser.add_argument("--splits", type=str, default="test", help="Comma-separated splits to process.")

    parser.add_argument("--window_len", type=int, default=5)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=500)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--mel_hop_length",
        type=int,
        default=160,
        help="Hop length used for log-mel extraction and strict time alignment.",
    )

    parser.add_argument("--inside_thr", type=float, default=0.8, help="overlap_ratio >= inside_thr -> fake")
    parser.add_argument("--outside_thr", type=float, default=0.0, help="overlap_ratio <= outside_thr -> real")

    parser.add_argument("--inside_per_fake", type=int, default=5)
    parser.add_argument("--outside_per_fake", type=int, default=5)
    parser.add_argument("--real_per_video", type=int, default=5)
    parser.add_argument(
        "--audio_conflict_policy",
        type=str,
        default="drop_outside",
        choices=["drop_outside", "keep_as_real"],
        help="How to handle outside windows when modify_audio=true.",
    )
    parser.add_argument(
        "--audio_only_policy",
        type=str,
        default="extract_inside",
        choices=["extract_inside", "drop_all"],
        help="How to handle modify_video=false and modify_audio=true fake videos.",
    )
    parser.add_argument(
        "--boundary_margin",
        type=float,
        default=0.15,
        help="Boundary safety margin in seconds. Boundary windows are dropped.",
    )

    parser.add_argument("--max_real_videos", type=int, default=100, help="Per split. 0 means no limit.")
    parser.add_argument("--max_fake_videos", type=int, default=100, help="Per split. 0 means no limit.")
    parser.add_argument("--video_workers", type=int, default=4, help="How many videos to process in parallel.")

    parser.add_argument("--shuffle_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ffmpeg_exe", type=str, default="", help="Optional explicit ffmpeg path.")
    parser.add_argument("--ffmpeg_timeout_sec", type=int, default=15, help="Timeout for ffmpeg extraction.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean_output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--png_compression",
        type=int,
        default=6,
        help="PNG compression level [0,9]. 0 is fastest/largest, 6 is balanced, 9 is smallest/slowest.",
    )
    parser.add_argument(
        "--stop_on_interrupt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop whole run on KeyboardInterrupt. Use --no-stop_on_interrupt to skip current video and continue.",
    )

    return parser.parse_args()


def parse_split_list(text):
    values = []
    for token in str(text).split(","):
        t = token.strip().lower()
        if t:
            values.append(t)
    return sorted(set(values))


def normalize_path_text(path_text):
    path_text = str(path_text or "").strip().strip('"').strip("'")
    path_text = path_text.replace("\\", "/")
    while "//" in path_text:
        path_text = path_text.replace("//", "/")
    return path_text


def safe_video_label(original_value):
    if original_value is None:
        return 0
    text = str(original_value).strip().lower()
    if text in {"", "none", "null"}:
        return 0
    return 1


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_rgb_png(path, image_rgb, compression=0):
    comp = int(np.clip(int(compression), 0, 9))
    img = np.asarray(image_rgb)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, comp])
    if not ok:
        raise RuntimeError(f"cv2.imwrite returned False: {path}")


def load_metadata_items(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("metadata json should be a list of dicts")
    return items


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


def fake_periods_to_frame_intervals(fake_periods, duration, frame_count):
    intervals = []
    if frame_count <= 0:
        return intervals

    duration = float(duration or 0.0)
    if duration <= 0.0:
        return intervals

    for pair in (fake_periods or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue

        try:
            start_sec = float(pair[0])
            end_sec = float(pair[1])
        except Exception:
            continue

        if not np.isfinite(start_sec) or not np.isfinite(end_sec):
            continue

        start_sec = max(0.0, start_sec)
        end_sec = min(duration, end_sec)
        if end_sec <= start_sec:
            continue

        start_frame = int(np.floor((start_sec / duration) * frame_count))
        end_frame = int(np.ceil((end_sec / duration) * frame_count))

        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(start_frame + 1, min(end_frame, frame_count))

        intervals.append((start_frame, end_frame))

    return merge_intervals(intervals)


def overlap_ratio_for_window(frame_start, window_len, intervals):
    frame_end = frame_start + window_len
    overlap = 0
    for s, e in intervals:
        if e <= frame_start:
            continue
        if s >= frame_end:
            break
        overlap += max(0, min(frame_end, e) - max(frame_start, s))
    return float(overlap) / float(max(window_len, 1))


def classify_window_position(frame_start, window_len, intervals, margin_frames=0):
    frame_start = int(frame_start)
    frame_end = int(frame_start + window_len)
    margin_frames = int(max(0, margin_frames))

    if not intervals:
        return "outside"

    # Strict inside: window is fully contained in a margin-shrunk fake interval.
    for s, e in intervals:
        if e <= s:
            continue
        # Keep at least one frame span for short fake intervals.
        interval_len = int(e - s)
        effective_margin = int(min(margin_frames, max((interval_len - 1) // 2, 0)))
        s_in = int(s + effective_margin)
        e_in = int(e - effective_margin)
        if frame_start >= s_in and frame_end <= e_in:
            return "inside"

    # Strict outside: window is outside all margin-expanded fake intervals.
    strictly_outside = True
    for s, e in intervals:
        s_out = int(s - margin_frames)
        e_out = int(e + margin_frames)
        if not (frame_end <= s_out or frame_start >= e_out):
            strictly_outside = False
            break

    if strictly_outside:
        return "outside"
    return "boundary"


def build_candidate_starts(frame_count, window_len, frame_stride):
    if frame_count <= 0:
        return []

    max_start = max(0, frame_count - window_len)
    stride = max(1, int(frame_stride))

    starts = list(range(0, max_start + 1, stride))
    if not starts:
        starts = [0]
    elif starts[-1] != max_start:
        starts.append(max_start)

    return sorted(set(int(x) for x in starts))


def select_evenly(candidates, k):
    candidates = sorted(set(int(x) for x in candidates))
    if k <= 0 or len(candidates) == 0:
        return []
    if len(candidates) <= k:
        return candidates

    idx = np.linspace(0, len(candidates) - 1, num=k, dtype=np.int64)
    selected = [candidates[int(i)] for i in idx]
    selected = list(dict.fromkeys(selected))

    if len(selected) < k:
        for x in candidates:
            if x not in selected:
                selected.append(x)
            if len(selected) >= k:
                break

    return sorted(selected[:k])


def interleave_labeled_videos(real_items, fake_items):
    """
    Interleave fake/real videos so partial runs still produce both classes.
    Fake is placed first to avoid early-stop runs ending with only real outputs.
    Example order: fake0, real0, fake1, real1, ...
    """
    ordered = []
    n = max(len(real_items), len(fake_items))
    for i in range(n):
        if i < len(fake_items):
            ordered.append((1, fake_items[i]))
        if i < len(real_items):
            ordered.append((0, real_items[i]))
    return ordered


def validate_ffmpeg(ffmpeg_path):
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_ffmpeg_exe(manual_path=""):
    candidates = []

    def add(path_like):
        if not path_like:
            return
        candidate = str(path_like).strip().strip('"').strip("'")
        if candidate:
            candidates.append(candidate)

    add(manual_path)
    add(os.environ.get("FFMPEG_EXE", ""))
    add(os.environ.get("IMAGEIO_FFMPEG_EXE", ""))
    add(shutil.which("ffmpeg"))
    add(shutil.which("ffmpeg.exe"))

    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        user_profile = os.environ.get("USERPROFILE", "")
        local_app_data = os.environ.get("LOCALAPPDATA", "")

        win_candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.join(program_files, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(program_files_x86, "ffmpeg", "bin", "ffmpeg.exe"),
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            os.path.join(user_profile, "scoop", "shims", "ffmpeg.exe"),
            os.path.join(user_profile, "scoop", "apps", "ffmpeg", "current", "bin", "ffmpeg.exe"),
            os.path.join(local_app_data, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        ]
        for p in win_candidates:
            add(p)

        winget_pkg_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg_root.exists():
            for exe in winget_pkg_root.rglob("ffmpeg.exe"):
                add(str(exe))

    checked = set()
    for exe in candidates:
        if exe in checked:
            continue
        checked.add(exe)
        if validate_ffmpeg(exe):
            return exe

    raise RuntimeError(
        "Cannot find a valid ffmpeg executable. Please set --ffmpeg_exe or add ffmpeg to PATH."
    )


def extract_audio_waveform_from_video(video_path, ffmpeg_exe, sample_rate, timeout_sec=15):
    cmd = [
        ffmpeg_exe,
        "-i",
        str(video_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-loglevel",
        "error",
        "-",
    ]

    timeout_sec = max(1.0, float(timeout_sec))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg audio extraction timed out after {timeout_sec:.1f}s: {video_path}")

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg audio extraction failed: {video_path}\\n{err}")

    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"No audio extracted from video: {video_path}")

    return waveform


def _hz_to_mel(freq_hz):
    return 2595.0 * np.log10(1.0 + (np.asarray(freq_hz, dtype=np.float64) / 700.0))


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _build_mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = float(sr) / 2.0
    if not (0.0 <= fmin < fmax):
        raise ValueError(f"Invalid mel fmin/fmax: {fmin}, {fmax}")

    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / float(sr)).astype(np.int32)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)
        if right <= left:
            continue

        for k in range(left, center):
            denom = float(max(center - left, 1))
            fb[m - 1, k] = (k - left) / denom
        for k in range(center, right):
            denom = float(max(right - center, 1))
            fb[m - 1, k] = (right - k) / denom

    return fb


def _waveform_to_log_mel(waveform, sr, n_fft=1024, hop_length=160, n_mels=128):
    y = np.asarray(waveform, dtype=np.float32).flatten()
    if y.size == 0:
        raise RuntimeError("Empty waveform.")

    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    window = np.hanning(n_fft).astype(np.float32)

    n_frames = 1 + max(0, (y.size - n_fft) // hop_length)
    if n_frames <= 0:
        n_frames = 1

    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = y[start:end]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        frame = frame * window
        fft = np.fft.rfft(frame, n=n_fft)
        power = (fft.real.astype(np.float32) ** 2) + (fft.imag.astype(np.float32) ** 2)
        spec[:, i] = power

    mel_fb = _build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = mel_fb @ spec
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = 10.0 * np.log10(mel_spec)
    return log_mel


def build_mel_image_from_video(video_path, ffmpeg_exe, sample_rate, hop_length, ffmpeg_timeout_sec, temp_mel_path):
    # temp_mel_path is kept for interface compatibility but no longer needed.
    _ = temp_mel_path
    waveform = extract_audio_waveform_from_video(
        video_path, ffmpeg_exe, sample_rate, timeout_sec=ffmpeg_timeout_sec
    )
    log_mel = _waveform_to_log_mel(
        waveform, sr=sample_rate, n_fft=1024, hop_length=int(hop_length), n_mels=128
    )

    mel_min = float(np.min(log_mel))
    mel_max = float(np.max(log_mel))
    if mel_max <= mel_min:
        mel_norm = np.zeros_like(log_mel, dtype=np.float32)
    else:
        mel_norm = (log_mel - mel_min) / (mel_max - mel_min)

    mel_gray = np.clip(mel_norm * 255.0, 0, 255).astype(np.uint8)
    mel_bgr = cv2.applyColorMap(mel_gray, cv2.COLORMAP_VIRIDIS)
    mel_rgb = cv2.cvtColor(mel_bgr, cv2.COLOR_BGR2RGB)
    return mel_rgb


def read_selected_frames(video_path, needed_indices, img_size):
    needed = sorted(set(int(x) for x in needed_indices))
    if not needed:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_map = {}
    current_idx = 0
    last_needed = needed[-1]
    needed_set = set(needed)

    while current_idx <= last_needed:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in needed_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            frame_map[current_idx] = frame

        current_idx += 1

    cap.release()
    return frame_map


def build_window_image(frame_start, window_len, frame_map, mel_img, fps, sample_rate, hop_length, img_size):
    mel_w = int(mel_img.shape[1])
    if mel_w <= 0:
        raise RuntimeError("mel image width is 0")
    if fps <= 0:
        fps = 25.0

    # 1) Build video strip first (bottom half)
    frame_list = []
    fallback = None
    for idx in range(frame_start, frame_start + window_len):
        frame = frame_map.get(idx)
        if frame is None:
            frame = fallback.copy() if fallback is not None else np.zeros((img_size, img_size, 4), dtype=np.uint8)
        else:
            fallback = frame
        frame_list.append(frame[:, :, :3])
    strip = np.concatenate(frame_list, axis=1)

    # 2) Compute mel slice by absolute time
    time_start_sec = float(frame_start) / float(max(fps, 1e-6))
    time_end_sec = float(frame_start + window_len) / float(max(fps, 1e-6))
    mel_fps = float(sample_rate) / float(max(int(hop_length), 1))

    begin = int(np.round(time_start_sec * mel_fps))
    end = int(np.round(time_end_sec * mel_fps))

    # Guard against invalid ranges
    begin = max(0, min(begin, mel_w - 1))
    end = max(begin + 1, min(end, mel_w))
    sub_mel = mel_img[:, begin:end, :3]

    # Extra safety for edge cases where slicing still yields an empty array.
    if sub_mel.shape[0] <= 0 or sub_mel.shape[1] <= 0:
        sub_mel = np.zeros((128, 1, 3), dtype=np.uint8)

    # 3) Force spatial alignment to video strip width using nearest-neighbor.
    # This preserves hard mel band transitions and avoids smoothing artifacts.
    target_w = int(img_size * window_len)
    sub_mel_final = cv2.resize(sub_mel, (target_w, int(img_size)), interpolation=cv2.INTER_NEAREST)

    # 4) Stack mel (top) + frames (bottom)
    merged = np.concatenate((sub_mel_final, strip), axis=0)
    return merged


def build_video_candidates(meta_item, frame_count, fps, window_len, frame_stride, inside_thr, outside_thr, boundary_margin):
    video_label = safe_video_label(meta_item.get("original", None))
    duration = float(meta_item.get("duration", 0.0) or 0.0)
    fake_periods = meta_item.get("fake_periods", []) if video_label == 1 else []

    fake_intervals = fake_periods_to_frame_intervals(fake_periods, duration, frame_count)

    starts = build_candidate_starts(frame_count, window_len, frame_stride)
    margin_frames = int(np.round(max(0.0, float(boundary_margin)) * float(max(fps, 1e-6))))

    inside = []
    outside = []
    boundary = []
    ambiguous = []
    ratio_map = {}

    for s in starts:
        ratio = overlap_ratio_for_window(s, window_len, fake_intervals)
        ratio_map[s] = ratio

        if ratio >= inside_thr:
            inside.append(s)
        else:
            pos = classify_window_position(s, window_len, fake_intervals, margin_frames=margin_frames)
            if ratio <= outside_thr and pos == "outside":
                outside.append(s)
            else:
                boundary.append(s)
                ambiguous.append(s)

    return {
        "video_label": video_label,
        "starts": starts,
        "inside": inside,
        "outside": outside,
        "boundary": boundary,
        "ambiguous": ambiguous,
        "ratio_map": ratio_map,
        "fake_intervals": fake_intervals,
        "margin_frames": margin_frames,
    }


def prepare_output_layout(output_root, split, clean_output=False):
    split_root = Path(output_root) / split
    dirs = {
        "real_from_real": split_root / "0_real" / "from_real_video",
        "real_from_fake_clean": split_root / "0_real" / "from_fake_video_clean",
        "fake_from_fake_tampered": split_root / "1_fake" / "from_fake_video_tampered",
        "fake_from_fake_audio_only": split_root / "1_fake" / "from_fake_video_audio_only",
    }

    if clean_output and split_root.exists():
        shutil.rmtree(split_root)

    for d in dirs.values():
        ensure_dir(d)

    return split_root, dirs


def process_one_video_item(
    split,
    video_label,
    item,
    dataset_root,
    split_dirs,
    ffmpeg_exe,
    args,
):
    split_manifest_rows = []
    failed_row = None

    stats_delta = {
        "videos_ok": 0,
        "videos_failed": 0,
        "videos_no_selected_windows": 0,
        "candidate_windows_total": 0,
        "ambiguous_windows_dropped": 0,
        "dropped_boundary": 0,
        "dropped_outside_due_audio": 0,
        "dropped_audio_only_videos": 0,
        "saved_real_from_real": 0,
        "saved_real_from_fake_clean": 0,
        "saved_fake_from_fake_tampered": 0,
        "saved_fake_from_fake_audio_only": 0,
        "skipped_existing": 0,
    }

    rel_path = normalize_path_text(item.get("file", ""))
    video_name = Path(rel_path).name
    video_id = Path(rel_path).stem
    video_path = (dataset_root / rel_path).resolve()
    modify_video = to_bool(item.get("modify_video", False))
    modify_audio = to_bool(item.get("modify_audio", False))
    policy_tag = f"mv{int(modify_video)}_ma{int(modify_audio)}"

    def fail(err_text):
        nonlocal failed_row
        failed_row = {
            "split": split,
            "video_id": video_id,
            "video_name": video_name,
            "video_rel_path": rel_path,
            "video_label": int(video_label),
            "error": str(err_text),
        }
        stats_delta["videos_failed"] += 1

    if not rel_path or not video_path.exists():
        fail("video_not_found")
        return split_manifest_rows, failed_row, stats_delta

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail("cannot_open_video")
        return split_manifest_rows, failed_row, stats_delta

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frame_count <= 0:
        fail("invalid_frame_count")
        return split_manifest_rows, failed_row, stats_delta

    if fps <= 0:
        fps = 25.0

    try:
        candidates = build_video_candidates(
            meta_item=item,
            frame_count=frame_count,
            fps=fps,
            window_len=args.window_len,
            frame_stride=args.frame_stride,
            inside_thr=args.inside_thr,
            outside_thr=args.outside_thr,
            boundary_margin=args.boundary_margin,
        )

        ratio_map = candidates["ratio_map"]
        inside = candidates["inside"]
        outside = candidates["outside"]
        boundary = candidates["boundary"]

        stats_delta["candidate_windows_total"] += int(len(candidates["starts"]))
        stats_delta["ambiguous_windows_dropped"] += int(len(boundary))
        stats_delta["dropped_boundary"] += int(len(boundary))

        selected_specs = {}
        policy_tag_curr = policy_tag
        if video_label == 0:
            selected_outside = select_evenly(outside, args.real_per_video)
            for s in selected_outside:
                selected_specs[s] = {
                    "window_label": 0,
                    "window_source": "from_real_video",
                    "out_dir": split_dirs["real_from_real"],
                }
        else:
            selected_inside = select_evenly(inside, args.inside_per_fake)
            selected_outside = select_evenly(outside, args.outside_per_fake)
            if (not modify_video) and modify_audio and args.audio_only_policy == "drop_all":
                stats_delta["dropped_audio_only_videos"] += 1
            else:
                if (not modify_video) and modify_audio:
                    inside_source = "from_fake_video_audio_only"
                    inside_dir = split_dirs["fake_from_fake_audio_only"]
                else:
                    inside_source = "from_fake_video_tampered"
                    inside_dir = split_dirs["fake_from_fake_tampered"]

                for s in selected_inside:
                    selected_specs[s] = {
                        "window_label": 1,
                        "window_source": inside_source,
                        "out_dir": inside_dir,
                    }

                if args.audio_conflict_policy == "drop_outside" and modify_audio:
                    stats_delta["dropped_outside_due_audio"] += int(len(selected_outside))
                else:
                    for s in selected_outside:
                        selected_specs[s] = {
                            "window_label": 0,
                            "window_source": "from_fake_video_clean",
                            "out_dir": split_dirs["real_from_fake_clean"],
                        }

        selected_starts = sorted(selected_specs.keys())
        if not selected_starts:
            stats_delta["videos_no_selected_windows"] += 1
            stats_delta["videos_ok"] += 1
            return split_manifest_rows, failed_row, stats_delta

        needed_indices = []
        for s in selected_starts:
            needed_indices.extend(range(s, s + args.window_len))

        mel_img = build_mel_image_from_video(
            video_path=video_path,
            ffmpeg_exe=ffmpeg_exe,
            sample_rate=args.sample_rate,
            hop_length=args.mel_hop_length,
            ffmpeg_timeout_sec=args.ffmpeg_timeout_sec,
            temp_mel_path="",
        )
        frame_map = read_selected_frames(
            video_path=video_path,
            needed_indices=needed_indices,
            img_size=args.img_size,
        )
        if not frame_map:
            raise RuntimeError("no_frames_extracted")

        duration = float(item.get("duration", 0.0) or 0.0)
        if duration <= 0:
            duration = float(frame_count) / float(max(fps, 1e-6))

        for s in selected_starts:
            spec = selected_specs[s]
            ratio = float(ratio_map.get(s, 0.0))
            frame_end = s + args.window_len

            image = build_window_image(
                frame_start=s,
                window_len=args.window_len,
                frame_map=frame_map,
                mel_img=mel_img,
                fps=fps,
                sample_rate=args.sample_rate,
                hop_length=args.mel_hop_length,
                img_size=args.img_size,
            )

            file_name = f"{video_id}_s{s:06d}_e{frame_end:06d}_r{ratio:.3f}.png"
            save_path = spec["out_dir"] / file_name

            status = "saved"
            if save_path.exists() and not args.overwrite:
                status = "skipped_existing"
                stats_delta["skipped_existing"] += 1
            else:
                save_rgb_png(save_path, image, compression=args.png_compression)
                if spec["window_source"] == "from_real_video":
                    stats_delta["saved_real_from_real"] += 1
                elif spec["window_source"] == "from_fake_video_clean":
                    stats_delta["saved_real_from_fake_clean"] += 1
                elif spec["window_source"] == "from_fake_video_tampered":
                    stats_delta["saved_fake_from_fake_tampered"] += 1
                elif spec["window_source"] == "from_fake_video_audio_only":
                    stats_delta["saved_fake_from_fake_audio_only"] += 1

            time_start = float(s) / float(max(fps, 1e-6))
            time_end = float(frame_end) / float(max(fps, 1e-6))

            split_manifest_rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "video_name": video_name,
                    "video_rel_path": rel_path,
                    "video_label": int(video_label),
                    "window_label": int(spec["window_label"]),
                    "window_source": spec["window_source"],
                    "frame_start": int(s),
                    "frame_end_exclusive": int(frame_end),
                    "time_start_sec": float(round(time_start, 6)),
                    "time_end_sec": float(round(time_end, 6)),
                    "overlap_ratio": float(round(ratio, 6)),
                    "modify_audio": to_bool(item.get("modify_audio", False)),
                    "modify_video": to_bool(item.get("modify_video", False)),
                    "policy_tag": policy_tag_curr,
                    "n_fakes": int(item.get("n_fakes", 0) or 0),
                    "status": status,
                    "output_path": str(save_path.resolve()),
                }
            )

        stats_delta["videos_ok"] += 1
        return split_manifest_rows, failed_row, stats_delta
    except KeyboardInterrupt:
        raise
    except Exception as e:
        fail(str(e))
        return split_manifest_rows, failed_row, stats_delta


def main():
    args = parse_args()

    if args.outside_thr >= args.inside_thr:
        raise ValueError("outside_thr must be smaller than inside_thr")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_path).expanduser().resolve()
        if str(args.metadata_path).strip()
        else (dataset_root / "metadata.min.json").resolve()
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

    ffmpeg_exe = resolve_ffmpeg_exe(args.ffmpeg_exe)
    ensure_dir(output_root)
    temp_root = output_root / "_temp"
    ensure_dir(temp_root)
    temp_mel_path = temp_root / "mel.png"

    requested_splits = parse_split_list(args.splits)
    if not requested_splits:
        raise ValueError("No valid splits found in --splits")

    rng = np.random.default_rng(args.seed)
    metadata_items = load_metadata_items(metadata_path)

    split_to_items = {s: [] for s in requested_splits}
    for item in metadata_items:
        split = str(item.get("split", "")).strip().lower()
        if split in split_to_items:
            split_to_items[split].append(item)

    manifests_dir = output_root / "manifests"
    ensure_dir(manifests_dir)

    all_stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "metadata_path": str(metadata_path),
        "output_root": str(output_root),
        "config": {
            "splits": requested_splits,
            "window_len": int(args.window_len),
            "frame_stride": int(args.frame_stride),
            "img_size": int(args.img_size),
            "sample_rate": int(args.sample_rate),
            "mel_hop_length": int(args.mel_hop_length),
            "inside_thr": float(args.inside_thr),
            "outside_thr": float(args.outside_thr),
            "inside_per_fake": int(args.inside_per_fake),
            "outside_per_fake": int(args.outside_per_fake),
            "real_per_video": int(args.real_per_video),
            "audio_conflict_policy": str(args.audio_conflict_policy),
            "audio_only_policy": str(args.audio_only_policy),
            "boundary_margin_sec": float(args.boundary_margin),
            "max_real_videos": int(args.max_real_videos),
            "max_fake_videos": int(args.max_fake_videos),
            "video_workers": int(max(1, args.video_workers)),
            "shuffle_videos": bool(args.shuffle_videos),
            "seed": int(args.seed),
            "overwrite": bool(args.overwrite),
            "clean_output": bool(args.clean_output),
            "png_compression": int(args.png_compression),
            "ffmpeg_exe": ffmpeg_exe,
            "ffmpeg_timeout_sec": int(args.ffmpeg_timeout_sec),
            "stop_on_interrupt": bool(args.stop_on_interrupt),
        },
        "splits": {},
    }

    stop_requested = False
    for split in requested_splits:
        if stop_requested:
            break
        split_items = split_to_items.get(split, [])

        real_items = []
        fake_items = []
        for it in split_items:
            if safe_video_label(it.get("original", None)) == 0:
                real_items.append(it)
            else:
                fake_items.append(it)

        if args.shuffle_videos:
            rng.shuffle(real_items)
            rng.shuffle(fake_items)

        if args.max_real_videos > 0:
            real_items = real_items[: args.max_real_videos]
        if args.max_fake_videos > 0:
            fake_items = fake_items[: args.max_fake_videos]

        split_root, split_dirs = prepare_output_layout(
            output_root=output_root,
            split=split,
            clean_output=args.clean_output,
        )

        split_manifest_rows = []
        split_failed_rows = []

        split_stats = {
            "videos_total": int(len(real_items) + len(fake_items)),
            "real_videos": int(len(real_items)),
            "fake_videos": int(len(fake_items)),
            "videos_ok": 0,
            "videos_failed": 0,
            "videos_no_selected_windows": 0,
            "candidate_windows_total": 0,
            "ambiguous_windows_dropped": 0,
            "dropped_boundary": 0,
            "dropped_outside_due_audio": 0,
            "dropped_audio_only_videos": 0,
            "saved_real_from_real": 0,
            "saved_real_from_fake_clean": 0,
            "saved_fake_from_fake_tampered": 0,
            "saved_fake_from_fake_audio_only": 0,
            "skipped_existing": 0,
        }

        ordered = interleave_labeled_videos(real_items, fake_items)
        workers = int(max(1, args.video_workers))

        print("\\n" + "=" * 80)
        print(f"[Split] {split}")
        print(f"[Info] videos(real/fake): {len(real_items)}/{len(fake_items)}")
        print(f"[Info] video_workers: {workers}")
        print(f"[Info] output: {split_root}")

        def merge_result(manifest_rows, failed_row, stats_delta):
            if manifest_rows:
                split_manifest_rows.extend(manifest_rows)
            if failed_row:
                split_failed_rows.append(failed_row)
            for k, v in stats_delta.items():
                split_stats[k] += int(v)

        if workers <= 1:
            for video_label, item in tqdm(ordered, desc=f"Processing {split}"):
                try:
                    result = process_one_video_item(
                        split=split,
                        video_label=video_label,
                        item=item,
                        dataset_root=dataset_root,
                        split_dirs=split_dirs,
                        ffmpeg_exe=ffmpeg_exe,
                        args=args,
                    )
                    merge_result(*result)
                except KeyboardInterrupt:
                    if args.stop_on_interrupt:
                        stop_requested = True
                        print("\n[Warn] Received KeyboardInterrupt signal. Saving partial results before exit...")
                        break
                    print("\n[Warn] Received KeyboardInterrupt signal, skip current video and continue (--no-stop_on_interrupt).")
                    continue
        else:
            futures = {}
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for video_label, item in ordered:
                        fut = executor.submit(
                            process_one_video_item,
                            split,
                            video_label,
                            item,
                            dataset_root,
                            split_dirs,
                            ffmpeg_exe,
                            args,
                        )
                        futures[fut] = (video_label, item)

                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
                        result = fut.result()
                        merge_result(*result)
            except KeyboardInterrupt:
                for fut in futures:
                    fut.cancel()
                if args.stop_on_interrupt:
                    stop_requested = True
                    print("\n[Warn] Received KeyboardInterrupt signal. Saving partial results before exit...")
                else:
                    print("\n[Warn] Received KeyboardInterrupt signal in main thread.")

        manifest_csv = manifests_dir / f"windows_{split}.csv"
        failed_csv = manifests_dir / f"windows_{split}_failed.csv"

        write_csv(manifest_csv, split_manifest_rows, MANIFEST_FIELDS)
        write_csv(failed_csv, split_failed_rows, FAILED_FIELDS)

        split_stats["manifest_csv"] = str(manifest_csv)
        split_stats["failed_csv"] = str(failed_csv)
        all_stats["splits"][split] = split_stats

        print(f"[Done] split={split} | rows={len(split_manifest_rows)} | failed={len(split_failed_rows)}")
        if stop_requested:
            print("[Info] Early stop requested. Partial split outputs are saved.")
            break

    stats_path = manifests_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("\\n" + "#" * 80)
    print("LAV-DF Window Preprocess Finished")
    print(f"Output root : {output_root}")
    print(f"Stats file  : {stats_path}")
    print("#" * 80)


if __name__ == "__main__":
    main()
