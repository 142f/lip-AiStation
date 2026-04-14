r"""
合并了 FakeAVCeleb 数据集选择逻辑 和 InsightFace 预处理逻辑的脚本。
针对 E:\data\FakeAVCeleb-test-pre 目录输出统一基座格式数据。
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

_DLL_DIR_HANDLES = []

def _setup_nvidia_dll_path():
    """将 nvidia pip 运行时的 DLL 目录注入 PATH / add_dll_directory。"""
    try:
        import nvidia
    except Exception:
        return

    nvidia_root = os.path.dirname(nvidia.__file__)
    candidates = [
        os.path.join(nvidia_root, 'cuda_runtime', 'bin'),
        os.path.join(nvidia_root, 'cuda_nvrtc', 'bin'),
        os.path.join(nvidia_root, 'cublas', 'bin'),
        os.path.join(nvidia_root, 'cudnn', 'bin'),
        os.path.join(nvidia_root, 'cufft', 'bin'),
        os.path.join(nvidia_root, 'curand', 'bin'),
        os.path.join(nvidia_root, 'nvjitlink', 'bin'),
    ]

    current_path = os.environ.get('PATH', '')
    path_parts = current_path.split(os.pathsep) if current_path else []

    for dll_dir in candidates:
        if not os.path.isdir(dll_dir):
            continue

        if dll_dir not in path_parts:
            os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')
            path_parts.append(dll_dir)

        if hasattr(os, 'add_dll_directory'):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))
            except Exception:
                pass

_setup_nvidia_dll_path()

import cv2
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnxruntime as ort

# =====================================================================
# 参数配置
# =====================================================================
DATASET_ROOT = r"E:\data\FakeAVCeleb\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
OUTPUT_ROOT = r"E:\data\FakeAVCeleb-test-pre"
METADATA_CSV = os.path.join(DATASET_ROOT, "meta_data.csv")

INSIGHTFACE_MODEL = "buffalo_l"
INSIGHTFACE_DET_SIZE = (640, 640)
FRAME_SKIP = 1  
MAX_FRAMES_PER_VIDEO = None  
FACE_DET_THRESHOLD = 0.2  
SAVE_FORMAT = ".jpg"
VERBOSE = False
FORCE_REPROCESS = False

LABEL_MODE = "visual"
EXCLUDE_RF_IN_VISUAL = True

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

def derive_binary_labels(original_type: str) -> dict:
    visual_fake = original_type in {"FakeVideo-RealAudio", "FakeVideo-FakeAudio"}
    audio_fake = original_type in {"RealVideo-FakeAudio", "FakeVideo-FakeAudio"}
    any_fake = original_type != "RealVideo-RealAudio"

    return {
        "visual": int(visual_fake),
        "audio": int(audio_fake),
        "any": int(any_fake),
    }

# =====================================================================
# FakeAVCeleb 数据集解析逻辑
# =====================================================================
def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
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

def build_sample_plan(metadata_csv: str, dataset_root: str) -> pd.DataFrame:
    if LABEL_MODE not in {"visual", "audio", "any"}:
        raise ValueError(f"Unsupported LABEL_MODE={LABEL_MODE!r}, expected one of ['visual', 'audio', 'any']")

    df = pd.read_csv(metadata_csv)
    df = normalize_metadata(df)
    
    # [优化] 引入哈希聚合映射，将 O(N^2) 级的大表扫描布尔索引转为 O(1) 的字典查询
    grouped_df = dict(tuple(df.groupby(['source', 'race', 'gender', 'type'])))
    
    records = []
    sample_idx = 1

    for race, gender_map in SOURCE_SPLIT.items():
        for gender, source_ids in gender_map.items():
            for source_id in source_ids:
                for original_type in TYPE_ORDER:
                    
                    # 极速 O(1) 数据拉取
                    rows = grouped_df.get((source_id, race, gender, original_type))
                    if rows is None or rows.empty:
                        raise ValueError(f"No metadata row found for {source_id}, {race}, {gender}, {original_type}")

                    rows = rows.copy()
                    rows["_sort_key"] = rows["filename"].astype(str) + "|" + rows["folder"].astype(str)
                    rows = rows.sort_values("_sort_key").reset_index(drop=True)
                    chosen = rows.iloc[0]

                    video_path = resolve_video_path(dataset_root, chosen["folder"], chosen["filename"])

                    if LABEL_MODE == "visual" and EXCLUDE_RF_IN_VISUAL and original_type == "RealVideo-FakeAudio":
                        continue

                    labels = derive_binary_labels(original_type)
                    label_val = labels[LABEL_MODE]
                    label_dir = "1_fake" if label_val == 1 else "0_real"
                    
                    v_name = f"{race}_{gender}_{source_id}_{TYPE_TO_SHORT[original_type]}.mp4".replace(" ", "_")

                    records.append({
                        "sample_id": f"S{sample_idx:04d}",
                        "video_path": video_path,
                        "v_name": v_name,
                        "label_val": label_val,
                        "out_dir": os.path.join(OUTPUT_ROOT, label_dir),
                        "original_type": original_type,
                        "label_mode": LABEL_MODE,
                        "visual_label": labels["visual"],
                        "audio_label": labels["audio"],
                        "any_label": labels["any"],
                    })
                    sample_idx += 1

    return pd.DataFrame(records)

# =====================================================================
# InsightFace 预处理逻辑
# =====================================================================
def safe_imwrite(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1] or SAVE_FORMAT
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            with open(file_path, "wb") as f:
                f.write(im_buf_arr.tobytes())
            return True
        if VERBOSE:
            print(f"[WARN] cv2.imencode 失败: {file_path}")
        return False
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] 图片写入异常 {file_path}: {e}")
        return False

def is_processed_video_complete(json_path: str, v_out_dir: str) -> bool:
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception:
        return False

    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        return False

    # [优化] 将 O(N) 级别基于磁盘的 os.path.exists 扫描，转为基于内存哈希表的 O(1) 检索
    try:
        existing_files = set(os.listdir(v_out_dir))
    except OSError:
        return False

    for item in frames:
        file_path = item.get("file_path") if isinstance(item, dict) else None
        if not file_path or file_path not in existing_files:
            return False

    return True

def _build_face_payload(face):
    """
    序列化 InsightFace 结果。
    [优化] 启用纯粹的 Numpy C 级别向量化操作，剔除低效的 Python 显式类型转换循环。
    """
    bbox = getattr(face, "bbox", None)
    bbox_list = np.round(bbox, 2).tolist() if bbox is not None else [0.0, 0.0, 0.0, 0.0]

    payload = {
        "bbox": bbox_list,
        "det_score": round(float(getattr(face, "det_score", 0.0)), 4),
        "kps_5": None,
        "kps_106": None,
    }

    kps_5_obj = getattr(face, "kps", None)
    if kps_5_obj is not None:
        payload["kps_5"] = np.round(kps_5_obj, 2).tolist()

    kps_106_obj = getattr(face, "landmark_2d_106", None)
    if kps_106_obj is not None:
        payload["kps_106"] = np.round(kps_106_obj, 2).tolist()

    return payload

class SafeVideoCapture:
    def __init__(self, path):
        self.path = str(path)
        self.temp_path = None
        self.cap = None

    def __enter__(self):
        if not self.path.isascii():
            fd, self.temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            os.remove(self.temp_path)
            try:
                os.link(self.path, self.temp_path)
            except OSError:
                shutil.copy2(self.path, self.temp_path)
            self.cap = cv2.VideoCapture(self.temp_path)
        else:
            self.cap = cv2.VideoCapture(self.path)
        return self

    def release(self):
        """[优化] 统一资源回收出口，消除职责冗余并采取防御性重置，防止野指针。"""
        if self.cap:
            self.cap.release()
            self.cap = None 
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass
            self.temp_path = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def isOpened(self):
        return self.cap.isOpened() if self.cap else False

    def get(self, prop_id):
        return self.cap.get(prop_id) if self.cap else 0

    def grab(self):
        return self.cap.grab() if self.cap else False

    def retrieve(self):
        return self.cap.retrieve() if self.cap else (False, None)


global_app = None

def init_worker():
    global global_app
    import logging
    logging.getLogger('insightface').setLevel(logging.ERROR)

    try:
        ort.preload_dlls()
    except Exception as e:
        print(f"[WARN] ONNX Runtime preload_dlls 失败，将继续尝试: {e}")

    try:
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            cuda_ok = False
            probe_model = os.path.expanduser(rf"~/.insightface/models/{INSIGHTFACE_MODEL}/det_10g.onnx")

            try:
                if os.path.exists(probe_model):
                    probe_session = ort.InferenceSession(
                        probe_model,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    cuda_ok = (
                        len(probe_session.get_providers()) > 0
                        and probe_session.get_providers()[0] == 'CUDAExecutionProvider'
                    )
                else:
                    cuda_ok = True
            except Exception as e:
                print(f"[WARN] CUDA provider 探测失败，回退 CPU: {e}")
                cuda_ok = False

            if cuda_ok:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
                print("[INFO] ONNX Runtime CUDA provider 可用，启用 GPU 加速。")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                print("[WARN] CUDA provider 不可用，已回退 CPU。")
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print("[WARN] ONNX Runtime 未找到 CUDAExecutionProvider，退回 CPU。")
    except Exception as e:
        print(f"[ERROR] 无法加载 ONNX Runtime: {e}")
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        print(f"[ERROR] 无法导入 insightface: {e}")
        print("请确认在当前虚拟环境中已安装 insightface, onnxruntime 以及 albumentations 等依赖。")
        raise

    global_app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    global_app.prepare(ctx_id=ctx_id, det_size=INSIGHTFACE_DET_SIZE)
    print(f"[InsightFace] 初始化完毕，运行设备: {'GPU (CUDA)' if ctx_id == 0 else 'CPU'}")


def process_single_video(task_dict: dict):
    v_path = task_dict['video_path']
    v_name = task_dict['v_name']
    label_val = task_dict['label_val']
    out_dir = task_dict['out_dir']
    original_type = task_dict.get('original_type', 'unknown')
    visual_label = task_dict.get('visual_label')
    audio_label = task_dict.get('audio_label')
    any_label = task_dict.get('any_label')

    v_stem = os.path.splitext(v_name)[0]
    v_out_dir = os.path.join(out_dir, v_stem)
    json_path = os.path.join(v_out_dir, 'metadata.json')

    if not FORCE_REPROCESS and is_processed_video_complete(json_path, v_out_dir):
        return True

    if os.path.isdir(v_out_dir):
        shutil.rmtree(v_out_dir, ignore_errors=True)

    os.makedirs(v_out_dir, exist_ok=True)
    with SafeVideoCapture(v_path) as cap:
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        metadata = {
            "video_info": {
                "video_name": v_name,
                "label": label_val,
                "fps": round(fps, 4),
                "total_frames": total_frames_meta,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "frame_skip": FRAME_SKIP,
                "face_det_threshold": FACE_DET_THRESHOLD,
                "original_type": original_type,
                "label_mode": LABEL_MODE,
                "visual_label": visual_label,
                "audio_label": audio_label,
                "any_label": any_label,
            },
            "frames": []
        }

        frame_idx, saved_count, face_hit_count = 0, 0, 0

        while True:
            if MAX_FRAMES_PER_VIDEO is not None and saved_count >= MAX_FRAMES_PER_VIDEO:
                break

            if not cap.grab():
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                break

            face_payload = None
            try:
                faces = global_app.get(frame)
                if faces:
                    valid_faces = [f for f in faces if getattr(f, 'det_score', 0) >= FACE_DET_THRESHOLD]
                    if valid_faces:
                        best_face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        face_payload = _build_face_payload(best_face)
            except Exception as e:
                if VERBOSE:
                    print(f"[WARN] 人脸检测异常 frame={frame_idx}: {e}")

            img_name = f"frame_{frame_idx:06d}{SAVE_FORMAT}"
            img_path = os.path.join(v_out_dir, img_name)

            if safe_imwrite(img_path, frame):
                metadata["frames"].append({
                    "file_path": img_name,
                    "frame_idx": frame_idx,
                    "has_face": face_payload is not None,
                    "face": face_payload,
                })
                saved_count += 1
                if face_payload is not None:
                    face_hit_count += 1

            frame_idx += 1

    metadata["video_info"]["saved_frames"] = saved_count
    metadata["video_info"]["face_frames"] = face_hit_count

    if metadata["frames"]:
        tmp_json_path = json_path + ".tmp"
        try:
            with open(tmp_json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            os.replace(tmp_json_path, json_path)
            return True
        except Exception:
            if os.path.exists(tmp_json_path):
                os.remove(tmp_json_path)
            return False
    else:
        try: shutil.rmtree(v_out_dir, ignore_errors=True)
        except: pass
        return False

# =====================================================================
# 主函数
# =====================================================================
def run():
    print("=" * 60)
    print("🚀 融合版 - FakeAVCeleb 特选视频 -> InsightFace 预处理 (终极版)")
    print(f"[配置] 视频源: {DATASET_ROOT}")
    print(f"[配置] 输出至: {OUTPUT_ROOT}")
    print(f"[配置] 标签模式: {LABEL_MODE}")
    if LABEL_MODE == "visual":
        print(f"[配置] 视觉任务排除 RF: {EXCLUDE_RF_IN_VISUAL}")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "1_fake"), exist_ok=True)
    
    print("[1/2] 解析 FakeAVCeleb 选取计划...")
    plan_df = build_sample_plan(METADATA_CSV, DATASET_ROOT)
    task_queue = plan_df.to_dict('records')
    print(f"计划处理视频数: {len(task_queue)}")
    if len(plan_df) > 0:
        print("标签分布:", plan_df["label_val"].value_counts().to_dict())
        print("类型分布:", plan_df["original_type"].value_counts().to_dict())

    print("[2/2] 初始化 InsightFace 并执行处理...")
    init_worker()

    success_count, fail_count = 0, 0
    with tqdm(total=len(task_queue), desc="处理进度") as pbar:
        for task_dict in task_queue:
            if process_single_video(task_dict):
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)

    print("\n" + "=" * 60)
    print(f"✅ 处理完毕！成功: {success_count}, 失败或跳过: {fail_count}")
    print("=" * 60)

if __name__ == "__main__":
    run()