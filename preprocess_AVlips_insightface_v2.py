"""
Deepfake 统一数据基座 (阶段一：高速预处理引擎)

方案A：InsightFace 一站式框架

修复清单：
  FIX-1:  图片写入移到 kps_106 检查之后，消除孤儿文件
  FIX-2:  bbox 写入 metadata，阶段二可做人脸裁剪
  FIX-3:  视频 fps/resolution/total_frames 写入 metadata
  FIX-4:  safe_imwrite 返回布尔值，写入失败时跳过该帧
  FIX-5:  人脸检测置信度过滤
  FIX-6:  ONNX providers 根据环境动态选择
  FIX-7:  帧采样间隔可配置
  FIX-8:  单视频帧数上限
  FIX-9:  裸 except 改为 except Exception
  FIX-10: 清理失败时用 shutil.rmtree 替代 os.rmdir
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# =====================================================================
# 【系统级防御】强制前置加载重型 C++ 库，防止 Windows I/O 阻塞
# =====================================================================
print("[系统] 正在预热底层 C++ 科学计算库...")
import torch
import cv2
from insightface.app import FaceAnalysis
from models.offline_paths import insightface_root
print("[系统] 底层计算库预热完毕！")

import json
import numpy as np
import tempfile
import shutil
from tqdm import tqdm

from config_preprocess import (
    DATASET_ROOT, OUTPUT_ROOT,
    INSIGHTFACE_MODEL, INSIGHTFACE_DET_SIZE,
    FRAME_SKIP, MAX_FRAMES_PER_VIDEO,
    FACE_DET_THRESHOLD, SAVE_FORMAT, VERBOSE
)


# =====================================================================
# I/O 安全组件
# =====================================================================

# [FIX-4] safe_imwrite 返回布尔值，调用方可知写入是否成功
def safe_imwrite(file_path, img):
    """
    安全写入图片（兼容中文路径）
    返回: True 成功, False 失败
    """
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = SAVE_FORMAT
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            with open(file_path, "wb") as f:
                f.write(im_buf_arr.tobytes())
            return True
        else:
            if VERBOSE:
                print(f"  [写入失败] cv2.imencode 返回失败: {file_path}")
            return False
    except Exception as e:
        if VERBOSE:
            print(f"  [写入异常] {file_path}: {e}")
        return False


class SafeVideoCapture:
    """安全视频读取器（兼容中文路径）"""

    def __init__(self, path):
        self.path = path
        self.temp_path = None
        self.cap = None

        if not path.isascii():
            fd, self.temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            shutil.copy2(path, self.temp_path)
            self.cap = cv2.VideoCapture(self.temp_path)
        else:
            self.cap = cv2.VideoCapture(path)

    def isOpened(self):
        return self.cap.isOpened() if self.cap else False

    def get(self, prop_id):
        """获取视频属性"""
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


# =====================================================================
# 核心业务组件
# =====================================================================
global_app = None


def init_worker():
    """初始化 InsightFace 模型"""
    global global_app
    import logging
    logging.getLogger('insightface').setLevel(logging.ERROR)

    # [FIX-6] 根据环境动态选择 ONNX Runtime providers
    if torch.cuda.is_available():
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                print("[警告] PyTorch检测到CUDA，但onnxruntime-gpu未安装，回退CPU")
        except Exception:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    global_app = FaceAnalysis(
        name=INSIGHTFACE_MODEL,
        root=insightface_root(),
        providers=providers
    )
    global_app.prepare(ctx_id=ctx_id, det_size=INSIGHTFACE_DET_SIZE)

    device_name = 'GPU (CUDA)' if ctx_id == 0 else 'CPU'
    print(f"[InsightFace] 初始化完毕，运行设备: {device_name}")


def process_single_video(task_args):
    """
    单视频处理流水线

    输出结构:
        video_dir/
        ├── frame_0000.jpg
        ├── frame_0002.jpg
        ├── ...
        └── metadata.json

    metadata.json 结构:
    {
        "video_info": {
            "video_name": "xxx.mp4",
            "label": 0,
            "fps": 25.0,                  # [FIX-3]
            "total_frames": 750,          # [FIX-3]
            "width": 1920,                # [FIX-3]
            "height": 1080                # [FIX-3]
        },
        "frames": [
            {
                "file_path": "frame_0000.jpg",
                "frame_idx": 0,
                "face": {
                    "bbox": [x1, y1, x2, y2],   # [FIX-2]
                    "det_score": 0.99,           # [FIX-5]
                    "kps_5": [[x,y], ...],
                    "kps_106": [[x,y], ...]
                }
            }
        ]
    }
    """
    v_path, v_name, label_val, out_dir = task_args
    v_stem = os.path.splitext(v_name)[0]
    v_out_dir = os.path.join(out_dir, v_stem)
    json_path = os.path.join(v_out_dir, 'metadata.json')

    # 断点续跑：JSON 存在说明已完整处理
    if os.path.exists(json_path):
        return True

    os.makedirs(v_out_dir, exist_ok=True)

    cap = SafeVideoCapture(v_path)
    if not cap.isOpened():
        if VERBOSE:
            print(f"  [跳过] 无法打开: {v_name}")
        return False

    # [FIX-3] 读取视频元信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        if VERBOSE:
            print(f"  [跳过] 视频元信息异常: {v_name} (fps={fps}, frames={total_frames})")
        cap.release()
        return False

    metadata = {
        "video_info": {
            "video_name": v_name,
            "label": label_val,
            "fps": round(fps, 4),              # [FIX-3]
            "total_frames": total_frames,      # [FIX-3]
            "width": width,                    # [FIX-3]
            "height": height                   # [FIX-3]
        },
        "frames": []
    }

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [FIX-8] 单视频帧数上限
        if saved_count >= MAX_FRAMES_PER_VIDEO:
            break

        # [FIX-7] 帧采样间隔可配置
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        faces = global_app.get(frame)

        if faces:
            # [FIX-5] 置信度过滤
            valid_faces = [
                f for f in faces
                if getattr(f, 'det_score', 0) >= FACE_DET_THRESHOLD
            ]

            if not valid_faces:
                frame_idx += 1
                continue

            # 选最大人脸
            best_face = max(
                valid_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            # 提取关键点
            kps_5 = best_face.kps.tolist()
            kps_106 = (
                best_face.landmark_2d_106.tolist()
                if getattr(best_face, 'landmark_2d_106', None) is not None
                else None
            )

            # [FIX-1] 只在所有数据都就绪后才写入图片
            if kps_106 is None:
                frame_idx += 1
                continue

            img_name = f"frame_{frame_idx:06d}{SAVE_FORMAT}"
            img_path = os.path.join(v_out_dir, img_name)

            # [FIX-4] 检查写入结果
            if not safe_imwrite(img_path, frame):
                frame_idx += 1
                continue

            # [FIX-2] bbox + [FIX-5] det_score 写入 metadata
            bbox = best_face.bbox.tolist()
            det_score = float(best_face.det_score)

            metadata["frames"].append({
                "file_path": img_name,
                "frame_idx": frame_idx,
                "face": {
                    "bbox": [round(v, 2) for v in bbox],          # [FIX-2]
                    "det_score": round(det_score, 4),              # [FIX-5]
                    "kps_5": [[round(x, 2), round(y, 2)] for x, y in kps_5],
                    "kps_106": [[round(x, 2), round(y, 2)] for x, y in kps_106]
                }
            })
            saved_count += 1

        frame_idx += 1

    cap.release()

    # 保存 metadata
    if metadata["frames"]:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            if VERBOSE:
                print(f"  [JSON写入失败] {v_name}: {e}")
            return False
    else:
        # [FIX-10] 无有效帧时清理整个目录（包括可能的残留文件）
        try:
            shutil.rmtree(v_out_dir, ignore_errors=True)
        except Exception:  # [FIX-9]
            pass
        return False


# =====================================================================
# 主进程调度器
# =====================================================================
def run_fast_preprocessing():
    """主处理入口"""
    print("=" * 60)
    print("🚀 Deepfake 统一数据基座 - 阶段一预处理引擎")
    print("   方案A：InsightFace 一站式框架")
    print("=" * 60)
    print(f"[配置] 数据集根目录:     {DATASET_ROOT}")
    print(f"[配置] 输出目录:         {OUTPUT_ROOT}")
    print(f"[配置] InsightFace模型:  {INSIGHTFACE_MODEL}")
    print(f"[配置] 帧采样间隔:       每 {FRAME_SKIP} 帧取 1 帧")
    print(f"[配置] 单视频最大帧数:   {MAX_FRAMES_PER_VIDEO}")
    print(f"[配置] 人脸置信度阈值:   {FACE_DET_THRESHOLD}")

    if torch.cuda.is_available():
        print(f"[硬件] GPU (CUDA) - {torch.cuda.get_device_name(0)}")
    else:
        print("[硬件] CPU (⚠️ 未检测到GPU，速度会较慢)")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 组装任务队列
    task_queue = []
    for label_dir, label_val in [('0_real', 0), ('1_fake', 1)]:
        in_dir = os.path.join(DATASET_ROOT, label_dir)
        out_dir = os.path.join(OUTPUT_ROOT, label_dir)

        if not os.path.exists(in_dir):
            print(f"[警告] 目录不存在，跳过: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [
            f for f in os.listdir(in_dir)
            if f.lower().endswith(video_exts)
        ]

        print(f"[Info] {label_dir}: 找到 {len(video_files)} 个视频文件")

        for v_name in video_files:
            v_path = os.path.join(in_dir, v_name)
            task_queue.append((v_path, v_name, label_val, out_dir))

    if not task_queue:
        print("❌ 没找到任何视频任务，请检查 DATASET_ROOT 路径！")
        return

    print(f"\n[Info] 总计任务: {len(task_queue)} 个视频")

    # 初始化模型
    init_worker()

    # 执行处理
    success_count = 0
    fail_count = 0

    with tqdm(total=len(task_queue), desc="预处理进度") as pbar:
        for task in task_queue:
            v_name = task[1]
            try:
                if process_single_video(task):
                    success_count += 1
                else:
                    fail_count += 1
                    if VERBOSE:
                        pbar.write(f"  [失败] {v_name}")
            except Exception as e:  # [FIX-9]
                fail_count += 1
                pbar.write(f"  [异常] {v_name}: {e}")
            pbar.update(1)

    # 汇总统计
    print("\n" + "=" * 60)
    print("✅ 阶段一预处理完毕！")
    print("=" * 60)
    print(f"[结果] 成功: {success_count} / {len(task_queue)}")
    print(f"[结果] 失败: {fail_count} / {len(task_queue)}")
    print(f"[结果] 跳过(已存在): {len(task_queue) - success_count - fail_count}")
    print(f"[输出] {OUTPUT_ROOT}")
    print("=" * 60)

    # 输出目录统计
    for label_dir in ['0_real', '1_fake']:
        out_dir = os.path.join(OUTPUT_ROOT, label_dir)
        if os.path.exists(out_dir):
            video_dirs = [
                d for d in os.listdir(out_dir)
                if os.path.isdir(os.path.join(out_dir, d))
            ]
            total_frames = 0
            for vd in video_dirs:
                vd_path = os.path.join(out_dir, vd)
                jpg_count = len([
                    f for f in os.listdir(vd_path)
                    if f.endswith(SAVE_FORMAT)
                ])
                total_frames += jpg_count
            print(f"  {label_dir}: {len(video_dirs)} 个视频, {total_frames} 帧")

    print("=" * 60)


if __name__ == '__main__':
    try:
        run_fast_preprocessing()
    except KeyboardInterrupt:
        print("\n\n[Info] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
