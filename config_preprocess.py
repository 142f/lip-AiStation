"""
Deepfake 统一数据基座 - 阶段一配置文件
方案A：InsightFace 一站式框架
"""

# ============ 路径配置 ============
DATASET_ROOT       = r"E:\data\AVlips-test"
OUTPUT_ROOT        = r"E:\data\AVlips-test\prepor"
INSIGHTFACE_MODEL  = "buffalo_l"
INSIGHTFACE_DET_SIZE = (640, 640)

# ============ 帧采样配置 ============
FRAME_SKIP             = 2       # 每隔N帧采样一帧 (1=每帧都处理, 2=隔一帧)
MAX_FRAMES_PER_VIDEO   = 500     # 单视频最大保存帧数，防止长视频磁盘爆满

# ============ 人脸检测配置 ============
FACE_DET_THRESHOLD     = 0.5     # 人脸检测置信度阈值 (过滤误检)

# ============ 输出配置 ============
SAVE_FORMAT            = ".jpg"  # 帧图像保存格式
VERBOSE                = True    # 详细日志