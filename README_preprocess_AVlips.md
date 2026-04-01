# AVlips-test 数据集预处理脚本（使用InsightFace）

## 概述

本脚本使用InsightFace一站式框架对AVlips-test数据集进行预处理，包括：
- 人脸检测和对齐
- 音频频谱图提取
- 音频和视频帧拼接

## 安装依赖

### 1. 安装InsightFace（推荐）

```bash
pip install insightface onnxruntime-gpu
```

如果GPU版本安装失败，可以使用CPU版本：
```bash
pip install insightface onnxruntime
```

### 2. 验证安装

```bash
python -c "from insightface.app import FaceAnalysis; print('InsightFace安装成功')"
```

## 使用方法

### 基本使用

```bash
python preprocess_AVlips_insightface.py
```

### 自定义参数

编辑脚本中的参数部分：

```python
# 自定义参数
N_EXTRACT = 10          # 从视频中提取的图像数量
WINDOW_LEN = 5          # 每个窗口的帧数
IMAGE_SIZE = 500        # 图像尺寸
MAX_REAL_IMAGES = 2000  # 最大真实图像数量
MAX_FAKE_IMAGES = 5000  # 最大伪造图像数量
NUM_THREADS = 4         # 线程数

dataset_root = r"E:\data\AVlips-test"      # 数据集根目录
output_root = r"E:\data\AVlips-test\预处理"  # 输出目录
```

## 输出结构

```
E:\data\AVlips-test\预处理\
├── 0_real\          # 真实视频的处理结果
│   ├── video1_0.png
│   ├── video1_1.png
│   └── ...
└── 1_fake\          # 伪造视频的处理结果
    ├── fake1_0.png
    ├── fake1_1.png
    └── ...
```

## 输出图像格式

每个输出图像包含：
- 上半部分：音频频谱图（500 x 2500像素）
- 下半部分：5帧视频帧的人脸区域（500 x 2500像素）

## 特性

### InsightFace模式（推荐）
- 高精度人脸检测
- 自动人脸对齐
- 支持多尺度检测
- 处理速度快

### OpenCV回退模式
- 如果InsightFace未安装，自动使用OpenCV
- 基于Haar级联分类器
- 保证脚本可用性

## 处理流程

1. **视频读取**：从指定目录读取视频文件
2. **人脸检测**：使用InsightFace检测人脸区域
3. **人脸对齐**：自动对齐人脸到标准尺寸
4. **音频提取**：从视频中提取音频
5. **频谱生成**：生成音频频谱图
6. **数据拼接**：将频谱图和视频帧拼接成训练样本

## 性能优化

- 支持断点续传（跳过已处理的图像）
- 自动检测已存在图像数量
- 智能调整处理数量
- 多线程支持（可扩展）

## 常见问题

### 1. InsightFace安装失败

**问题**：`pip install insightface` 失败

**解决方案**：
- 使用国内镜像：`pip install insightface -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 或者使用CPU版本：`pip install insightface onnxruntime`

### 2. GPU内存不足

**问题**：处理时出现CUDA OOM错误

**解决方案**：
- 使用CPU版本：`pip install onnxruntime`
- 减小检测尺寸：修改`det_size=(640, 640)`为`det_size=(320, 320)`

### 3. 人脸检测失败

**问题**：某些帧检测不到人脸

**解决方案**：
- 脚本会自动回退到整帧处理
- 可以调整`margin`参数来扩展检测区域

## 数据集要求

输入数据集应包含以下结构：
```
E:\data\AVlips-test\
├── 0_real\          # 真实视频
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── 1_fake\          # 伪造视频
    ├── fake1.mp4
    ├── fake2.mp4
    └── ...
```

## 依赖项

- Python 3.8+
- OpenCV
- NumPy
- Librosa
- Matplotlib
- InsightFace（推荐）
- FFmpeg（用于音频提取）

## 注意事项

1. 确保FFmpeg已安装并在PATH中
2. 确保有足够的磁盘空间存储输出图像
3. 处理大量视频时可能需要较长时间
4. 建议使用GPU加速InsightFace处理

## 联系方式

如有问题，请检查：
1. 依赖是否正确安装
2. 数据集路径是否正确
3. FFmpeg是否可用
4. 磁盘空间是否充足