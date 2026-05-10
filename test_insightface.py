"""
InsightFace快速测试脚本
用于验证InsightFace是否正确安装和配置
"""

import os
import sys
from models.offline_paths import insightface_root

def test_insightface():
    """测试InsightFace是否可用"""
    print("=" * 60)
    print("InsightFace安装测试")
    print("=" * 60)
    
    # 测试1: 导入InsightFace
    print("\n[测试1] 导入InsightFace...")
    try:
        from insightface.app import FaceAnalysis
        print("✓ InsightFace导入成功")
    except ImportError as e:
        print(f"✗ InsightFace导入失败: {e}")
        print("\n请运行以下命令安装InsightFace:")
        print("  pip install insightface onnxruntime-gpu")
        print("或")
        print("  pip install insightface onnxruntime")
        return False
    
    # 测试2: 检查CUDA可用性
    print("\n[测试2] 检查CUDA可用性...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用 (GPU数量: {torch.cuda.device_count()})")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            use_cuda = True
        else:
            print("○ CUDA不可用，将使用CPU")
            use_cuda = False
    except ImportError:
        print("○ PyTorch未安装，跳过CUDA检测")
        use_cuda = False
    
    # 测试3: 初始化FaceAnalysis
    print("\n[测试3] 初始化FaceAnalysis...")
    try:
        ctx_id = 0 if use_cuda else -1
        app = FaceAnalysis(name='buffalo_l', root=insightface_root())
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("✓ FaceAnalysis初始化成功")
        print(f"  模型: buffalo_l")
        print(f"  设备: {'GPU' if use_cuda else 'CPU'}")
    except Exception as e:
        print(f"✗ FaceAnalysis初始化失败: {e}")
        print("\n可能的原因:")
        print("1. 模型文件下载失败（网络问题）")
        print("2. 磁盘空间不足")
        print("3. 权限问题")
        return False
    
    # 测试4: 测试人脸检测
    print("\n[测试4] 测试人脸检测...")
    try:
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = app.get(test_image)
        print(f"✓ 人脸检测成功 (检测到 {len(faces)} 个人脸)")
        
        if len(faces) > 0:
            face = faces[0]
            print(f"  人脸框: {face.bbox}")
            print(f"  关键点: {face.kps.shape}")
            print(f"  姿态: {face.pose}")
        
    except Exception as e:
        print(f"✗ 人脸检测失败: {e}")
        return False
    
    # 测试5: 检查FFmpeg
    print("\n[测试5] 检查FFmpeg...")
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg可用")
            version_line = result.stdout.split('\n')[0]
            print(f"  版本: {version_line}")
        else:
            print("✗ FFmpeg不可用")
            print("\n请安装FFmpeg:")
            print("1. 下载: https://ffmpeg.org/download.html")
            print("2. 解压到指定目录")
            print("3. 将bin目录添加到系统PATH")
    except Exception as e:
        print(f"✗ FFmpeg检测失败: {e}")
        print("FFmpeg可能未安装，但预处理脚本可以继续运行")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✓ InsightFace已正确安装并可以使用")
    print("✓ 可以运行预处理脚本: python preprocess_AVlips_insightface.py")
    print("\n注意事项:")
    print("1. 首次运行会下载模型文件（约500MB）")
    print("2. 确保有足够的磁盘空间")
    print("3. 如果使用CPU，处理速度会较慢")
    
    return True

if __name__ == "__main__":
    try:
        success = test_insightface()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
