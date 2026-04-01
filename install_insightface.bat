@echo off
REM InsightFace安装脚本（Windows）

echo ========================================
echo InsightFace安装脚本
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python未安装或不在PATH中
    pause
    exit /b 1
)

echo [1/3] 检查Python版本...
python --version
echo.

echo [2/3] 安装InsightFace...
echo 选择安装模式:
echo 1. GPU版本（推荐，需要CUDA）
echo 2. CPU版本（兼容性更好）
echo.
set /p choice="请输入选择 (1 或 2): "

if "%choice%"=="1" (
    echo 正在安装InsightFace和onnxruntime-gpu...
    pip install insightface onnxruntime-gpu
) else if "%choice%"=="2" (
    echo 正在安装InsightFace和onnxruntime...
    pip install insightface onnxruntime
) else (
    echo [错误] 无效的选择
    pause
    exit /b 1
)

if %errorlevel% neq 0 (
    echo.
    echo [错误] 安装失败，尝试使用国内镜像...
    echo.
    if "%choice%"=="1" (
        pip install insightface onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    ) else (
        pip install insightface onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
)

if %errorlevel% neq 0 (
    echo.
    echo [错误] 安装失败，请检查网络连接或手动安装
    pause
    exit /b 1
)

echo.
echo [3/3] 验证安装...
python -c "from insightface.app import FaceAnalysis; print('InsightFace安装成功！')"

if %errorlevel% neq 0 (
    echo [错误] InsightFace验证失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 现在可以运行预处理脚本:
echo python preprocess_AVlips_insightface.py
echo.
pause