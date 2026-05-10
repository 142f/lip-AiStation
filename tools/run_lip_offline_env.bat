@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

if "%LIPFD_CHECKPOINT_ROOT%"=="" set "LIPFD_CHECKPOINT_ROOT=%PROJECT_ROOT%\checkpoints"
if "%LIPFD_PRETRAINED_ROOT%"=="" set "LIPFD_PRETRAINED_ROOT=%LIPFD_CHECKPOINT_ROOT%\pretrained"
if "%LIPFD_CLIP_ROOT%"=="" set "LIPFD_CLIP_ROOT=%LIPFD_PRETRAINED_ROOT%\clip"
if "%LIPFD_TORCH_CHECKPOINT_DIR%"=="" set "LIPFD_TORCH_CHECKPOINT_DIR=%LIPFD_PRETRAINED_ROOT%\torch\hub\checkpoints"
if "%LIPFD_INSIGHTFACE_ROOT%"=="" set "LIPFD_INSIGHTFACE_ROOT=%LIPFD_PRETRAINED_ROOT%\insightface"

if "%TORCH_HOME%"=="" set "TORCH_HOME=%LIPFD_PRETRAINED_ROOT%\torch"
if "%HF_HOME%"=="" set "HF_HOME=%LIPFD_PRETRAINED_ROOT%\huggingface"
if "%XDG_CACHE_HOME%"=="" set "XDG_CACHE_HOME=%LIPFD_PRETRAINED_ROOT%\cache"

if "%HF_HUB_OFFLINE%"=="" set "HF_HUB_OFFLINE=1"
if "%TRANSFORMERS_OFFLINE%"=="" set "TRANSFORMERS_OFFLINE=1"
if "%NO_ALBUMENTATIONS_UPDATE%"=="" set "NO_ALBUMENTATIONS_UPDATE=1"

mkdir "%LIPFD_CLIP_ROOT%" 2>nul
mkdir "%LIPFD_TORCH_CHECKPOINT_DIR%" 2>nul
mkdir "%LIPFD_INSIGHTFACE_ROOT%\models" 2>nul
mkdir "%HF_HOME%" 2>nul
mkdir "%XDG_CACHE_HOME%" 2>nul

cd /d "%PROJECT_ROOT%"

if "%~1"=="" (
  echo Offline environment configured.
  echo Project root: %PROJECT_ROOT%
  echo Pretrained root: %LIPFD_PRETRAINED_ROOT%
  echo.
  echo Run a command through this script, for example:
  echo   tools\run_lip_offline_env.bat python test.py --ckpt checkpoints\latest_checkpoint.pth
  exit /b 0
)

%*
