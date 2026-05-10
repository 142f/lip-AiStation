#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LIPFD_CHECKPOINT_ROOT="${LIPFD_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
export LIPFD_PRETRAINED_ROOT="${LIPFD_PRETRAINED_ROOT:-${LIPFD_CHECKPOINT_ROOT}/pretrained}"
export LIPFD_CLIP_ROOT="${LIPFD_CLIP_ROOT:-${LIPFD_PRETRAINED_ROOT}/clip}"
export LIPFD_TORCH_CHECKPOINT_DIR="${LIPFD_TORCH_CHECKPOINT_DIR:-${LIPFD_PRETRAINED_ROOT}/torch/hub/checkpoints}"
export LIPFD_INSIGHTFACE_ROOT="${LIPFD_INSIGHTFACE_ROOT:-${LIPFD_PRETRAINED_ROOT}/insightface}"

export TORCH_HOME="${TORCH_HOME:-${LIPFD_PRETRAINED_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${LIPFD_PRETRAINED_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LIPFD_PRETRAINED_ROOT}/cache}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

mkdir -p \
  "${LIPFD_CLIP_ROOT}" \
  "${LIPFD_TORCH_CHECKPOINT_DIR}" \
  "${LIPFD_INSIGHTFACE_ROOT}/models" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}"

cd "${PROJECT_ROOT}"

if [ "$#" -eq 0 ]; then
  echo "Offline environment configured."
  echo "Project root: ${PROJECT_ROOT}"
  echo "Pretrained root: ${LIPFD_PRETRAINED_ROOT}"
  echo
  echo "Run a command through this script, for example:"
  echo "  bash tools/run_lip_offline_env.sh python test.py --ckpt checkpoints/latest_checkpoint.pth"
  exit 0
fi

exec "$@"
