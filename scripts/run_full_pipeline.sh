#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/damelikassym/retails-bilstm"
VENV_DIR="${PROJECT_DIR}/.venv"

cd "${PROJECT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Prefer CUDA-enabled PyTorch wheels that are compatible with modern NVIDIA drivers.
# If this fails in your environment, remove the index-url line and install your prebuilt torch package.
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
    print('torch_cuda', torch.version.cuda)
PY

RUN_DIR="runs/full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

retails-bilstm full \
  --config configs/default.yaml \
  --output-dir "${RUN_DIR}"

printf '\nDone. Outputs:\n  %s\n' "${RUN_DIR}"
printf 'Train summary: %s\n' "${RUN_DIR}/train/train_summary.json"
printf 'Eval report:   %s\n' "${RUN_DIR}/eval/eval_report.json"
