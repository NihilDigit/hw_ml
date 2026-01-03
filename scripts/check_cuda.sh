#!/usr/bin/env bash
set -euo pipefail

echo '== nvidia-smi =='
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo 'nvidia-smi not found'
fi

echo

echo '== nvcc =='
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo 'nvcc not found'
fi

echo

echo '== torch cuda =='
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device', torch.cuda.get_device_name(0))
    print('cuda version', torch.version.cuda)
PY

echo

echo '== cuml import =='
python - <<'PY'
try:
    import cuml
    from cuml.manifold import TSNE
    print('cuml', cuml.__version__)
    print('TSNE', TSNE)
except Exception as e:
    print('cuml import failed:', type(e).__name__, e)
PY
