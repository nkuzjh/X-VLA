#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

cd "${PROJECT_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python 3.12 is required (override with PYTHON_BIN=/path/to/python3.12)." >&2
    exit 1
  fi
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    # Some minimal Debian Python installs omit ensurepip/python3-venv.
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
  fi
fi

"${VENV_DIR}/bin/python" - <<'PY'
import sys
import sysconfig
from pathlib import Path

cfg = Path(sys.prefix) / "pyvenv.cfg"
if not cfg.is_file() or "include-system-site-packages = false" not in cfg.read_text().lower():
    raise SystemExit(".venv must be an isolated venv with system site packages disabled")
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12 in .venv, got {sys.version.split()[0]}")
print(f"Using isolated Python {sys.version.split()[0]} at {sys.prefix}")
PY

# Keep model downloads and all Hugging Face caches inside this project.
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
WHEELHOUSE_DIR="${PROJECT_ROOT}/.cache/wheels"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-${WHEELHOUSE_DIR}}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TRANSFORMERS_CACHE}" "${PIP_CACHE_DIR}" "${WHEELHOUSE_DIR}"

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  # Bootstrap the pinned pip into the project venv, never into a shared env.
  if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip --python "${VENV_DIR}" install 'pip==25.1.1'
  else
    BOOTSTRAP_DIR="${PROJECT_ROOT}/.cache/bootstrap"
    mkdir -p "${BOOTSTRAP_DIR}"
    "${VENV_DIR}/bin/python" - "${BOOTSTRAP_DIR}/get-pip.py" <<'PY'
import sys
import urllib.request
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", sys.argv[1])
PY
    "${VENV_DIR}/bin/python" "${BOOTSTRAP_DIR}/get-pip.py" 'pip==25.1.1'
  fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade 'pip==25.1.1'

# If exact model/CUDA wheels are available locally, install those exact files
# before dependency resolution so pip does not choose equal-version index
# candidates and download the same multi-gigabyte files again.
LOCAL_WHEEL_PINS=(
  'torch==2.7.1+cu128'
  'torchvision==0.22.1+cu128'
  'nvidia-cublas-cu12==12.8.3.14'
  'nvidia-cuda-cupti-cu12==12.8.57'
  'nvidia-cuda-nvrtc-cu12==12.8.61'
  'nvidia-cuda-runtime-cu12==12.8.57'
  'nvidia-cudnn-cu12==9.7.1.26'
  'nvidia-cufft-cu12==11.3.3.41'
  'nvidia-curand-cu12==10.3.9.55'
  'nvidia-cusolver-cu12==11.7.2.55'
  'nvidia-cusparse-cu12==12.5.7.53'
  'nvidia-cusparselt-cu12==0.6.3'
  'nvidia-nccl-cu12==2.26.2'
  'nvidia-nvtx-cu12==12.8.55'
  'nvidia-nvjitlink-cu12==12.8.61'
  'nvidia-cufile-cu12==1.13.0.11'
)
AVAILABLE_LOCAL_WHEEL_PINS=()
for pin in "${LOCAL_WHEEL_PINS[@]}"; do
  package="${pin%%==*}"
  version="${pin##*==}"
  wheel_pattern="${WHEELHOUSE_DIR}/${package//-/_}-${version}-*.whl"
  if compgen -G "${wheel_pattern}" >/dev/null; then
    AVAILABLE_LOCAL_WHEEL_PINS+=("${pin}")
  fi
done
if (( ${#AVAILABLE_LOCAL_WHEEL_PINS[@]} > 0 )); then
  echo "Installing available pinned model/CUDA wheels from ${WHEELHOUSE_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --no-index --no-deps \
    --find-links "${WHEELHOUSE_DIR}" "${AVAILABLE_LOCAL_WHEEL_PINS[@]}"
fi

"${VENV_DIR}/bin/python" -m pip install -r requirements-csgo.txt

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import torchvision

if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected PyTorch CUDA 12.8, found {torch.version.cuda!r}")
print(f"PyTorch {torch.__version__}; torchvision {torchvision.__version__}; CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}; capability: {torch.cuda.get_device_capability(0)}")
PY

echo "CSGO venv installed at ${VENV_DIR}"
