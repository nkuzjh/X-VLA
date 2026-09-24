#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

usage() {
  cat <<'EOF'
Usage: scripts/setup_csgo_seen10.sh [--check-only]

Create the project .venv with Python 3.10, 3.11, or 3.12 (3.12 preferred)
and install the pinned CSGO dependencies. Set PYTHON_BIN to select a
specific interpreter for a new .venv. If none is available, Conda can provision Python 3.12
inside .cache/csgo-python without changing the active environment.

--check-only  Check the existing venv or show which Python would be used;
              do not create environments or install packages.
--help        Show this help.
EOF
}

CHECK_ONLY=0
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --check-only) CHECK_ONLY=1; shift ;;
esac
if (( $# != 0 )); then
  usage >&2
  echo "Unknown argument: $1" >&2
  exit 2
fi

python_version() {
  "$1" -c 'import sys; print(".".join(map(str, sys.version_info[:2])) if (3, 10) <= sys.version_info[:2] <= (3, 12) else "")' 2>/dev/null
}

validate_python() {
  local candidate="$1" version
  if ! command -v "${candidate}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${candidate}. Set PYTHON_BIN to an installed Python 3.10–3.12 executable." >&2
    return 1
  fi
  version="$(python_version "${candidate}")" || true
  if [[ -z "${version}" ]]; then
    echo "${candidate} is not a working Python 3.10–3.12 interpreter. Set PYTHON_BIN to a compatible executable." >&2
    return 1
  fi
  echo "${version}"
}

check_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Existing ${VENV_DIR} has no executable bin/python (possibly incomplete or moved). It was preserved; repair or move it before retrying." >&2
    return 1
  fi
  "${VENV_DIR}/bin/python" - "${VENV_DIR}" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit("Existing .venv points to another location (possibly moved); it was preserved. Repair or move it before retrying.")
cfg = Path(sys.prefix) / "pyvenv.cfg"
if not cfg.is_file() or "include-system-site-packages = false" not in cfg.read_text().lower():
    raise SystemExit("Existing .venv is not an isolated venv; it was preserved. Repair or move it before retrying.")
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit(f"Existing .venv uses unsupported Python {sys.version.split()[0]}; it was preserved. Use Python 3.10–3.12.")
print(f"Using isolated Python {sys.version.split()[0]} at {sys.prefix}")
PY
}

cd "${PROJECT_ROOT}"

SELECTED_PYTHON=""
SELECTED_VERSION=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  SELECTED_VERSION="$(validate_python "${PYTHON_BIN}")"
  SELECTED_PYTHON="${PYTHON_BIN}"
fi

if [[ -e "${VENV_DIR}" || -L "${VENV_DIR}" ]]; then
  check_venv
  if [[ -n "${SELECTED_VERSION}" && "$(python_version "${VENV_DIR}/bin/python")" != "${SELECTED_VERSION}" ]]; then
    echo "Existing .venv uses a different Python version than PYTHON_BIN=${PYTHON_BIN}; it was preserved. Move it before creating a new .venv." >&2
    exit 1
  fi
  if (( CHECK_ONLY )); then exit 0; fi
else
  if [[ -z "${SELECTED_PYTHON}" ]]; then
    for candidate in python3.12 python3.11 python3.10 python3 python; do
      if command -v "${candidate}" >/dev/null 2>&1 && [[ -n "$(python_version "${candidate}" || true)" ]]; then
        SELECTED_PYTHON="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${SELECTED_PYTHON}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "No Python 3.10–3.12 found. Install one or Conda, then rerun; PYTHON_BIN can select an installed interpreter." >&2
      exit 1
    fi
    SELECTED_PYTHON="${PROJECT_ROOT}/.cache/csgo-python/bin/python"
    if [[ -e "${PROJECT_ROOT}/.cache/csgo-python" || -L "${PROJECT_ROOT}/.cache/csgo-python" ]]; then
      validate_python "${SELECTED_PYTHON}" >/dev/null || {
        echo "Existing Conda prefix was preserved; repair or move ${PROJECT_ROOT}/.cache/csgo-python before retrying." >&2
        exit 1
      }
    else
      if (( CHECK_ONLY )); then
        echo "Would provision Python 3.12 with Conda at ${PROJECT_ROOT}/.cache/csgo-python"
        exit 0
      fi
      mkdir -p "${PROJECT_ROOT}/.cache"
      conda create --yes --prefix "${PROJECT_ROOT}/.cache/csgo-python" python=3.12 || {
        echo "Conda could not provision project-local Python 3.12; check Conda configuration/network, or set PYTHON_BIN." >&2
        exit 1
      }
      validate_python "${SELECTED_PYTHON}" >/dev/null
    fi
  fi
  echo "Selected $(validate_python "${SELECTED_PYTHON}") at ${SELECTED_PYTHON}"
  if (( CHECK_ONLY )); then exit 0; fi
  # --without-pip also works on Debian installs without python3-venv/ensurepip.
  if ! "${SELECTED_PYTHON}" -m venv --without-pip "${VENV_DIR}"; then
    echo "Could not create ${VENV_DIR}. Any partial directory was preserved; check Python's venv module and permissions." >&2
    exit 1
  fi
  check_venv
fi

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
  # Bootstrap pip only in this project's isolated venv.
  if ! "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
    BOOTSTRAPPED=0
    if [[ -n "${SELECTED_PYTHON}" ]] &&
       "${SELECTED_PYTHON}" -m pip --version >/dev/null 2>&1 &&
       "${SELECTED_PYTHON}" -m pip --help | grep -q -- '--python'; then
      if "${SELECTED_PYTHON}" -m pip --python "${VENV_DIR}" install 'pip==25.1.1'; then
        BOOTSTRAPPED=1
      fi
    fi
    if (( ! BOOTSTRAPPED )); then
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
WHEEL_PATHS_OUTPUT="$("${VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/select_compatible_csgo_wheels.py" "${WHEELHOUSE_DIR}" "${LOCAL_WHEEL_PINS[@]}")"
AVAILABLE_LOCAL_WHEELS=()
if [[ -n "${WHEEL_PATHS_OUTPUT}" ]]; then
  mapfile -t AVAILABLE_LOCAL_WHEELS <<< "${WHEEL_PATHS_OUTPUT}"
fi
if (( ${#AVAILABLE_LOCAL_WHEELS[@]} > 0 )); then
  echo "Installing available pinned model/CUDA wheels from ${WHEELHOUSE_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --no-index --no-deps \
    "${AVAILABLE_LOCAL_WHEELS[@]}"
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
