#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/pretrained/X-VLA-Pt"
PYTHON_BIN="${CSGO_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! "${PYTHON_BIN}" -c 'import sys' >/dev/null 2>&1; then
  echo "Python not found at ${PYTHON_BIN}; run scripts/setup_csgo_seen10.sh first." >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import huggingface_hub' >/dev/null 2>&1; then
  echo "huggingface_hub is missing in ${PYTHON_BIN}; complete scripts/setup_csgo_seen10.sh first." >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download the checkpoint; install curl and rerun this script." >&2
  exit 1
fi

# The Xet transfer path stalls on some proxy routes; force the official HTTP resolve path.
export HF_HUB_DISABLE_XET=1
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TRANSFORMERS_CACHE}" "${MODEL_DIR}"

"${PYTHON_BIN}" - "${MODEL_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "2toINF/X-VLA-Pt"
revision = "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b"
model_dir = Path(sys.argv[1])

print(f"Fetching {repo_id}@{revision} metadata and processor files", flush=True)
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=str(model_dir),
    allow_patterns=[".gitattributes", "*.md", "*.json", "*.py"],
    max_workers=4,
)
PY

WEIGHTS_URL="https://huggingface.co/2toINF/X-VLA-Pt/resolve/c1c4a64a7e03ac5b95c468bf1578f3d03651b53b/model.safetensors"
WEIGHTS_TMP="${MODEL_DIR}/model.safetensors.incomplete"
WEIGHTS_FILE="${MODEL_DIR}/model.safetensors"

if [[ ! -f "${WEIGHTS_FILE}" ]]; then
  CURL_ROUTE="${CSGO_CURL_ROUTE:-auto}"
  CURL_ROUTE_ARGS=()
  case "${CURL_ROUTE}" in
    auto)
      HTTP_CODE="$(curl --silent --fail --location --noproxy '*' \
        --connect-timeout 5 --max-time 10 --range 0-1023 \
        --output /dev/null --write-out '%{http_code}' "${WEIGHTS_URL}" 2>/dev/null || true)"
      if [[ "${HTTP_CODE}" == "206" ]]; then
        CURL_ROUTE_ARGS=(--noproxy '*')
        echo "Direct Hugging Face Range probe succeeded; downloading without the proxy."
      else
        echo "Direct Hugging Face Range probe unavailable; using configured proxy."
      fi
      ;;
    direct)
      CURL_ROUTE_ARGS=(--noproxy '*')
      ;;
    proxy)
      ;;
    *)
      echo "CSGO_CURL_ROUTE must be auto, direct, or proxy (got ${CURL_ROUTE})." >&2
      exit 1
      ;;
  esac
  echo "Downloading model.safetensors from the pinned official resolve URL (resumable)"
  EXPECTED_WEIGHTS_SIZE=3519068172
  MAX_DOWNLOAD_ATTEMPTS="${CSGO_DOWNLOAD_ATTEMPTS:-8}"
  for attempt in $(seq 1 "${MAX_DOWNLOAD_ATTEMPTS}"); do
    current_size=0
    if [[ -f "${WEIGHTS_TMP}" ]]; then
      current_size="$(stat -c '%s' "${WEIGHTS_TMP}")"
    fi
    if (( current_size > EXPECTED_WEIGHTS_SIZE )); then
      echo "Partial weight exceeds expected size: ${current_size} bytes." >&2
      exit 1
    fi
    if (( current_size == EXPECTED_WEIGHTS_SIZE )); then
      break
    fi

    # A fresh resolve query forces fresh signed CDN metadata on every resume.
    DOWNLOAD_URL="${WEIGHTS_URL}?download=true&cachebust=${attempt}-$(date +%s%N)"
    echo "Download attempt ${attempt}/${MAX_DOWNLOAD_ATTEMPTS} from byte ${current_size}."
    if curl "${CURL_ROUTE_ARGS[@]}" --http1.1 --fail --location \
      --connect-timeout 10 --speed-time 90 --speed-limit 1024 \
      --continue-at - --progress-bar \
      --output "${WEIGHTS_TMP}" "${DOWNLOAD_URL}"; then
      curl_status=0
    else
      curl_status=$?
    fi

    current_size=0
    if [[ -f "${WEIGHTS_TMP}" ]]; then
      current_size="$(stat -c '%s' "${WEIGHTS_TMP}")"
    fi
    if (( current_size == EXPECTED_WEIGHTS_SIZE )); then
      break
    fi
    if (( current_size > EXPECTED_WEIGHTS_SIZE )); then
      echo "Partial weight exceeds expected size: ${current_size} bytes." >&2
      exit 1
    fi
    echo "curl exited with status ${curl_status}; preserving ${current_size}/${EXPECTED_WEIGHTS_SIZE} bytes."
    if (( attempt < MAX_DOWNLOAD_ATTEMPTS )); then
      sleep 5
    fi
  done

  current_size=0
  if [[ -f "${WEIGHTS_TMP}" ]]; then
    current_size="$(stat -c '%s' "${WEIGHTS_TMP}")"
  fi
  if (( current_size != EXPECTED_WEIGHTS_SIZE )); then
    echo "Checkpoint download stopped before completion at ${current_size}/${EXPECTED_WEIGHTS_SIZE} bytes." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" - "${MODEL_DIR}" <<'PY'
import hashlib
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
weights = model_dir / "model.safetensors"
temporary = model_dir / "model.safetensors.incomplete"
expected_size = 3_519_068_172
expected_sha256 = "433acffc992f457b1737d35ae20f81a895521f38eadaf03c736b6c0b0af1c93e"

candidate = weights if weights.is_file() else temporary
if not candidate.is_file():
    raise SystemExit(f"Checkpoint weight file is missing; expected {temporary}")
if candidate.stat().st_size != expected_size:
    raise SystemExit(
        f"Incomplete checkpoint at {candidate}: has {candidate.stat().st_size} bytes; expected {expected_size}"
    )

digest = hashlib.sha256()
with candidate.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
actual_sha256 = digest.hexdigest()
print(f"model.safetensors SHA-256: {actual_sha256}", flush=True)
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"Checkpoint SHA-256 mismatch; rejected file is preserved at {candidate}"
    )

if candidate == temporary:
    temporary.replace(weights)

for partial in (model_dir / ".cache/huggingface/download").glob("*.incomplete"):
    partial.unlink(missing_ok=True)
for stale in (model_dir / ".cache/huggingface/download").glob("*.incomplete.aria2"):
    stale.unlink(missing_ok=True)
print(f"Verified 2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b", flush=True)
PY
