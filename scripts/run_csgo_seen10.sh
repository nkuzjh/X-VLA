#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${CSGO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
UNILIP_PYTHON="${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}"
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-$PROJECT_ROOT/../csgo_benchmark_v2_eval_general}"
EVALUATOR="$SHARED_EVAL_DIR/run_eval.py"
CONFIG="$PROJECT_ROOT/configs/csgo_seen10.json"
PRETRAINED="$PROJECT_ROOT/pretrained/X-VLA-Pt"
SEED=0
OUTPUT_ROOT=""
EXTRA_ARGS=()
export PYTHONDONTWRITEBYTECODE=1

usage() {
    cat <<'EOF'
Usage: bash scripts/run_csgo_seen10.sh {train|infer|eval|smoke} [--seed N]
  --config FILE        Configuration file
  --data-root DIR      Published, read-only Benchmark v2 data
  --pretrained DIR     Native pretrained X-VLA (train/smoke)
  --output-root DIR    Exact seed directory (smoke requires a separate path)
  Other options are forwarded to train_seen10.py or infer_seen10.py.
  CSGO_PYTHON, UNILIP_PYTHON and SHARED_EVAL_DIR can override local paths.
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 2; fi
COMMAND="$1"
shift
case "$COMMAND" in
    train|infer|eval|smoke) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed|--config|--data-root|--pretrained|--output-root)
            if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --seed) SEED="$2" ;;
                --config) CONFIG="$2" ;;
                --data-root) DATA_ROOT="$2" ;;
                --pretrained) PRETRAINED="$2" ;;
                --output-root) OUTPUT_ROOT="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then echo "--seed must be a nonnegative integer" >&2; exit 2; fi
check_model_python() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "Project Python is missing: $PYTHON. Run scripts/setup_csgo_seen10.sh first." >&2
        return 1
    fi
}

check_evaluator() {
    if [[ ! -f "$EVALUATOR" ]]; then
        echo "Shared CSGO evaluator is missing: $EVALUATOR" >&2
        echo "Set SHARED_EVAL_DIR to a directory containing run_eval.py." >&2
        return 1
    fi
    if [[ ! -x "$UNILIP_PYTHON" ]]; then
        echo "UniLIP evaluator Python is missing: $UNILIP_PYTHON" >&2
        return 1
    fi
}

case "$COMMAND" in
    train|infer) check_model_python || exit 2 ;;
    eval|smoke)
        check_evaluator || exit 2
        if [[ "$COMMAND" == smoke ]]; then check_model_python || exit 2; fi
        ;;
esac

FORMAL_ROOT="$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10/X-VLA/seed_$SEED"
if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ "$COMMAND" == smoke ]]; then
        OUTPUT_ROOT="$PROJECT_ROOT/outputs/csgo_benchmark_v2_seen10_smoke/X-VLA/seed_$SEED/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    else
        OUTPUT_ROOT="$FORMAL_ROOT"
    fi
fi
COMMON=(--config "$CONFIG" --seed "$SEED" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT")

evaluate() {
    local smoke_mode="$1"
    local metrics_dir="$OUTPUT_ROOT/metrics/localization"
    if [[ -e "$metrics_dir" ]]; then echo "Refusing to overwrite existing metrics: $metrics_dir" >&2; return 2; fi
    if [[ "$smoke_mode" == 1 ]]; then
        local smoke_metrics="$OUTPUT_ROOT/metrics/localization_smoke.json"
        local smoke_tmp
        if [[ -e "$smoke_metrics" ]]; then echo "Refusing to overwrite existing smoke metrics: $smoke_metrics" >&2; return 2; fi
        mkdir -p "$(dirname "$smoke_metrics")"
        smoke_tmp="$(mktemp "$OUTPUT_ROOT/metrics/.localization_smoke.XXXXXX")"
        # The shared evaluator reads a global GT prefix. The model smoke emits
        # one prediction per map, so its first row is the matching prefix of 1.
        if ! "$UNILIP_PYTHON" "$EVALUATOR" smoke localization \
            --pred-root "$OUTPUT_ROOT/localization" --data-root "$DATA_ROOT" \
            --limit 1 > "$smoke_tmp"; then
            rm -f "$smoke_tmp"
            return 2
        fi
        mv "$smoke_tmp" "$smoke_metrics"
        cat "$smoke_metrics"
        echo "Nonformal localization smoke metrics (smoke_only=true): $smoke_metrics"
        return 0
    else
        if [[ -e "$OUTPUT_ROOT/NON_FORMAL_SMOKE.json" ]]; then
            echo "A smoke output directory cannot be evaluated as a formal result: $OUTPUT_ROOT" >&2
            return 2
        fi
    fi
    "$UNILIP_PYTHON" "$EVALUATOR" localization \
        --pred-root "$OUTPUT_ROOT/localization" --data-root "$DATA_ROOT" \
        --output "$metrics_dir"
}

case "$COMMAND" in
    train)
        exec "$PYTHON" train_seen10.py "${COMMON[@]}" --pretrained "$PRETRAINED" "${EXTRA_ARGS[@]}"
        ;;
    infer)
        exec "$PYTHON" infer_seen10.py "${COMMON[@]}" "${EXTRA_ARGS[@]}"
        ;;
    eval)
        if [[ ${#EXTRA_ARGS[@]} -ne 0 ]]; then echo "Unsupported eval arguments: ${EXTRA_ARGS[*]}" >&2; exit 2; fi
        evaluate 0
        ;;
    smoke)
        if [[ ${#EXTRA_ARGS[@]} -ne 0 ]]; then echo "Unsupported smoke arguments: ${EXTRA_ARGS[*]}" >&2; exit 2; fi
        "$PYTHON" - "$PROJECT_ROOT" "$OUTPUT_ROOT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

project, output, seed = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), int(sys.argv[3])
formal = (project / "outputs" / "csgo_benchmark_v2_seen10").resolve()
if output == formal or formal in output.parents:
    raise SystemExit("Smoke requires an output directory outside the formal result tree")
if output.exists():
    raise SystemExit(f"Refusing to overwrite an existing smoke directory: {output}")
output.mkdir(parents=True)
(output / "NON_FORMAL_SMOKE.json").write_text(json.dumps({
    "non_formal_smoke": True, "formal_benchmark_result": False, "seed": seed,
    "train_steps": 2, "inference_samples_per_map": 1,
}) + "\n")
PY
        echo "Nonformal model smoke output: $OUTPUT_ROOT"
        "$PYTHON" train_seen10.py "${COMMON[@]}" --pretrained "$PRETRAINED" \
            --smoke --limit-per-map 1 --iters 1 --eval-interval 1
        "$PYTHON" train_seen10.py "${COMMON[@]}" --pretrained "$PRETRAINED" \
            --smoke --limit-per-map 1 --iters 2 --eval-interval 1 \
            --resume "$OUTPUT_ROOT/checkpoints/step_00000001"
        "$PYTHON" infer_seen10.py "${COMMON[@]}" --smoke --limit-per-map 1
        echo "Model forward/backward, save/reload, resume and 10 predictions completed: $OUTPUT_ROOT"
        evaluate 1
        ;;
esac
