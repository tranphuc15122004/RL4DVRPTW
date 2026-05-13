#!/usr/bin/env bash
# ===========================================================================
# AM Inference Script
# Supports: single CSV, single .pyth, batch CSV directory
# ===========================================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Problem parameters ----
PROBLEM_TYPE="${PROBLEM_TYPE:-dvrptw}"
CUSTOMERS="${CUSTOMERS:-400}"
VEHICLES="${VEHICLES:-20}"
VEH_CAPA="${VEH_CAPA:-200}"
VEH_SPEED="${VEH_SPEED:-1}"

# ---- Model ----
MODEL_WEIGHT="${MODEL_WEIGHT:-data/_AM/chkpt_best.pyth}"
MODEL_ARGS="${MODEL_ARGS:-}"                     # optional: path to training args.json
MODEL_SIZE="${MODEL_SIZE:-128}"
LAYER_COUNT="${LAYER_COUNT:-5}"
HEAD_COUNT="${HEAD_COUNT:-8}"
FF_SIZE="${FF_SIZE:-256}"
TANH_XPLOR="${TANH_XPLOR:-10}"

# ---- Inference mode ----
# MODE can be: single-csv, single-pyth, batch-csv, batch-pyth
MODE="${MODE:-batch-csv}"
DATA_PATH="${DATA_PATH:-data/datasets/h400}"

# ---- Output ----
OUTPUT_DIR="${OUTPUT_DIR:-infer/am_infer_400_cap200_s}"
SAVE_JSON="${SAVE_JSON:-}"
VERIFY="${VERIFY:-1}"
PRINT_INSTANCES="${PRINT_INSTANCES:-3}"
STOCH_ROLLOUTS="${STOCH_ROLLOUTS:-100}"

# ---- Decode ----
DECODE="${DECODE:-sample}"   # greedy or sample

# ---- Misc ----
NO_CUDA="${NO_CUDA:-0}"
RNG_SEED="${RNG_SEED:-}"
NO_NORMALIZE="${NO_NORMALIZE:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXTRA_ARGS=()

if [[ "$NO_CUDA" == "1" ]]; then
  EXTRA_ARGS+=(--no-cuda)
fi
if [[ -n "$RNG_SEED" ]]; then
  EXTRA_ARGS+=(--rng-seed "$RNG_SEED")
fi
if [[ "$NO_NORMALIZE" == "1" ]]; then
  EXTRA_ARGS+=(--no-normalize)
fi

# Decode flags and other single-instance args will be added conditionally below

# Use training args.json for auto-config if provided
if [[ -n "$MODEL_ARGS" ]]; then
  EXTRA_ARGS+=(--model-args "$MODEL_ARGS")
fi

# Parse MODE and DATA_PATH to decide which script+args to use
case "$MODE" in
  single-csv|single-pyth)
    # Use the dedicated single-instance script
    CMD="am/infer_single.py"
    
    EXTRA_ARGS+=(--max-print-instances "$PRINT_INSTANCES")
    EXTRA_ARGS+=(--stoch-rollouts "$STOCH_ROLLOUTS")
    
    if [[ "$DECODE" == "sample" ]]; then
      EXTRA_ARGS+=(--sample)
    else
      EXTRA_ARGS+=(--greedy)
    fi

    if [[ "$MODE" == "single-csv" ]]; then
      EXTRA_ARGS+=(--data-csv "$DATA_PATH")
    else
      EXTRA_ARGS+=(--data-file "$DATA_PATH")
    fi
    if [[ -n "$SAVE_JSON" ]]; then
      EXTRA_ARGS+=(--save-json "$SAVE_JSON")
    fi
    ;;
  batch-csv)
    CMD="infer_batch.py"
    EXTRA_ARGS+=(--model am --model-weight "$MODEL_WEIGHT")
    EXTRA_ARGS+=(--csv-dir "$DATA_PATH")
    EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
    if [[ -n "$SAVE_JSON" ]]; then
      EXTRA_ARGS+=(--csv-output "$SAVE_JSON")
    fi
    ;;
  batch-pyth)
    CMD="infer_batch.py"
    EXTRA_ARGS+=(--model am --model-weight "$MODEL_WEIGHT")
    EXTRA_ARGS+=(--pyth-dir "$DATA_PATH")
    EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
    if [[ -n "$SAVE_JSON" ]]; then
      EXTRA_ARGS+=(--csv-output "$SAVE_JSON")
    fi
    ;;
  *)
    echo "Unknown MODE: $MODE"
    echo "Valid modes: single-csv, single-pyth, batch-csv, batch-pyth"
    exit 1
    ;;
esac

echo "========================================"
echo "AM Inference"
echo "  Mode      : $MODE"
echo "  Data      : $DATA_PATH"
echo "  Model     : $MODEL_WEIGHT"
echo "  Problem   : $PROBLEM_TYPE (n=$CUSTOMERS, m=$VEHICLES)"
echo "  Decode    : $DECODE"
echo "========================================"

PYTHONPATH=. "$PYTHON_BIN" "$CMD" \
  --problem-type      "$PROBLEM_TYPE" \
  --customers-count   "$CUSTOMERS" \
  --vehicles-count    "$VEHICLES" \
  --veh-capa          "$VEH_CAPA" \
  --veh-speed         "$VEH_SPEED" \
  --model-size        "$MODEL_SIZE" \
  --layer-count       "$LAYER_COUNT" \
  --head-count        "$HEAD_COUNT" \
  --ff-size           "$FF_SIZE" \
  --tanh-xplor        "$TANH_XPLOR" \
  --model-weight      "$MODEL_WEIGHT" \
  "${EXTRA_ARGS[@]}"
