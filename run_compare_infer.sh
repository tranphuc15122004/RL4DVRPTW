#!/usr/bin/env bash
# ===========================================================================
# Compare AM vs PolyNet on the same data
# ===========================================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- Problem parameters ----
PROBLEM_TYPE="${PROBLEM_TYPE:-dvrptw}"
CUSTOMERS="${CUSTOMERS:-100}"
VEHICLES="${VEHICLES:-5}"
VEH_CAPA="${VEH_CAPA:-1300}"
VEH_SPEED="${VEH_SPEED:-1}"

# ---- Models ----
AM_WEIGHT="${AM_WEIGHT:-data/_AM/chkpt_best.pyth}"
POLYNET_WEIGHT="${POLYNET_WEIGHT:-data/_PolyNet/chkpt_best.pyth}"
AM_ARGS="${AM_ARGS:-}"                          # optional: AM training args.json
POLYNET_ARGS="${POLYNET_ARGS:-}"                # optional: PolyNet training args.json
MODEL_SIZE="${MODEL_SIZE:-128}"
LAYER_COUNT="${LAYER_COUNT:-3}"
HEAD_COUNT="${HEAD_COUNT:-8}"
FF_SIZE="${FF_SIZE:-512}"
TANH_XPLOR="${TANH_XPLOR:-10}"

# ---- Data ----
# MODE: single-csv, single-pyth, batch-csv, batch-pyth
MODE="${MODE:-batch-csv}"
DATA_PATH="${DATA_PATH:-data/datasets/100}"

# ---- Output ----
OUTPUT_DIR="${OUTPUT_DIR:-output/compare_infer}"

# ---- Misc ----
NO_CUDA="${NO_CUDA:-0}"
RNG_SEED="${RNG_SEED:-42}"
MAX_FILES="${MAX_FILES:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXTRA_ARGS=()
if [[ "$NO_CUDA" == "1" ]]; then
  EXTRA_ARGS+=(--no-cuda)
fi
if [[ -n "$RNG_SEED" ]]; then
  EXTRA_ARGS+=(--rng-seed "$RNG_SEED")
fi
if [[ -n "$MAX_FILES" ]]; then
  EXTRA_ARGS+=(--max-files "$MAX_FILES")
fi

# Use training args.json for auto-config if provided
if [[ -n "$AM_ARGS" ]]; then
  EXTRA_ARGS+=(--am-args "$AM_ARGS")
fi
if [[ -n "$POLYNET_ARGS" ]]; then
  EXTRA_ARGS+=(--polynet-args "$POLYNET_ARGS")
fi

echo "========================================"
echo "AM vs PolyNet Comparison"
echo "  Mode      : $MODE"
echo "  Data      : $DATA_PATH"
echo "  AM weight : $AM_WEIGHT"
echo "  PolyNet   : $POLYNET_WEIGHT"
echo "  Output    : $OUTPUT_DIR"
echo "========================================"

# Build data args based on MODE
DATA_ARGS=()
case "$MODE" in
  single-csv)   DATA_ARGS=(--data-csv "$DATA_PATH") ;;
  single-pyth)  DATA_ARGS=(--data-file "$DATA_PATH") ;;
  batch-csv)    DATA_ARGS=(--csv-dir "$DATA_PATH") ;;
  batch-pyth)   DATA_ARGS=(--pyth-dir "$DATA_PATH") ;;
esac

PYTHONPATH=. "$PYTHON_BIN" infer_batch.py \
  --model compare \
  --am-weight        "$AM_WEIGHT" \
  --polynet-weight   "$POLYNET_WEIGHT" \
  --problem-type     "$PROBLEM_TYPE" \
  --customers-count  "$CUSTOMERS" \
  --vehicles-count   "$VEHICLES" \
  --veh-capa         "$VEH_CAPA" \
  --veh-speed        "$VEH_SPEED" \
  --model-size       "$MODEL_SIZE" \
  --layer-count      "$LAYER_COUNT" \
  --head-count       "$HEAD_COUNT" \
  --ff-size          "$FF_SIZE" \
  --tanh-xplor       "$TANH_XPLOR" \
  --output-dir       "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "${DATA_ARGS[@]}"
