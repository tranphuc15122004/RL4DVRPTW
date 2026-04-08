#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Problem
PROBLEM_TYPE="${PROBLEM_TYPE:-dvrptw}"
CUSTOMERS="${CUSTOMERS:-50}"
VEHICLES="${VEHICLES:-3}"

# Training schedule
EPOCHS="${EPOCHS:-500}"
ITERS="${ITERS:-100}"
BATCH="${BATCH:-128}"
TEST_BATCH="${TEST_BATCH:-100}"
LR="${LR:-1e-4}"
RATE_DECAY="${RATE_DECAY:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-2}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Attention model
MODEL_SIZE="${MODEL_SIZE:-128}"
LAYER_COUNT="${LAYER_COUNT:-3}"
HEAD_COUNT="${HEAD_COUNT:-8}"
FF_SIZE="${FF_SIZE:-512}"
TANH_XPLOR="${TANH_XPLOR:-10}"

# Baseline / optimization
BASELINE_TYPE="${BASELINE_TYPE:-critic}"
CRITIC_LR="${CRITIC_LR:-1e-3}"
CRITIC_DECAY="${CRITIC_DECAY:-}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"

# Boolean flags: set to 1 to enable, 0 to disable
AMP="${AMP:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
RESOURCE_SAFE="${RESOURCE_SAFE:-1}"
ADV_NORM="${ADV_NORM:-1}"
NO_CUDA="${NO_CUDA:-0}"
REGEN_TRAIN_DATA_EACH_EPOCH="${REGEN_TRAIN_DATA_EACH_EPOCH:-0}"

# Checkpoint / reproducibility
RESUME="${RESUME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RNG_SEED="${RNG_SEED:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EXTRA_ARGS=()
if [[ -n "$RESUME" ]]; then
  EXTRA_ARGS+=(--resume-state "$RESUME")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$RNG_SEED" ]]; then
  EXTRA_ARGS+=(--rng-seed "$RNG_SEED")
fi
if [[ -n "$RATE_DECAY" ]]; then
  EXTRA_ARGS+=(--rate-decay "$RATE_DECAY")
fi
if [[ -n "$CRITIC_DECAY" ]]; then
  EXTRA_ARGS+=(--critic-decay "$CRITIC_DECAY")
fi
if [[ "$AMP" == "1" ]]; then
  EXTRA_ARGS+=(--amp)
fi
if [[ "$PIN_MEMORY" == "1" ]]; then
  EXTRA_ARGS+=(--pin-memory)
fi
if [[ "$RESOURCE_SAFE" == "1" ]]; then
  EXTRA_ARGS+=(--resource-safe)
fi
if [[ "$ADV_NORM" == "1" ]]; then
  EXTRA_ARGS+=(--adv-norm)
fi
if [[ "$NO_CUDA" == "1" ]]; then
  EXTRA_ARGS+=(--no-cuda)
fi
if [[ "$REGEN_TRAIN_DATA_EACH_EPOCH" == "1" ]]; then
  EXTRA_ARGS+=(--regen-train-data-each-epoch)
fi

PYTHONPATH=. "$PYTHON_BIN" -m am.train \
  --problem-type      "$PROBLEM_TYPE" \
  --customers-count   "$CUSTOMERS" \
  --vehicles-count    "$VEHICLES" \
  --epoch-count       "$EPOCHS" \
  --iter-count        "$ITERS" \
  --batch-size        "$BATCH" \
  --test-batch-size   "$TEST_BATCH" \
  --learning-rate     "$LR" \
  --weight-decay      "$WEIGHT_DECAY" \
  --max-grad-norm     "$MAX_GRAD_NORM" \
  --checkpoint-period "$CHECKPOINT_PERIOD" \
  --num-workers       "$NUM_WORKERS" \
  --model-size        "$MODEL_SIZE" \
  --layer-count       "$LAYER_COUNT" \
  --head-count        "$HEAD_COUNT" \
  --ff-size           "$FF_SIZE" \
  --tanh-xplor        "$TANH_XPLOR" \
  --baseline-type     "$BASELINE_TYPE" \
  --critic-rate       "$CRITIC_LR" \
  --entropy-coef      "$ENTROPY_COEF" \
  "${EXTRA_ARGS[@]}" \
  "$@"
