#!/usr/bin/env bash
set -euo pipefail

AICITY_ROOT="${AICITY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COSMOS_ROOT="${COSMOS_ROOT:-$AICITY_ROOT/cosmos/packages/cosmos3}"
ACTIVATE_COSMOS3="${ACTIVATE_COSMOS3:-}"
LOG_DIR="${LOG_DIR:-$AICITY_ROOT/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

RUN="${RUN:-c3super1680}"
PROJECT="${PROJECT:-aicity_track5}"
GROUP="${GROUP:-track5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$AICITY_ROOT/outputs}"
OUT="$OUTPUT_ROOT/$PROJECT/$GROUP/$RUN"
LOG="$LOG_DIR/${RUN}.log"
PIDFILE="$LOG_DIR/${RUN}.pid"
LAUNCH_ENV="$LOG_DIR/${RUN}.launch.env"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$COSMOS_ROOT/examples/checkpoints/Cosmos3-Super}"
WAN_VAE="${WAN_VAE:-$COSMOS_ROOT/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
SFT_TOML="${SFT_TOML:-$COSMOS_ROOT/examples/toml/sft_config/vision_sft_super.toml}"

WTS_ROOT="${WTS_ROOT:-$AICITY_ROOT/data/wts}"
BDD_ROOT="${BDD_ROOT:-$AICITY_ROOT/data/bdd}"
WTS_JSONL="${WTS_JSONL:-$WTS_ROOT/train/video_dataset_file.jsonl}"
BDD_JSONL="${BDD_JSONL:-$BDD_ROOT/train/video_dataset_file.jsonl}"
TRAIN_JSONLS="$WTS_JSONL,$BDD_JSONL"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-50133}"
MAX_ITER="${MAX_ITER:-1680}"
SAVE_ITER="${SAVE_ITER:-100}"
EVAL_SAVE_ITER="${EVAL_SAVE_ITER:-20}"
LR="${LR:-5e-4}"
CFG_DROPOUT_RATE="${CFG_DROPOUT_RATE:-0.05}"
CONDITIONING_CONFIG="${CONDITIONING_CONFIG:-0:0.01,1:0.04,5:0.10,8:0.20,10:0.25,12:0.40}"
SCHEDULER_CYCLE_LENGTHS="${SCHEDULER_CYCLE_LENGTHS:-1000000}"
SCHEDULER_WARM_UP_STEPS="${SCHEDULER_WARM_UP_STEPS:-50}"
TOKENIZER_CHUNK_DURATION="${TOKENIZER_CHUNK_DURATION:-165}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_MODE="${WANDB_MODE:-disabled}"
RESUME_CHECKPOINT_PATH="${RESUME_CHECKPOINT_PATH:-}"
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-1}"
NO_STRICT_RESUME="${NO_STRICT_RESUME:-0}"
DRYRUN="${DRYRUN:-0}"
# Optional Hydra-style overrides appended after `--`, e.g.
# EXTRA_OVERRIDES='model.config.max_num_tokens_after_packing=65536 dataloader_train.max_sequence_length=65536'
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

mkdir -p "$LOG_DIR"

require_file() { [[ -f "$1" ]] || { echo "ERROR missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "ERROR missing dir: $1" >&2; exit 2; }; }

require_file "$SFT_TOML"
require_file "$WAN_VAE"
require_file "$WTS_JSONL"
require_file "$BDD_JSONL"
require_dir "$BASE_CHECKPOINT"
if [[ ! -f "$BASE_CHECKPOINT/model/.metadata" ]]; then
  echo "ERROR Cosmos3-Super DCP checkpoint is not ready: $BASE_CHECKPOINT/model/.metadata missing" >&2
  echo "Run: python -m cosmos_framework.scripts.convert_model_to_dcp -o $BASE_CHECKPOINT --checkpoint-path Cosmos3-Super" >&2
  exit 3
fi

{
  echo "RUN=$RUN"
  echo "PROJECT=$PROJECT"
  echo "GROUP=$GROUP"
  echo "OUT=$OUT"
  echo "LOG=$LOG"
  echo "PIDFILE=$PIDFILE"
  echo "BASE_CHECKPOINT=$BASE_CHECKPOINT"
  echo "WAN_VAE=$WAN_VAE"
  echo "SFT_TOML=$SFT_TOML"
  echo "WTS_JSONL=$WTS_JSONL"
  echo "BDD_JSONL=$BDD_JSONL"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "NPROC_PER_NODE=$NPROC_PER_NODE"
  echo "MASTER_PORT=$MASTER_PORT"
  echo "MAX_ITER=$MAX_ITER"
  echo "SAVE_ITER=$SAVE_ITER"
  echo "EVAL_SAVE_ITER=$EVAL_SAVE_ITER"
  echo "LR=$LR"
  echo "CFG_DROPOUT_RATE=$CFG_DROPOUT_RATE"
  echo "CONDITIONING_CONFIG=$CONDITIONING_CONFIG"
  echo "SCHEDULER_CYCLE_LENGTHS=$SCHEDULER_CYCLE_LENGTHS"
  echo "SCHEDULER_WARM_UP_STEPS=$SCHEDULER_WARM_UP_STEPS"
  echo "TOKENIZER_CHUNK_DURATION=$TOKENIZER_CHUNK_DURATION"
  echo "NUM_VIDEO_FRAMES=$NUM_VIDEO_FRAMES"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "WANDB_MODE=$WANDB_MODE"
  echo "RESUME_CHECKPOINT_PATH=$RESUME_CHECKPOINT_PATH"
  echo "RESUME_TRAINING_STATE=$RESUME_TRAINING_STATE"
  echo "NO_STRICT_RESUME=$NO_STRICT_RESUME"
} > "$LAUNCH_ENV"

if [[ "$RESUME_TRAINING_STATE" == "1" && -z "$RESUME_CHECKPOINT_PATH" && -f "$OUT/checkpoints/latest_checkpoint.txt" ]]; then
  RESUME_CHECKPOINT_PATH="$OUT"
fi
if [[ -n "$RESUME_CHECKPOINT_PATH" ]]; then
  require_dir "$RESUME_CHECKPOINT_PATH"
fi

cmd=(
  torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT"
  "$AICITY_ROOT/scripts/train.py"
  --sft-toml "$SFT_TOML"
  --dataset-path "$WTS_ROOT"
  --train-jsonl-paths "$TRAIN_JSONLS"
  --base-checkpoint-path "$BASE_CHECKPOINT"
  --wan-vae-path "$WAN_VAE"
  --output-root "$OUTPUT_ROOT"
  --project "$PROJECT"
  --group "$GROUP"
  --name "$RUN"
  --max-iter "$MAX_ITER"
  --save-iter "$SAVE_ITER"
  --eval-save-iter "$EVAL_SAVE_ITER"
  --lr "$LR"
  --cfg-dropout-rate "$CFG_DROPOUT_RATE"
  --conditioning-config "$CONDITIONING_CONFIG"
  --scheduler-cycle-lengths "$SCHEDULER_CYCLE_LENGTHS"
  --scheduler-warm-up-steps "$SCHEDULER_WARM_UP_STEPS"
  --tokenizer-chunk-duration "$TOKENIZER_CHUNK_DURATION"
  --num-video-frames "$NUM_VIDEO_FRAMES"
  --frame-selection-mode first
  --temporal-interval-mode force_one
  --num-workers "$NUM_WORKERS"
  --wandb-mode "$WANDB_MODE"
)
if [[ -n "$RESUME_CHECKPOINT_PATH" ]]; then
  cmd+=(--resume-checkpoint-path "$RESUME_CHECKPOINT_PATH")
  if [[ "$RESUME_TRAINING_STATE" == "1" ]]; then
    cmd+=(--resume-training-state)
  fi
  if [[ "$NO_STRICT_RESUME" == "1" ]]; then
    cmd+=(--no-strict-resume)
  fi
fi
if [[ "$DRYRUN" == "1" ]]; then
  cmd+=(--dryrun)
fi
if [[ -n "$EXTRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  extra_override_args=( $EXTRA_OVERRIDES )
  cmd+=(-- "${extra_override_args[@]}")
fi

cd "$AICITY_ROOT"
if [[ -n "$ACTIVATE_COSMOS3" ]]; then
  require_file "$ACTIVATE_COSMOS3"
  source "$ACTIVATE_COSMOS3"
elif [[ -f "$COSMOS_ROOT/.venv/bin/activate" ]]; then
  source "$COSMOS_ROOT/.venv/bin/activate"
else
  echo "ERROR: activate the Cosmos3 environment or set ACTIVATE_COSMOS3" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH=""

echo $$ > "$PIDFILE"

set -x
"${cmd[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set +x
exit "$status"
