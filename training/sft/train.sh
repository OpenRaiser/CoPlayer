#!/bin/bash
# LlamaFactory launcher for the shared SFT stage.
#   single node : bash training/sft/train.sh [training/sft/coplayer_sft.yaml] [key=value ...]
#   multi node  : NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank-0 IP> MASTER_PORT=29500 bash training/sft/train.sh ...
# Needs llamafactory-cli on PATH (training/requirements-sft.txt) and the data from build_sft_data.py.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CFG=${1:-$REPO/training/sft/coplayer_sft.yaml}
[ $# -gt 0 ] && shift
COPLAY_DATA=${COPLAY_DATA:-$REPO/data/CoPlayBench}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/saves/coplayer_sft}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false FORCE_TORCHRUN=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
[ -n "${NNODES:-}" ] && export NNODES NODE_RANK MASTER_ADDR MASTER_PORT
mkdir -p "$OUTPUT_DIR"
echo "[$(date +%T)] config=$CFG data=$COPLAY_DATA/training output=$OUTPUT_DIR overrides=$*"
llamafactory-cli train "$CFG" dataset_dir="$COPLAY_DATA/training" output_dir="$OUTPUT_DIR" \
    deepspeed="$REPO/training/sft/ds_z3_config.json" "$@" 2>&1 | tee -a "$OUTPUT_DIR/train_rank${NODE_RANK:-0}.log"
