#!/bin/bash
# Serve a trained checkpoint (SFT or an RL policy exported to HF format) with vLLM for the benchmark.
#   bash training/serve_vllm.sh <model dir> [port 8000] [tp 4]
# then evaluate it like any API model:
#   COPLAY_BASE_URL=http://127.0.0.1:8000/v1 python coplaybench/run_reactive.py --models coplayer
# Thinking is turned off server-side (--default-chat-template-kwargs), exactly as in training.
set -euo pipefail
MODEL=${1:?usage: serve_vllm.sh <model dir> [port] [tp]}
PORT=${2:-8000}
TP=${3:-4}
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec vllm serve "$MODEL" --served-model-name "${SERVED_MODEL_NAME:-coplayer}" --tensor-parallel-size "$TP" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" --trust-remote-code --host 127.0.0.1 --port "$PORT" \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-num-seqs "${MAX_NUM_SEQS:-256}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-32768}" \
    --default-chat-template-kwargs '{"enable_thinking": false}'
