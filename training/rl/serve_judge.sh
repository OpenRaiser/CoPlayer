#!/bin/bash
# Serve the RL reward judge with vLLM (OpenAI-compatible). The paper used Qwen3.8-Flash-Next on one 8-GPU node; it
# receives the benchmark judge prompts with thinking disabled.
#   VLLM_API_KEY=<key> bash training/rl/serve_judge.sh <judge model dir> [port 8200] [tp 8]
# then on the training side: COPLAY_REWARD_URL=http://<host>:8200/v1 COPLAY_REWARD_API_KEY=<key>
set -euo pipefail
MODEL=${1:?usage: serve_judge.sh <judge model dir> [port] [tp]}
PORT=${2:-8200}
TP=${3:-8}
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
exec vllm serve "$MODEL" --served-model-name "${COPLAY_REWARD_MODEL:-coplay-reward}" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" --dtype bfloat16 --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --default-chat-template-kwargs '{"enable_thinking": false}'
