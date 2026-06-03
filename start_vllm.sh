#!/bin/bash
# Script 2: Start vLLM with LMCacheMPConnector, one request at a time
#
# Disables vLLM GPU KV cache reuse by:
#   --no-enable-prefix-caching   : no radix attention / GPU prefix cache
#   --max-num-seqs 1             : process only 1 request at a time
#
# Usage:
#   bash start_vllm.sh                   # baseline (eager prefetch off)
#   bash start_vllm.sh --eager-prefetch  # enable lmcache.mp.eager_prefetch
#
# Overridable env vars:
#   MODEL            (default: /mnt/workspace/models/Llama-3.1-8B)
#   VLLM_PORT        (default: 8000)
#   LMCACHE_PORT     (default: 6567)
#   GPU              (default: 0)
#   GPU_MEM_UTIL     (default: 0.8)
#   MAX_MODEL_LEN    (default: 16384)
#   VLLM_LOG         (default: /tmp/vllm.log)  log file path

set -euo pipefail

MODEL="${MODEL:-/mnt/workspace/models/Llama-3.1-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_PORT="${LMCACHE_PORT:-6567}"
GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm.log}"

EAGER_PREFETCH=false
for arg in "$@"; do
    case "$arg" in
        --eager-prefetch) EAGER_PREFETCH=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "=== Starting vLLM ==="
echo "  Model:          ${MODEL}"
echo "  Port:           ${VLLM_PORT}"
echo "  LMCache port:   ${LMCACHE_PORT}"
echo "  GPU:            ${GPU}"
echo "  Max seqs:       1 (one request at a time)"
echo "  Prefix cache:   disabled"
echo "  Early prefetch: ${EAGER_PREFETCH}"
echo "  Log file:       ${VLLM_LOG}"
echo ""

KV_CONFIG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":${LMCACHE_PORT},\"lmcache.mp.eager_prefetch\":${EAGER_PREFETCH}}}"

PYTHONHASHSEED=0 \
CUDA_VISIBLE_DEVICES="${GPU}" \
LMCACHE_USE_UPSTREAM_MP=1 \
vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --safetensors-load-strategy prefetch \
    --kv-transfer-config "${KV_CONFIG}" \
    2>&1 | tee "${VLLM_LOG}"
