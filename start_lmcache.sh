#!/bin/bash
# Script 1: Start LMCache server with PYTHONHASHSEED=0 and L2-ONLY disk backend
#
# Usage:
#   bash scripts/start_lmcache.sh
#

set -euo pipefail

LMCACHE_PORT="${LMCACHE_PORT:-6567}"
DISK_PATH="${DISK_PATH:-/mnt/workspace/lmcache_kvcache}"
L1_SIZE_GB="${L1_SIZE_GB:-20}"
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/lmcache.log}"

mkdir -p "${DISK_PATH}"

echo "=== Starting LMCache server ==="
echo "  Port:      ${LMCACHE_PORT}"
echo "  Disk path: ${DISK_PATH}"
echo "  L1 size:   ${L1_SIZE_GB} GB"
echo "  Hash seed: PYTHONHASHSEED=0"
echo "  Log file:  ${LMCACHE_LOG}"
echo ""

PYTHONHASHSEED=0 \
lmcache server \
    --port "${LMCACHE_PORT}" \
    --l1-size-gb "${L1_SIZE_GB}" \
    --eviction-policy LRU \
    --chunk-size 256 \
    --l2-store-policy skip_l1 \
    --l2-adapter "{\"type\":\"fs\",\"base_path\":\"${DISK_PATH}\",\"use_odirect\":false}" \
    2>&1 | tee "${LMCACHE_LOG}"
