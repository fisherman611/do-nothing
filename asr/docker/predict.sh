#!/usr/bin/env bash
# Lối chạy của BTC:  docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/predict.sh
# Đọc /data/<split>/, ghi /result/submission.jsonl (mỗi dòng: id, tool_calls, state_hash, time). Không cần mạng.
# Tham số mặc định nằm trong /code/predict.py (CFG); thời gian nạp model và thời gian predict được in ra stdout.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
exec python3 predict.py "$@"
