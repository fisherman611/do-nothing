#!/usr/bin/env bash
# BTC:  docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/start_jupyter.sh
# Mở http://localhost:9777 , mật khẩu/token: zac2026 → mở predict_notebook.ipynb → Run All → /result/jupyter_submission.jsonl
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} TOKENIZERS_PARALLELISM=false
exec jupyter lab --port 9777 --ip 0.0.0.0 --IdentityProvider.token='zac2026' --ServerApp.root_dir="$PWD" \
    --allow-root --no-browser
