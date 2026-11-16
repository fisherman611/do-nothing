#!/usr/bin/env bash
# D1: chuẩn bị context docker/ctx (code, bộ lọc app, định dạng prompt). Chạy từ thư mục gốc repo:
#   PACK=<pack_b> [FMT=...] bash docker/ctx.sh
set -euo pipefail
: "${PACK:?đặt PACK=<thư mục pack_b>}"
FMT=${FMT:-agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json}
CTX=docker/ctx
rm -rf "$CTX" && mkdir -p "$CTX"
rsync -a --exclude runs --exclude 'clef/data' --exclude '__pycache__' --exclude '*.pyc' --exclude 'emb_cache' \
      --exclude 'clef/app_filter/zs' --exclude 'clef/app_filter/eval_*.json' agent/ "$CTX/agent/"
mkdir -p "$CTX/docker" && cp docker/asr_stage.py docker/run.sh "$CTX/docker/"
cp -r "$PACK" "$CTX/pack"
python3 -c "import json,sys; m=json.load(open('$FMT',encoding='utf-8')); json.dump(m,open('$CTX/fmt.json','w'),ensure_ascii=False)"
