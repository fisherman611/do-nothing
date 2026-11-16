#!/usr/bin/env bash
# D1: chuẩn bị context rồi dựng image.  Chạy từ thư mục gốc repo (zaic/):
#   PACK=<thư mục bộ lọc app (pack_b)> FMT=agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json bash docker/build.sh [tag]
# Tuỳ chọn: ASR_REPO ASR_REV FC_REPO FC_REV (mặc định: ASR v1 large e46f + FC fmtA_48e). Cần mạng lúc build.
set -euo pipefail
TAG=${1:-lifemate:dev}
: "${PACK:?đặt PACK=<thư mục pack_b: catalog.json keywords.json model/ pack.json>}"
FMT=${FMT:-agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json}
bash docker/ctx.sh
ARGS=()
for v in ASR_REPO ASR_REV FC_REPO FC_REV; do [ -n "${!v:-}" ] && ARGS+=(--build-arg "$v=${!v}"); done
docker build -f docker/Dockerfile -t "$TAG" "${ARGS[@]}" docker
echo "xong: $TAG"
