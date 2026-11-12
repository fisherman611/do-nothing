#!/usr/bin/env bash
# Chuẩn bị context rồi dựng image nộp bài.  Chạy từ thư mục asr/ của repo:
#   PACK=<thư mục bộ lọc app (pack_b)> FMT=agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json bash docker/build.sh [tag]
# Tuỳ chọn: ASR_REPO ASR_REV FC_REPO FC_REV (mặc định: ASR v1 large e46f + FC fmtA_48e). Cần mạng lúc build.
# Sau khi chạy thử xong: bash docker/save.sh <tag> <TeamName>  → zac2026_<TeamName>.tar.gz + MD5 để nộp.
set -euo pipefail
TAG=${1:-zac2026-lifemate:v1}
: "${PACK:?đặt PACK=<thư mục pack_b: catalog.json keywords.json model/ pack.json>}"
FMT=${FMT:-agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json}
PACK="$PACK" FMT="$FMT" bash docker/ctx.sh
ARGS=()
for v in ASR_REPO ASR_REV FC_REPO FC_REV; do [ -n "${!v:-}" ] && ARGS+=(--build-arg "$v=${!v}"); done
docker build -f docker/Dockerfile -t "$TAG" "${ARGS[@]}" docker
cat <<MSG
xong: $TAG
Chạy thử như BTC (tắt mạng):
  docker run --rm --network none --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result $TAG /bin/bash /code/predict.sh
  docker run --rm -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result $TAG /bin/bash /code/start_jupyter.sh
MSG
