#!/usr/bin/env bash
# Xuất image để nộp (mục 5 bước 11 và mục 8 của hướng dẫn BTC):  bash docker/save.sh <tag> <TeamName>
# Ra zac2026_<TeamName>.tar.gz + MD5. Upload lên Google Drive, chia sẻ "Anyone with the link", nộp link + checksum.
set -euo pipefail
TAG=${1:?tag image}; TEAM=${2:?tên đội}
OUT="zac2026_${TEAM}.tar.gz"
docker save "$TAG" | gzip > "$OUT"
md5sum "$OUT" | tee "$OUT.md5"
echo "Kiểm trên máy khác: docker load -i $OUT  rồi chạy lại /code/predict.sh"
