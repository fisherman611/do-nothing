#!/usr/bin/env bash
# Kéo model đã fine-tune (ghim commit) vào saved_models/, rồi dựng image.  Cần mạng lúc BUILD, không cần lúc CHẠY.
#   bash build.sh [tag]            # mặc định tag: zac2026-lifemate:v1
#   SKIP_DOWNLOAD=1 bash build.sh  # saved_models/ đã có sẵn
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-zac2026-lifemate:v1}
PY=${PY:-python3}

# repo @ commit (xem README.md, mục Model)
ASR_REPO=fisherman611/pw-tmp-e46f;          ASR_REV=85081b30df5a9de0b792aa82a7af7876da74a253
FC_REPO=fisherman611/q-v5-6c1e;             FC_REV=c2db4cff5f4cb7b3e2013c20ecd85a91224ca516
EMB_REPO=fisherman611/vibi-appfilter-v2-b;  EMB_REV=386ba3baf57876a556d06bb5a0eb3a05fa7944ec
TOK_REPO=Qwen/Qwen3.5-4B;                   TOK_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a   # chỉ file tokenizer

if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  mkdir -p saved_models/filter
  "$PY" - <<EOF
from huggingface_hub import snapshot_download as d
d("$ASR_REPO", revision="$ASR_REV", local_dir="saved_models/asr")
d("$FC_REPO", revision="$FC_REV", local_dir="saved_models/fc")
d("$EMB_REPO", revision="$EMB_REV", local_dir="saved_models/filter/model")
d("$TOK_REPO", revision="$TOK_REV", local_dir="saved_models/tokenizer",
  allow_patterns=["tokenizer*", "vocab*", "merges*", "special_tokens_map.json", "chat_template*"])
EOF
  cp config/filter/*.json saved_models/filter/
  rm -rf saved_models/*/.cache saved_models/filter/model/.cache
fi
du -sh saved_models/*
[ -n "${DOWNLOAD_ONLY:-}" ] && exit 0
docker build -t "$TAG" .
echo "xong: $TAG"
