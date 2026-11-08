#!/usr/bin/env bash
# Chạy agent zero-shot cho nhiều model song song (mỗi model một server vLLM trên Modal).
#   LLM_API_KEY=... ./run_zeroshot.sh <data_dir> <urls_file> <out_dir> [extra args, vd. --limit 10]
# urls_file: mỗi dòng "<model_key> <url>" (lấy bằng modal.Function.from_name(...).get_web_url())
set -u
DATA=$1; URLS=$2; OUT=$3; shift 3
PY=${PY:-python3}
mkdir -p "$OUT"
while read -r key url; do
  [ -z "$key" ] && continue
  "$PY" "$(dirname "$0")/agent.py" --data "$DATA" --split train --model "$key" --url "$url" \
      --out "$OUT/$key.jsonl" "$@" > "$OUT/$key.log" 2>&1 &
done < "$URLS"
wait
