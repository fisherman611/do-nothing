#!/usr/bin/env bash
# Chạy bộ test trong /data → /result/submission.jsonl. Container không cần mạng.
set -euo pipefail
cd /code
mkdir -p /result
/opt/app/bin/python /code/predict.py
