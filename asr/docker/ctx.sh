#!/usr/bin/env bash
# Chuẩn bị context docker/ctx theo cấu trúc /code của BTC. Chạy từ thư mục asr/ của repo:
#   PACK=<pack_b> [FMT=...] bash docker/ctx.sh
# Ra: docker/ctx/code/ (→ /code), docker/ctx/pack (→ /code/saved_models/pack), docker/ctx/fmt.json
set -euo pipefail
: "${PACK:?đặt PACK=<thư mục pack_b: catalog.json keywords.json model/ pack.json>}"
FMT=${FMT:-agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json}
for f in "$PACK/pack.json" "$PACK/keywords.json" "$FMT"; do [ -e "$f" ] || { echo "không thấy $f" >&2; exit 2; }; done
CTX=docker/ctx
EXC=(--exclude runs --exclude 'clef/data' --exclude '__pycache__' --exclude '*.pyc' --exclude 'emb_cache'
     --exclude 'clef/app_filter/zs' --exclude 'clef/app_filter/eval_*.json')
rm -rf "$CTX" && mkdir -p "$CTX/code/docker" "$CTX/code/training_code"

# 1) mã suy luận
rsync -a "${EXC[@]}" agent/ "$CTX/code/agent/"
cp docker/asr_stage.py docker/run.sh "$CTX/code/docker/"
cp docker/predict.py docker/predict.sh docker/start_jupyter.sh docker/predict_notebook.ipynb docker/selfcheck.py "$CTX/code/"
cp docker/code_README.md "$CTX/code/README.md"
cp docker/code_requirements.txt "$CTX/code/requirements.txt"
chmod +x "$CTX/code/predict.sh" "$CTX/code/start_jupyter.sh"

# 2) training_code/: toàn bộ mã train + README (BTC chạy lại training, có mạng; data và model lấy theo URL trong README).
#    Không kèm các công cụ gán nhãn / duyệt transcript public_test (chỉ dùng để đánh giá, không phải mã train).
rsync -a "${EXC[@]}" --exclude 'docker/ctx' --exclude 'data' --exclude '.venv' --exclude '*.wav' --exclude '*.safetensors' \
      --exclude '*.bin' --exclude '*.pt' --max-size=20m \
      --exclude 'qwen_cloud_label.py' --exclude 'label_remaining.py' --exclude 'review_ui.py' --exclude 'build_review_drafts.py' \
      --exclude 'QWEN_CLOUD.md' --exclude 'tests/test_qwen_cloud_label.py' --exclude 'tests/test_label_remaining.py' \
      ./ "$CTX/code/training_code/asr/"
for d in ../metrics ../plans ../synth; do [ -d "$d" ] && rsync -a --exclude '__pycache__' --max-size=20m "$d" "$CTX/code/training_code/"; done
cp ../requirements*.txt "$CTX/code/training_code/" 2>/dev/null || true
cp docker/training_README.md "$CTX/code/training_code/README.md"

# 3) bộ lọc app + định dạng prompt
cp -r "$PACK" "$CTX/pack"
python3 -c "import json,sys; m=json.load(open('$FMT',encoding='utf-8')); json.dump(m,open('$CTX/fmt.json','w'),ensure_ascii=False)"
echo "ctx xong: $(du -sh "$CTX" | cut -f1) | code $(find "$CTX/code" -type f | wc -l) file"
