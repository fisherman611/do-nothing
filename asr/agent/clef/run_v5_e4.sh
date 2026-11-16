#!/bin/bash
# FC v4 epoch 4: khởi tạo từ adapter cuối epoch 3, optimizer + lịch LR mới (đỉnh 2e-5, cosine về 0, 1 epoch).
set -euo pipefail
cd /mnt/hungpv/projects/do-nothing
RUN=asr/agent/clef/runs/fmtA_v5_e4
mkdir -p $RUN
set -a; source "$HOME/.zaic_wandb.env"; set +a
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/hungpv/projects/do-nothing/asr TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "== E4 gpu=1 $(date)"
exec .venv_clef/bin/python -u asr/agent/clef/train_e4.py --model models/qwen35_4b_text \
  --train asr/agent/clef/data/fmtA_v5/train.jsonl --val asr/agent/clef/data/fmtA_v5/val_in.jsonl --out "$RUN" \
  --init-adapter asr/agent/clef/runs/fmtA_v4_e1/final --epochs 1 --lr 2e-5 --seed 1 --save-every 100 \
  --wandb lifemate-clef --wandb-name fmtA_v5_e4
