#!/bin/bash
# FC v4 epoch 3: resume từ final của epoch 2, lịch LR 3 epoch (cosine về 0 ở cuối epoch 3). Chạy SAU KHI epoch 2 xong.
# Chạy từng epoch: EPOCHS=1 (hết epoch 1) → sau khi xem val: EPOCHS=2 … (cùng lệnh, --resume last).
#   EPOCHS=1 nohup bash asr/agent/clef/run_v4_resume.sh > asr/agent/clef/runs/fmtA_v4_e1/resume.log 2>&1 &
set -euo pipefail
cd /mnt/hungpv/projects/do-nothing
EPOCHS=${EPOCHS:-1}
RUN=asr/agent/clef/runs/fmtA_v4_e1
set -a; source "$HOME/.zaic_wandb.env"; set +a
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=/mnt/hungpv/projects/do-nothing/asr TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo $$ > "$RUN/resume.pid"
echo "== RESUME e3 (epochs=3, sched=3) gpu=$CUDA_VISIBLE_DEVICES $(date)"
exec .venv_clef/bin/python -u asr/agent/clef/train.py --model models/qwen35_4b_text \
  --train asr/agent/clef/data/fmtA_v4/train.jsonl --val asr/agent/clef/data/fmtA_v4/val_in.jsonl --out "$RUN" \
  --resume "$RUN/final" --epochs 3 --sched-epochs 3 --save-every 100 --wandb lifemate-clef --wandb-name fmtA_v4_e1
