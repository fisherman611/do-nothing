#!/bin/bash
# FC v4: resume epoch 1 trên myserver GPU 1 từ checkpoint Modal (bước 100/1319), lịch LR tính cho 5 epoch.
# Chạy từng epoch: EPOCHS=1 (hết epoch 1) → sau khi xem val: EPOCHS=2 … (cùng lệnh, --resume last).
#   EPOCHS=1 nohup bash asr/agent/clef/run_v4_resume.sh > asr/agent/clef/runs/fmtA_v4_e1/resume.log 2>&1 &
set -euo pipefail
cd /mnt/hungpv/projects/do-nothing
EPOCHS=${EPOCHS:-1}
RUN=asr/agent/clef/runs/fmtA_v4_e1
set -a; source "$HOME/.zaic_wandb.env"; set +a
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=/mnt/hungpv/projects/do-nothing/asr TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo $$ > "$RUN/resume.pid"
echo "== RESUME epochs=$EPOCHS gpu=$CUDA_VISIBLE_DEVICES $(date)"
exec .venv_clef/bin/python -u asr/agent/clef/train.py --model models/qwen35_4b_text \
  --train asr/agent/clef/data/fmtA_v4/train.jsonl --val asr/agent/clef/data/fmtA_v4/val_in.jsonl --out "$RUN" \
  --resume "$RUN/last" --epochs "$EPOCHS" --sched-epochs 5 --save-every 100 --wandb lifemate-clef --wandb-name fmtA_v4_e1
