#!/bin/bash
# đợi epoch 4 (v5) xong (train_e4.py thoát, final bước 1310) rồi chạy epoch 5 + watcher đánh giá
cd /mnt/hungpv/projects/do-nothing
L=asr/agent/clef/runs/chain_v5e5.log
while pgrep -f "[c]lef/train_e4.py" > /dev/null; do sleep 30; done
st=$(.venv_clef/bin/python -c "import torch;print(torch.load(\"asr/agent/clef/runs/fmtA_v5_e4/final/train_state.pt\",map_location=\"cpu\",weights_only=False)[\"step\"])")
echo "$(date -u) epoch 4 thoát, final step=$st" >> $L
if [ "$st" != "1310" ]; then echo "final không phải bước 1310 → KHÔNG chạy epoch 5" >> $L; exit 1; fi
mkdir -p asr/agent/clef/runs/fmtA_v5_e5
nohup bash asr/agent/clef/run_v5_e5.sh > asr/agent/clef/runs/fmtA_v5_e5/train.log 2>&1 < /dev/null &
echo "$(date -u) đã chạy epoch 5" >> $L
while pgrep -f "[w]atch_eval_v5.py" > /dev/null; do sleep 30; done
nohup .venv_clef/bin/python -u asr/agent/clef/watch_eval_v5e5.py > asr/agent/clef/runs/fmtA_v5_e5/watch.out 2>&1 < /dev/null &
echo "$(date -u) đã bật watcher e5" >> $L
