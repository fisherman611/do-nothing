#!/bin/bash
# đợi epoch 3 xong (train.py thoát, final ở bước 3955) rồi chạy epoch 4 (v5) + watcher đánh giá
cd /mnt/hungpv/projects/do-nothing
while pgrep -f "[c]lef/train.py" > /dev/null; do sleep 30; done
st=$(.venv_clef/bin/python -c "import torch;print(torch.load(\"asr/agent/clef/runs/fmtA_v4_e1/final/train_state.pt\",map_location=\"cpu\",weights_only=False)[\"step\"])")
echo "$(date -u) epoch 3 thoát, final step=$st" >> asr/agent/clef/runs/chain_v5.log
if [ "$st" != "3955" ]; then echo "final không phải bước 3955 → KHÔNG chạy epoch 4 (v5)" >> asr/agent/clef/runs/chain_v5.log; exit 1; fi
mkdir -p asr/agent/clef/runs/fmtA_v5_e4
nohup bash asr/agent/clef/run_v5_e4.sh > asr/agent/clef/runs/fmtA_v5_e4/train.log 2>&1 < /dev/null &
echo "$(date -u) đã chạy epoch 4 (v5)" >> asr/agent/clef/runs/chain_v5.log
while pgrep -f "[w]atch_eval_e3.py" > /dev/null; do sleep 30; done
nohup .venv_clef/bin/python -u asr/agent/clef/watch_eval_v5.py > asr/agent/clef/runs/fmtA_v5_e4/watch.out 2>&1 < /dev/null &
echo "$(date -u) đã bật watcher v5" >> asr/agent/clef/runs/chain_v5.log
