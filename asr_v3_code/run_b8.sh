#!/bin/bash
# B8: train tiếp e46f trên data gốc v1 (4 631 câu thật, không TTS, không augment) — 5 epoch, cùng công thức B4b trừ data
cd /mnt/hungpv/projects/do-nothing/asr_v3_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 ZAIC_DATA=/mnt/hungpv/projects/do-nothing/data/asr_v3 ZAIC_OUT=/mnt/hungpv/projects/do-nothing/models/asr_out/b8
venv/bin/python -u asr_train.py --model-id fisherman611/pw-tmp-e46f --model-revision 85081b30df5a9de0b792aa82a7af7876da74a253   --use-prompt --label-smoothing 0.1 --ctc-weight 0.3 --bs 4 --accum 8 --lr 1e-4 --epochs 5 --patience 5 --no-resume --tag-suffix _b8_v1data
echo EXIT $?
