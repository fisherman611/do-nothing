#!/bin/bash
# B8c: train tiếp e46f trên data gốc v1, BỎ CTC (e46f không lưu ctc_head), LR 3e-5, 2 epoch — công thức dự định cho data mới
cd /mnt/hungpv/projects/do-nothing/asr_v3_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 ZAIC_DATA=/mnt/hungpv/projects/do-nothing/data/asr_v3 ZAIC_OUT=/mnt/hungpv/projects/do-nothing/models/asr_out/b8c
venv/bin/python -u asr_train.py --model-id fisherman611/pw-tmp-e46f --model-revision 85081b30df5a9de0b792aa82a7af7876da74a253   --use-prompt --label-smoothing 0.1 --ctc-weight 0 --bs 4 --accum 8 --lr 3e-5 --epochs 2 --patience 2 --no-resume --tag-suffix _b8c_noctc_lr3e-5
echo EXIT $?
