#!/bin/bash
# chạy 3 ASR trên 584 mẫu tự tin của public test (1748 đoạn), GPU 0. Dùng: bash run_pub_eval.sh <tên> <ZAIC_OUT> <tag>
cd /mnt/hungpv/projects/do-nothing/asr_v4_code
export CUDA_VISIBLE_DEVICES=0 ZAIC_DATA=/mnt/hungpv/projects/do-nothing/data/asr_pub_eval TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
name=$1; export ZAIC_OUT=$2; tag=$3
venv/bin/python -u asr_infer_v2.py "$tag" public ../data/asr_pub_eval/out/$name.json --nbest 0 > logs/pub_eval_$name.log 2>&1
echo EXIT $? >> logs/pub_eval_$name.log
