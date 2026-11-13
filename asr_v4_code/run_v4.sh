#!/bin/bash
# ASR v4 (13/11, chủ repo cho phép GPU 0): lọc vòng ngược TTS P1 → GĐ1 full (thật + TTS P1), tối đa 20 epoch, early stop 2
# → GĐ2 chỉ data thật, 3 epoch (early stop 2), khởi tạo từ adapter tốt nhất GĐ1. Nhãn không dấu câu. Log wandb.
set -euo pipefail
cd /mnt/hungpv/projects/do-nothing/asr_v4_code
echo $$ > logs/run_v4.pid
set -a; source "$HOME/.zaic_wandb.env"; set +a          # WANDB_API_KEY (chmod 600, ngoài repo)
export WANDB_PROJECT=zaic-lifemate-asr WANDB_LOG_MODEL=false
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export ZAIC_DATA=/mnt/hungpv/projects/do-nothing/data/asr_v3 ZAIC_OUT=/mnt/hungpv/projects/do-nothing/models/asr_out/v4
mkdir -p "$ZAIC_OUT"
M="--model-id fisherman611/pw-tmp-e46f --model-revision 85081b30df5a9de0b792aa82a7af7876da74a253"
COMMON="$M --use-prompt --label-smoothing 0.1 --ctc-weight 0 --bs 4 --accum 8 --patience 2 --strip-punct --aug-v3 0.3 --lr-sched constant_with_warmup --report-to wandb"

# 13/11: BỎ lọc vòng ngược (chủ repo). Lần chạy thử lọc chỉ bỏ 12/1 499 và đa số là e46f nghe sai tên/mã (SCTV, ZKS, SJC),
# đúng loại câu khó cần dạy → dùng nguyên tts_manifest_pub.json.
echo "== GĐ1: full (thật + TTS P1), ≤20 epoch, early stop 2 $(date)"
WANDB_NAME=v4_p1_full venv/bin/python -u asr_train.py $COMMON --lr 3e-5 --epochs 20 --warmup-steps 60 \
    --tts-manifest tts_manifest_pub.json --tag-suffix v4p1
P1=$(ls -d "$ZAIC_OUT"/*v4p1_adapter | head -1)
echo "adapter GĐ1: $P1"

echo "== GĐ2: chỉ data thật, 3 epoch, early stop 2 $(date)"
WANDB_NAME=v4_p2_real venv/bin/python -u asr_train.py $COMMON --lr 1e-5 --epochs 3 --warmup-steps 30 --lr-sched cosine \
    --init-adapter "$P1" --no-resume --tag-suffix v4p2_real
echo "== XONG $(date)"
