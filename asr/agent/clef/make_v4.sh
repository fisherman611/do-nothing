#!/bin/bash
# FC v4 (13/11, PLAN_public_text_augment §3): BTC (tập con ~11 500 như v3, nhãn bỏ dấu câu 80%) + biến thể public (data/fc_pub)
# + newapps_q1. Nhiễu ASR như v3 (B0 PhoWhisper-medium gốc + bảng cụm v2). Catalog đủ ở 25% ví dụ (v3: 40%) cho train nhanh hơn.
#   bash agent/clef/make_v4.sh           (từ thư mục zaic/; ~CPU, chạy nền)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/thanhlh13/.works/miniconda3/envs/myenv/bin/python}
PACK=agent/clef/app_filter/pack_b
R=agent/runs/asr
FLAGS="--history --style v1 --q2 all --q1-desc short --sys compact --rules short --app-filter $PACK --decoy-apps 2 --full-catalog-p 0.25 \
  --asr-errors $R/asr_errors_v2.jsonl --asr-subst $R/asr_subst_v2.json --asr-p 0.6 --asr-hist-p 0.15 \
  --asr-b0 $R/b0.base_medium.asr.json --asr-b0-manifest data/tts_v2/manifest_v2.json --asr-q 0.45 --asr-table-mix 0.3 \
  --strip-punct-p 0.8"

# 1) BTC đầy đủ (train + val_ood, kèm phonetic/alias/iface) → tập con 11 500 + newapps_q1
OUT=agent/clef/data/fmtA_v4_full
mkdir -p $OUT; rm -f $OUT/train.s*.jsonl*
for i in 0 1 2 3 4 5; do
  $PY agent/clef/build_data.py --data data/train+public_test --split-name train,val_ood --out $OUT/train.s$i.jsonl \
    --phonetic synth_phonetic/train_phonetic.jsonl --synth synth/train_alias.jsonl --synth synth/train_iface.jsonl \
    $FLAGS --shard $i/6 > $OUT/build.s$i.log 2>&1 < /dev/null &
done
wait
grep -l -E "Traceback|Error" $OUT/build.s*.log && { echo "lỗi build BTC, xem $OUT/build.s*.log"; exit 1; } || true
$PY agent/clef/merge_shards.py $OUT
$PY agent/clef/make_v3.py --src $OUT --newapps agent/clef/data/newapps_q1/train.jsonl --out agent/clef/data/fmtA_v4_btc --target 11500

# 2) biến thể public (không có mẫu train: --limit 0), cùng cờ
PUB=agent/clef/data/fmtA_v4_pub
mkdir -p $PUB; rm -f $PUB/train.s*.jsonl*
for i in 0 1 2 3 4 5; do
  $PY agent/clef/build_data.py --data data/train+public_test --split-name train,val_ood --out $PUB/train.s$i.jsonl --limit 0 \
    --extra data/fc_pub/fc_pub_items.jsonl $FLAGS --shard $i/6 > $PUB/build.s$i.log 2>&1 < /dev/null &
done
wait
grep -l -E "Traceback|Error" $PUB/build.s*.log && { echo "lỗi build public, xem $PUB/build.s*.log"; exit 1; } || true
$PY agent/clef/merge_shards.py $PUB

# 3) gộp + xáo → fmtA_v4
$PY - <<'EOF'
import json, random
A = [l for l in open("agent/clef/data/fmtA_v4_btc/train.jsonl", encoding="utf-8")]
B = [l for l in open("agent/clef/data/fmtA_v4_pub/train.jsonl", encoding="utf-8")]
random.Random(4).shuffle(L := A + B)
import os; os.makedirs("agent/clef/data/fmtA_v4", exist_ok=True)
open("agent/clef/data/fmtA_v4/train.jsonl", "w", encoding="utf-8").writelines(L)
m = json.load(open("agent/clef/data/fmtA_v4_btc/train.jsonl.meta.json"))
m.update(n_examples=len(L), n_btc_newapps=len(A), n_pub=len(B), pub_items="data/fc_pub/fc_pub_items.jsonl", version="fmtA_v4")
json.dump(m, open("agent/clef/data/fmtA_v4/train.jsonl.meta.json", "w"), ensure_ascii=False, indent=1)
print(f"fmtA_v4: {len(L)} ví dụ = BTC+newapps {len(A)} + public {len(B)}")
EOF
echo "== XONG make_v4"
