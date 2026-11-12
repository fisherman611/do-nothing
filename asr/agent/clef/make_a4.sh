#!/usr/bin/env bash
# A4 (PLAN_72h_v3) — dựng data FC vòng 2 = đúng lệnh của fmtA_v3 (A1, §1o), chỉ đổi NGUỒN NHIỄU sang e46f:
#   (1) bản nghe thật: câu thật giữ lỗi PhoWhisper-medium gốc (e46f đã train trên chính audio đó → nghe "quá đúng");
#       câu TTS dùng e46f nghe bản đã xử lý augment_v3 (agent/clef/nbest_to_noise.py → e46f_tts.asr.json)
#   (4) bảng cụm: asr_subst_v2 ∪ lỗi e46f trên ~9 800 câu TTS, kể cả phương ngữ ElevenLabs ("bả nghen" → "ba ngàn")
# Bộ lọc noise_reject GIỮ NGUYÊN (không đổi số/giờ/phủ định/điều kiện): bản 1-best không có cách nghe khác thì nhãn sạch
# với số sai là không học được. Nới phần số chỉ khi có A4 (5) "cách nghe khác" (chưa làm, cần đổi cả ASR lúc infer).
# Cần trước: agent/runs/asr/{e46f_tts.asr.json, e46f_tts.manifest.json, asr_subst_e46f_tts.json} (nbest_to_noise.py)
#     bash agent/clef/make_a4.sh [pack_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/thanhlh13/.works/miniconda3/envs/myenv/bin/python}
PACK=${1:-/tmp/claude-1001/-home-thanhlh13--works-research/390e52ec-5fa4-4959-a5d9-1959c372d465/scratchpad/pack_b}
R=agent/runs/asr
for f in $R/e46f_tts.asr.json $R/e46f_tts.manifest.json $R/asr_subst_e46f_tts.json $PACK/pack.json; do [ -f "$f" ] || { echo "thiếu $f"; exit 1; }; done

$PY - <<'EOF'
import json
R = "agent/runs/asr"
b0 = json.load(open(f"{R}/b0.base_medium.asr.json"))["hyps"]
e46 = json.load(open(f"{R}/e46f_tts.asr.json"))["hyps"]
hyps = {k: v for k, v in b0.items() if k.startswith("train_")}          # câu thật: PhoWhisper-medium
hyps.update(e46)                                                         # câu TTS: e46f
json.dump({"model": "b0 medium (thật) + e46f (TTS augment_v3)", "hyps": hyps}, open(f"{R}/b0_e46f.asr.json", "w"), ensure_ascii=False)
v2 = json.load(open(f"{R}/asr_subst_v2.json")); ne = json.load(open(f"{R}/asr_subst_e46f_tts.json"))
tab = {k: list(v) for k, v in v2["table"].items()}
for k, v in ne["table"].items():
    have = {x[0] for x in tab.get(k, [])}
    tab.setdefault(k, []).extend([x for x in v if x[0] not in have])
json.dump({"stats": {"v2": v2["stats"], "e46f_tts": ne["stats"], "merged_entries": len(tab)}, "table": tab},
          open(f"{R}/asr_subst_a4.json", "w"), ensure_ascii=False)
print("hyps:", len(hyps), "(thật", sum(k.startswith("train_") for k in hyps), "+ TTS", len(e46), ") | bảng cụm:", len(tab))
EOF

OUT=agent/clef/data/fmtA_a4_full
mkdir -p $OUT; rm -f $OUT/train.s*.jsonl*
for i in 0 1 2 3 4 5; do
  $PY agent/clef/build_data.py --data data/train+public_test --split-name train,val_ood --out $OUT/train.s$i.jsonl \
    --history --phonetic synth_phonetic/train_phonetic.jsonl --synth synth/train_alias.jsonl --synth synth/train_iface.jsonl \
    --style v1 --q2 all --q1-desc short --sys compact --rules short --app-filter $PACK --decoy-apps 2 --full-catalog-p 0.4 \
    --asr-errors $R/asr_errors_v2.jsonl --asr-subst $R/asr_subst_a4.json --asr-p 0.6 --asr-hist-p 0.15 \
    --asr-b0 $R/b0_e46f.asr.json --asr-b0-manifest data/tts_v2/manifest_v2.json --asr-b0-manifest $R/e46f_tts.manifest.json \
    --asr-q 0.45 --asr-table-mix 0.3 --shard $i/6 > $OUT/build.s$i.log 2>&1 < /dev/null &
done
wait
grep -l -E "Traceback|Error" $OUT/build.s*.log && { echo "lỗi build, xem $OUT/build.s*.log"; exit 1; } || true
$PY agent/clef/merge_shards.py $OUT
$PY agent/clef/make_v3.py --src $OUT --newapps agent/clef/data/newapps_q1/train.jsonl --out agent/clef/data/fmtA_a4 --target 11500
echo "xong: agent/clef/data/fmtA_a4 — chạy tiếp audit_v3.py / audit_gold.py trước khi train"
