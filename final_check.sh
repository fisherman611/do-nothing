#!/bin/bash
# kiểm tra bản gộp cuối (models/qwen35_4b_v5_e5s400): vLLM trên GPU 1 → 634 mẫu public (e46f) → so hash
cd /mnt/hungpv/projects/do-nothing
OUT=asr/agent/runs/clef/final_check; mkdir -p $OUT
M=models/qwen35_4b_v5_e5s400; KEY=01a5b8fcd2364e79; PORT=8013
CUDA_VISIBLE_DEVICES=1 nohup /mnt/hungpv/envs_eval/vllm031/bin/vllm serve $M --served-model-name llm --host 127.0.0.1 --port $PORT --max-model-len 16384 --gpu-memory-utilization 0.85 --seed 0 --enable-prefix-caching --api-key $KEY --max-logprobs 64 --logprobs-mode processed_logprobs > $OUT/vllm.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/v1/models | grep -q llm && break; sleep 5; done
echo "$(date -u +%T) vLLM lên" >> $OUT/check.log
rm -f $OUT/run.jsonl
cd asr && export LLM_API_KEY=$(cat ~/.llm_api_key_fc) && /home/hungpv/miniconda3/envs/vllm/bin/python agent/clef_agent.py --data ../data/train+public_test --split public_test --url http://127.0.0.1:$PORT --key "$LLM_API_KEY" --out ../$OUT/run.jsonl --asr ../data/asr_pub_eval/out/flat_e46f_all.json --workers 16 --fmt-from agent/clef/data/fmtA_v4/train.jsonl.meta.json --app-filter agent/clef/app_filter/pack_b --retry-empty > ../$OUT/run.log 2>&1
cd ..
python3 - <<"PYEOF" >> $OUT/check.log
import json
gold = {json.loads(l)["id"]: json.loads(l)["state_hash"] for l in open("result/claude_submission_refined_v4.jsonl")}
conf = set(open("asr/agent/clef/pub_eval_ids.txt").read().split())
ok = {}
for l in open("asr/agent/runs/clef/final_check/run.jsonl"):
    x = json.loads(l); ok[x["id"]] = x.get("state_hash") == gold.get(x["id"])
a = sum(v for k, v in ok.items() if k in conf); b = sum(v for k, v in ok.items() if k not in conf)
print(f"RESULT final merged: tổng {a+b}/{len(ok)} | 584 tự tin {a}/{len(conf)} | 50 không chắc {b}/{len(ok)-len(conf)}")
PYEOF
echo "XONG" >> $OUT/check.log
pkill -f "vllm serve.*--port 8013"
