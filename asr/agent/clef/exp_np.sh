#!/bin/bash
# thí nghiệm: bước 2700, transcript e46f bỏ dấu câu. Đợi watcher đánh giá xong bước 2800 rồi tạm dừng watcher.
P=/mnt/hungpv/projects/do-nothing; RUN=$P/asr/agent/clef/runs/fmtA_v4_e1; L=$RUN/exp_np.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; }
until grep -q "RESULT step 2800\|LỖI" $RUN/e3_eval.log; do sleep 15; done
log "watcher xong 2800 → dừng watcher"
W=$(pgrep -f "[w]atch_eval_e3.py" | head -1); [ -n "$W" ] && kill $W
sleep 5; pkill -f '[v]llm serve.*--port 8011'; sleep 15
cd $P/asr/agent/clef
CUDA_VISIBLE_DEVICES=0 ../../../.venv_clef/bin/python merge_local.py --base ../../../models/qwen35_4b_text --adapter runs/fmtA_v4_e1/e3_ckpts/step2700 --out ../../../models/_eval_2700 > $RUN/exp_merge.log 2>&1 || { log "LỖI merge"; exit 1; }
cd $P; CUDA_VISIBLE_DEVICES=0 nohup /mnt/hungpv/envs_eval/vllm031/bin/vllm serve models/_eval_2700 --served-model-name llm --host 127.0.0.1 --port 8011 --max-model-len 16384 --gpu-memory-utilization 0.85 --seed 0 --enable-prefix-caching --api-key 01a5b8fcd2364e79 --max-logprobs 64 --logprobs-mode processed_logprobs > $RUN/exp_vllm.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s -m 5 -H 'Authorization: Bearer 01a5b8fcd2364e79' http://127.0.0.1:8011/v1/models | grep -q llm && break; sleep 10; done
log "server sẵn sàng"
python3 - <<E
import json,re
a=json.load(open("$P/data/asr_pub_eval/out/flat_e46f_all.json"))
def np(s): return re.sub(r"\s+"," ",re.sub(r"[,.?!;:…]"," ",s)).strip()
json.dump({k:np(v) for k,v in a.items()},open("$P/data/asr_pub_eval/out/flat_e46f_all_np.json","w"),ensure_ascii=False)
E
cd $P/asr; export LLM_API_KEY=$(cat ~/.llm_api_key_fc)
/home/hungpv/miniconda3/envs/vllm/bin/python agent/clef_agent.py --data ../data/train+public_test --split public_test --url http://127.0.0.1:8011 --key "$LLM_API_KEY" --out agent/runs/clef/pubconf_e3s2700_np.jsonl --asr ../data/asr_pub_eval/out/flat_e46f_all_np.json --workers 16 --fmt-from agent/clef/data/fmtA_v4/train.jsonl.meta.json --app-filter agent/clef/app_filter/pack_b --retry-empty > agent/runs/clef/pubconf_e3s2700_np.log 2>&1
log "xong chạy bỏ dấu câu → khôi phục watcher"
pkill -f '[v]llm serve.*--port 8011'; sleep 10; rm -rf $P/models/_eval_2700
cd $P; nohup .venv_clef/bin/python -u asr/agent/clef/watch_eval_e3.py > $RUN/e3_watch.out 2>&1 < /dev/null &
log "DONE"
