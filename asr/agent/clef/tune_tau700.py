"""Tune --ask-tau trên checkpoint e4 step 700: merge (GPU 1) → vLLM cổng 8012 (GPU 1, util 0.42) → chạy 634 với từng τ. Chỉ đọc log/ghi file mới."""
import json, os, shutil, subprocess, time
P = "/mnt/hungpv/projects/do-nothing/"
RUN = P + "asr/agent/clef/runs/fmtA_v5_e4/"
OUT = P + "asr/agent/runs/clef/tau700/"; os.makedirs(OUT, exist_ok=True)
TMP = P + "models/_tune700"; LOG = OUT + "tune.log"
KEY = "01a5b8fcd2364e79"; VLLM = "/mnt/hungpv/envs_eval/vllm031/bin/vllm"; PORT = 8012
gold = {}
for l in open(P + "result/claude_submission_refined_v4.jsonl", encoding="utf-8"):
    r = json.loads(l); gold[r["id"]] = r["state_hash"]
conf = set(open(P + "asr/agent/clef/pub_eval_ids.txt").read().split())
def say(s):
    t = time.strftime("%H:%M:%S")
    open(LOG, "a").write(t + " " + s + "\n")
def sh(c): return subprocess.run(c, shell=True)
def up():
    c = "curl -s -m 5 -H \"Authorization: Bearer " + KEY + "\" http://127.0.0.1:" + str(PORT) + "/v1/models | grep -q llm"
    return sh(c).returncode == 0
need = not os.path.exists(TMP + "/config.json")
if need: shutil.rmtree(TMP, ignore_errors=True)
say("gộp step700 (GPU 1)" if need else "dùng lại bản đã gộp")
r = sh("true") if not need else sh(f"cd {P}asr/agent/clef && CUDA_VISIBLE_DEVICES=1 ../../../.venv_clef/bin/python merge_local.py --base ../../../models/qwen35_4b_text --adapter {RUN}e4_ckpts/step700 --out {TMP} > {OUT}merge.log 2>&1")
if r.returncode: say("LỖI gộp"); raise SystemExit(1)
sh(f"cd {P} && CUDA_VISIBLE_DEVICES=1 nohup {VLLM} serve {TMP} --served-model-name llm --host 127.0.0.1 --port {PORT} --max-model-len 16384 --gpu-memory-utilization 0.45 --max-num-seqs 48 --seed 0 --enable-prefix-caching --api-key {KEY} --max-logprobs 64 --logprobs-mode processed_logprobs > {OUT}vllm.log 2>&1 < /dev/null &")
t0 = time.time()
while not up():
    if time.time() - t0 > 900: say("LỖI vLLM không lên"); raise SystemExit(1)
    time.sleep(10)
say("vLLM lên")
for tau in ["0.2", "0.35", "0.5", "0.1"]:
    out = OUT + f"pubconf_tau{tau}.jsonl"
    if os.path.exists(out): os.rename(out, out + f".old{int(time.time())}")
    sh(f"cd {P}asr && export LLM_API_KEY=$(cat ~/.llm_api_key_fc) && /home/hungpv/miniconda3/envs/vllm/bin/python agent/clef_agent.py --data ../data/train+public_test --split public_test --url http://127.0.0.1:{PORT} --key \"$LLM_API_KEY\" --out {out} --asr ../data/asr_pub_eval/out/flat_e46f_all.json --workers 12 --fmt-from agent/clef/data/fmtA_v4/train.jsonl.meta.json --app-filter agent/clef/app_filter/pack_b --retry-empty --ask-tau {tau} > {out}.log 2>&1")
    ok = {}
    for l in open(out, encoding="utf-8"):
        x = json.loads(l); ok[x["id"]] = x.get("state_hash") == gold.get(x["id"])
    a = sum(v for k, v in ok.items() if k in conf); b = sum(v for k, v in ok.items() if k not in conf)
    say(f"RESULT tau={tau}: tổng {a + b}/{len(ok)} | 584 tự tin {a}/{len(conf)} | 50 không chắc {b}/{len(ok) - len(conf)}")
sh("pkill -f \"[v]llm serve.*--port 8012\"")
shutil.rmtree(TMP, ignore_errors=True)
say("XONG")
