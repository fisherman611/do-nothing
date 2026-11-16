"""Theo dõi checkpoint `last` của job FC epoch 3 trên myserver; mỗi khi có checkpoint mới (mỗi 100 bước) thì:
copy adapter → gộp trên CPU → dựng vLLM trên GPU 0 → chạy 634 mẫu public với transcript e46f → chấm hash theo nhãn → ghi e5_eval.log.
Sau khi train kết thúc thì đánh giá thêm `final`. Chạy:  nohup .venv_clef/bin/python agent/clef/watch_eval_e3.py &"""
import json, os, shutil, subprocess, sys, time, torch

P = "/mnt/hungpv/projects/do-nothing/"
RUN = P + "asr/agent/clef/runs/fmtA_v5_e5/"
CK = RUN + "e5_ckpts/"
LOG = RUN + "e5_eval.log"
TMP = P + "models/_eval_tmp"
KEY = "01a5b8fcd2364e79"
VLLM = "/mnt/hungpv/envs_eval/vllm031/bin/vllm"
os.makedirs(CK, exist_ok=True)
gold = {}
for l in open(P + "result/claude_submission_refined_v4.jsonl", encoding="utf-8"):
    r = json.loads(l); gold[r["id"]] = r["state_hash"]
conf = set(open(P + "asr/agent/clef/pub_eval_ids.txt").read().split())


def say(s):
    with open(LOG, "a") as f: f.write(f"{time.strftime('%H:%M:%S')} {s}\n")
    print(s, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, **kw)


def training():
    return sh("pgrep -f '[c]lef/train_e5.py' > /dev/null").returncode == 0


def server_up():
    return sh(f"curl -s -m 5 -H 'Authorization: Bearer {KEY}' http://127.0.0.1:8011/v1/models | grep -q llm").returncode == 0


def evaluate(step, adapter):
    say(f"== step {step}: gộp (GPU 0)")
    shutil.rmtree(TMP, ignore_errors=True)
    sh("pkill -f '[v]llm serve.*--port 8011'"); time.sleep(15)
    r = sh(f"cd {P}asr/agent/clef && CUDA_VISIBLE_DEVICES=0 ../../../.venv_clef/bin/python merge_local.py --base ../../../models/qwen35_4b_text --adapter {adapter} --out {TMP} > {RUN}e4_merge_{step}.log 2>&1")
    if r.returncode: say(f"LỖI gộp step {step}"); return False
    sh(f"cd {P} && CUDA_VISIBLE_DEVICES=0 nohup {VLLM} serve models/_eval_tmp --served-model-name llm --host 127.0.0.1 --port 8011 --max-model-len 16384 --gpu-memory-utilization 0.85 --seed 0 --enable-prefix-caching --api-key {KEY} --max-logprobs 64 --logprobs-mode processed_logprobs > {RUN}e4_vllm.log 2>&1 < /dev/null &")
    t0 = time.time()
    while not server_up():
        if time.time() - t0 > 900: say("LỖI vLLM không lên"); return False
        time.sleep(10)
    out = P + f"asr/agent/runs/clef/pubconf_e5_step{step}.jsonl"
    if os.path.exists(out): os.rename(out, out + f".old{int(time.time())}")   # tránh dùng lại kết quả cũ (clef_agent bỏ qua id đã có)
    r = sh(f"""cd {P}asr && export LLM_API_KEY=$(cat ~/.llm_api_key_fc) && /home/hungpv/miniconda3/envs/vllm/bin/python agent/clef_agent.py --data ../data/train+public_test --split public_test --url http://127.0.0.1:8011 --key "$LLM_API_KEY" --out {out} --asr ../data/asr_pub_eval/out/flat_e46f_all.json --workers 16 --fmt-from agent/clef/data/fmtA_v4/train.jsonl.meta.json --app-filter agent/clef/app_filter/pack_b --retry-empty > {out}.log 2>&1""")
    ok = {}
    for l in open(out, encoding="utf-8"):
        x = json.loads(l); ok[x["id"]] = x.get("state_hash") == gold.get(x["id"])
    a = sum(v for k, v in ok.items() if k in conf); b = sum(v for k, v in ok.items() if k not in conf)
    say(f"RESULT step {step}: tổng {a + b}/{len(ok)} | 584 tự tin {a}/{len(conf)} | 50 không chắc {b}/{len(ok) - len(conf)}")
    shutil.rmtree(TMP, ignore_errors=True)
    return True


seen = set()
last_m = None
say("watcher bắt đầu")
while True:
    f = RUN + "last/adapter_model.safetensors"; ts = RUN + "last/train_state.pt"
    if os.path.exists(f) and os.path.exists(ts):
        m = max(os.path.getmtime(f), os.path.getmtime(ts))
        if m != last_m and time.time() - m > 60:
            try:
                step = torch.load(ts, map_location="cpu", weights_only=False)["step"]
            except Exception as e:  # đang ghi dở
                say(f"đọc train_state lỗi ({e!r}), thử lại"); time.sleep(30); continue
            last_m = m
            if step not in seen and step > 0:
                seen.add(step)
                d = CK + f"step{step}"; os.makedirs(d, exist_ok=True)
                for n in ("adapter_config.json", "adapter_model.safetensors"): shutil.copy(RUN + "last/" + n, d)
                evaluate(step, d)
    if not training():
        if 1310 not in seen and os.path.exists(RUN + "final/adapter_model.safetensors") and os.path.getmtime(RUN + "final/adapter_model.safetensors") > time.time() - 6 * 3600 * 4:
            st = torch.load(RUN + "final/train_state.pt", map_location="cpu", weights_only=False)["step"]
            if st == 1310:
                seen.add(1310); d = CK + "step1310"; os.makedirs(d, exist_ok=True)
                for n in ("adapter_config.json", "adapter_model.safetensors"): shutil.copy(RUN + "final/" + n, d)
                evaluate(1310, d)
        say("train đã dừng, watcher kết thúc"); break
    time.sleep(30)
