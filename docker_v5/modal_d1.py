"""D1: chạy thử image Docker (docker/Dockerfile) trên Modal để đo thời gian và bộ nhớ GPU. Modal không có A30; mặc định A10 (24 GB,
băng thông bộ nhớ thấp hơn A30 → đo bảo thủ). Đổi bằng D1_GPU=L40S ...

    PACK=<pack_b> bash docker/ctx.sh                                    # dựng context
    modal volume create zaic-d1-data && modal volume put zaic-d1-data data/train+public_test/public_test /public_test
    modal run docker/modal_d1.py --limit 20                              # image dựng từ Dockerfile trên Modal (lần đầu ~20–40 phút)
Dữ liệu mount ở /data như lúc BTC chạy (<split>/<split>.jsonl, audio/, zaic_tools/, <split>_environment/). Kết quả: agent/runs/d1/*.json.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("D1_GPU", "A10")
app = modal.App("lifemate-d1")
data_vol = modal.Volume.from_name("zaic-d1-data")
image = (modal.Image.from_dockerfile(os.path.join(HERE, "Dockerfile"), context_dir=HERE, add_python="3.12")
         .entrypoint([]))


@app.function(image=image, gpu=GPU, cpu=8, memory=49152, volumes={"/data": data_vol}, timeout=3 * 3600)
def run(limit: int = 20, split: str = "public_test", workers: int = 16, agent_flags: str = "--retry-empty",
        time_budget: int = 30):
    import subprocess
    import time
    env = dict(os.environ, SPLIT=split, DATA="/data", OUT="/out", WORK="/work", WORKERS=str(workers),
               AGENT_FLAGS=agent_flags, TIME_BUDGET=str(time_budget), **({"LIMIT": str(limit)} if limit else {}))
    t0 = time.time()
    p = subprocess.run(["bash", "/app/docker/run.sh"], env=env, capture_output=True, text=True)
    res = {"gpu": GPU, "limit": limit, "rc": p.returncode, "total_s": round(time.time() - t0, 1),
           "stdout_tail": p.stdout[-5000:], "stderr_tail": p.stderr[-3000:]}
    def load(path):
        return open(path, encoding="utf-8").read() if os.path.exists(path) else None
    run_txt = load("/work/run.jsonl")
    if run_txt:
        res["samples"] = [{k: r.get(k) for k in ("id", "latency", "steps", "n_requests", "timed_out", "error", "n_retry", "n_n8",
                                                 "n_i5", "n_invalid")} for r in map(json.loads, run_txt.splitlines())]
        res["run_jsonl"] = run_txt
    res["asr_secs"] = json.loads(load("/work/asr_secs.json") or "{}")
    res["asr"] = json.loads(load("/work/asr.json") or "{}")
    res["gpu_mem_csv"] = load("/work/gpu_mem.csv")
    res["vllm_log_tail"] = (load("/work/vllm.log") or "")[-2500:]
    for name in ("submission.jsonl", "submission_smoke.jsonl"):
        if os.path.exists(f"/out/{name}"):
            res[name] = open(f"/out/{name}", encoding="utf-8").read()
            res["submission_lines"] = len(res[name].splitlines())
    return res


@app.local_entrypoint()
def main(limit: int = 20, split: str = "public_test", agent_flags: str = "--retry-empty", time_budget: int = 30):
    res = run.remote(limit=limit, split=split, agent_flags=agent_flags, time_budget=time_budget)
    os.makedirs(os.path.join(HERE, "..", "agent", "runs", "d1"), exist_ok=True)
    out = os.path.join(HERE, "..", "agent", "runs", "d1", f"d1_{GPU}_{limit or 'all'}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("rc", res["rc"], "| tổng", res["total_s"], "s →", out)
    L = sorted(x["latency"] for x in res.get("samples", []) if isinstance(x.get("latency"), (int, float)))
    if L:
        asr = res["asr_secs"]
        tot = sorted(x["latency"] + asr.get(x["id"], 0) for x in res["samples"] if isinstance(x.get("latency"), (int, float)))
        print(f"FC latency p50 {L[len(L)//2]:.1f}s p90 {L[int(len(L)*.9)]:.1f}s max {L[-1]:.1f}s | ASR+FC p50 {tot[len(tot)//2]:.1f}s "
              f"p90 {tot[int(len(tot)*.9)]:.1f}s max {tot[-1]:.1f}s | timed_out {sum(bool(x.get('timed_out')) for x in res['samples'])}")
    print(res["stdout_tail"][-1500:])
