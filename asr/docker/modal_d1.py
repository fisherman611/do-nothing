"""D1: chạy thử image Docker (docker/Dockerfile) trên Modal bằng ĐÚNG lối chạy của BTC (/code/predict.sh: lần lượt từng mẫu,
ASR và vLLM cùng nằm trên GPU) để đo thời gian từng mẫu và bộ nhớ GPU. Modal không có A30; mặc định A10 (24 GB,
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
def run(limit: int = 20, split: str = "public_test", time_budget: int = 30, gpu_util: str = "0.70"):
    import subprocess
    import time
    env = dict(os.environ, SPLIT=split, DATA="/data", OUT="/out/submission.jsonl", WORK="/work", GPU_LOG="1",
               GPU_UTIL=gpu_util, TIME_BUDGET=str(time_budget), **({"LIMIT": str(limit)} if limit else {}))
    t0 = time.time()
    p = subprocess.run(["bash", "/code/predict.sh"], env=env, capture_output=True, text=True)
    res = {"gpu": GPU, "limit": limit, "rc": p.returncode, "total_s": round(time.time() - t0, 1),
           "stdout_tail": p.stdout[-5000:], "stderr_tail": p.stderr[-3000:]}
    def load(path):
        return open(path, encoding="utf-8").read() if os.path.exists(path) else None
    run_txt = load("/work/submission.run.jsonl")          # log chi tiết của predict.py: time (ms, cả ASR), asr_s, latency (FC) ...
    if run_txt:
        res["samples"] = [{k: r.get(k) for k in ("id", "time", "asr_s", "latency", "steps", "n_requests", "timed_out", "error",
                                                 "n_retry", "n_invalid", "asr_errors")} for r in map(json.loads, run_txt.splitlines())]
        res["run_jsonl"] = run_txt
    res["asr_log_tail"] = (load("/work/asr.log") or "")[-1500:]
    res["gpu_mem_csv"] = load("/work/gpu_mem.csv")
    res["vllm_log_tail"] = (load("/work/vllm.log") or "")[-2500:]
    if os.path.exists("/out/submission.jsonl"):
        res["submission_lines"] = len(open("/out/submission.jsonl", encoding="utf-8").read().splitlines())
    return res


@app.local_entrypoint()
def main(limit: int = 20, split: str = "public_test", time_budget: int = 30, gpu_util: str = "0.70"):
    res = run.remote(limit=limit, split=split, time_budget=time_budget, gpu_util=gpu_util)
    os.makedirs(os.path.join(HERE, "..", "agent", "runs", "d1"), exist_ok=True)
    out = os.path.join(HERE, "..", "agent", "runs", "d1", f"d1_{GPU}_{limit or 'all'}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("rc", res["rc"], "| tổng", res["total_s"], "s →", out)
    T = sorted(x["time"] / 1000 for x in res.get("samples", []) if isinstance(x.get("time"), (int, float)))
    if T:
        A = sorted(x.get("asr_s") or 0 for x in res["samples"])
        print(f"mỗi mẫu (ASR+FC, một luồng) p50 {T[len(T)//2]:.1f}s p90 {T[int(len(T)*.9)]:.1f}s max {T[-1]:.1f}s | ASR p50 {A[len(A)//2]:.1f}s "
              f"max {A[-1]:.1f}s | timed_out {sum(bool(x.get('timed_out')) for x in res['samples'])} | "
              f"lỗi {sum(bool(x.get('error')) for x in res['samples'])}")
    print(res["stdout_tail"][-1500:])
