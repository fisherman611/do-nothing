"""Chạy toàn bộ split trong /data, ghi /result/submission.jsonl (id, tool_calls, state_hash, time).

    python3 /code/predict.py            # mặc định: /data, /result
Biến môi trường tuỳ chọn: DATA_ROOT, RESULT_DIR, LIMIT (chỉ N mẫu đầu, để chạy thử).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pipeline as P  # noqa: E402


def main():
    t_start = time.time()
    P.seed_everything(P.SEED)
    root, split = P.find_split(os.environ.get("DATA_ROOT", "/data"))
    out_dir = Path(os.environ.get("RESULT_DIR", "/result"))
    out_dir.mkdir(parents=True, exist_ok=True)
    items = P.load_items(root, split)
    if os.environ.get("LIMIT"):
        items = items[: int(os.environ["LIMIT"])]
    print(f"split={split}: {len(items)} mẫu", flush=True)

    pipe = P.Pipeline(root, split)
    print(f"load model: {pipe.load_secs * 1000:.0f} ms", flush=True)
    out = out_dir / "submission.jsonl"
    log = out_dir / "predict_log.jsonl"
    times = []
    t_pred = time.time()
    try:
        with open(out, "w", encoding="utf-8") as f, open(log, "w", encoding="utf-8") as fl:
            for k, item in enumerate(items, 1):
                tool_calls, state_hash, ms, r = pipe.run_sample(item)
                times.append(ms)
                f.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash, "time": ms},
                                   ensure_ascii=False) + "\n")
                f.flush()
                fl.write(json.dumps({"id": item["id"], "time": ms, "asr": r.get("asr_final"), "steps": r.get("steps"),
                                     "timed_out": r.get("timed_out"), "error": r.get("error")}, ensure_ascii=False) + "\n")
                if k % 20 == 0 or k == len(items):
                    print(f"  {k}/{len(items)}  trung bình {sum(times) / len(times):.0f} ms/mẫu, max {max(times)} ms", flush=True)
    finally:
        pipe.close()
    n = max(len(times), 1)
    print(f"predict: {len(times)} mẫu, {(time.time() - t_pred) * 1000:.0f} ms, trung bình {sum(times) / n:.0f} ms/mẫu, "
          f"max {max(times or [0])} ms, quá 30s: {sum(t > 30000 for t in times)}", flush=True)
    print(f"output: {out}  (tổng {time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
