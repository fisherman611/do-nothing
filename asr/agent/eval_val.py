"""M1/M3 (PLAN_v3_improve §0): chạy clef_agent trên một split val rồi chấm BTC (replay), một lệnh.

    python agent/eval_val.py --name fmtA_48e_notau --split val_in --url <vLLM> \
        --fmt-from agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json --app-filter <pack_b> \
        [--ref agent/runs/clef/reports/fmtA_48e.val_in.json] [--asr <transcript.json>] -- [cờ thêm cho clef_agent]

Ra: agent/runs/clef/<name>.<split>.jsonl (+ .timings.jsonl), agent/runs/clef/reports/<name>.<split>.json.
In: điểm, F1, state, số mẫu lỗi; với --ref: so cặp từng mẫu (tốt hơn / kém hơn / bằng, chênh ± SE).
Điểm chính KHÔNG tính giới hạn 30 s (mạng local làm sai lệch thời gian); thời gian chỉ in để tham khảo.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_in", choices=["val_in", "val_ood"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--fmt-from", required=True)
    ap.add_argument("--app-filter")
    ap.add_argument("--asr", help="JSON {uid: transcript}; mặc định câu gõ sẵn (sạch)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ref", help="report của lần chạy mốc để so cặp")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--skip-run", action="store_true", help="chỉ chấm lại output đã có")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]

    runs = os.path.join(HERE, "runs", "clef")
    out = os.path.join(runs, f"{a.name}.{a.split}.jsonl")
    rep = os.path.join(runs, "reports", f"{a.name}.{a.split}.json")
    os.makedirs(os.path.dirname(rep), exist_ok=True)

    if not a.skip_run:
        cmd = [sys.executable, os.path.join(HERE, "clef_agent.py"), "--data", a.data, "--split", "train", "--url", a.url,
               "--key", os.environ.get("LLM_API_KEY", ""), "--out", out, "--ids-split", a.split,
               "--workers", str(a.workers), "--fmt-from", a.fmt_from, "--log-probs", *extra]
        if a.app_filter:
            cmd += ["--app-filter", a.app_filter]
        if a.asr:
            cmd += ["--asr", a.asr]
        log = out.replace(".jsonl", ".log")
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            sys.exit(f"clef_agent lỗi rc={rc}, xem {log}")

    ids = set(json.load(open(os.path.join(HERE, "clef", "splits.json")))[a.split])
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as g:
        for l in open(os.path.join(runs, "gold_train.jsonl"), encoding="utf-8"):
            if json.loads(l)["id"] in ids:
                g.write(l)
        gold = g.name
    rc = subprocess.run([sys.executable, "-m", "metrics", "score", "--gold", gold, "--submission", out,
                         "--catalog", f"{a.data}/train/train_tools_catalog.json", "--replay", "--data-root", a.data,
                         "--split", "train", "--output", rep], cwd=ROOT, capture_output=True, text=True).returncode
    os.unlink(gold)
    if rc:
        sys.exit("metrics lỗi")
    d = json.load(open(rep))
    s = d["summary"]
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    lat = sorted(r["latency"] for r in rows if isinstance(r.get("latency"), (int, float)))
    print(f"{a.name} {a.split}: score {s['score']:.1f} | F1 {s['tool_call_f1']:.3f} | state {s['state_match_rate']:.3f} | "
          f"lỗi {s['num_failed_samples']} | mẫu {s['num_samples']} | request/mẫu {st.mean(r['n_requests'] for r in rows):.2f} | "
          f"latency p50 {lat[len(lat) // 2]:.1f}s p90 {lat[int(len(lat) * .9)]:.1f}s")
    if a.ref:
        A = {x["id"]: x["score"] for x in json.load(open(a.ref))["samples"]}
        B = {x["id"]: x["score"] for x in d["samples"]}
        diff = [B[i] - A[i] for i in A if i in B]
        se = st.pstdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0
        print(f"so với {os.path.basename(a.ref)}: tốt hơn {sum(x > 0 for x in diff)} | kém hơn {sum(x < 0 for x in diff)} | "
              f"bằng {sum(x == 0 for x in diff)} | chênh {100 * st.mean(diff):+.1f} ± {100 * se:.1f}")
    print("→", rep)


if __name__ == "__main__":
    main()
