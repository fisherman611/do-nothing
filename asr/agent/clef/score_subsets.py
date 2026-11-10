"""Chấm một file kết quả agent (agent.py / clef_agent.py) bằng đúng bộ `metrics/` của challenge, trên các tập trong
splits.json, đủ 3 điều kiện: hash (replay bằng ToolEnv là có thẩm quyền), call-F1, luật 30 giây (--timings).

    python agent/clef/score_subsets.py --gold agent/runs/clef/gold_train.jsonl --run agent/runs/clef/<run>.jsonl \\
        [--sets val_in val_ood] [--timings <run>.timings.jsonl] [--no-replay] [--no-timings] [--out-dir <dir>]
gold: `python -m metrics prepare-gold --data-root data/train+public_test --output agent/runs/clef/gold_train.jsonl`
Mặc định: --timings = <run>.timings.jsonl nếu có (clef_agent.py tự ghi), replay bật. Timings đo với workers > 1 bị
đội lên do chen GPU → in cảnh báo; số áp luật 30 s chỉ tin khi workers = 1.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score(gold, run, ids, catalog, timings=None, replay=True, data_root=None, split="train", out_json=None):
    with tempfile.TemporaryDirectory() as d:
        g, s, t, o = (os.path.join(d, x) for x in ("g.jsonl", "s.jsonl", "t.jsonl", "o.json"))
        for src, dst in ((gold, g), (run, s), (timings, t)):
            if src is None:
                continue
            with open(dst, "w", encoding="utf-8") as f:
                for l in open(src, encoding="utf-8"):
                    if l.strip() and json.loads(l)["id"] in ids:
                        f.write(l)
        cmd = [sys.executable, "-m", "metrics", "score", "--gold", g, "--submission", s, "--catalog", catalog,
               "--output", o, "--split", split]
        if timings:
            cmd += ["--timings", t]
        if replay:
            cmd += ["--replay", "--data-root", data_root or os.path.join(ROOT, "data/train+public_test")]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode:
            raise SystemExit(f"metrics score lỗi:\n{p.stderr[-2000:]}")
        rep = json.load(open(o))
        if out_json:
            json.dump(rep, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--sets", nargs="+", default=["val_in", "val_ood"])
    ap.add_argument("--catalog", default=os.path.join(ROOT, "data/train+public_test/train/train_tools_catalog.json"))
    ap.add_argument("--timings", help="mặc định <run>.timings.jsonl nếu tồn tại")
    ap.add_argument("--no-timings", action="store_true", help="không áp luật 30 s (chỉ chẩn đoán)")
    ap.add_argument("--no-replay", action="store_true", help="tin hash khai báo (nhanh, chỉ chẩn đoán)")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data/train+public_test"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", help="lưu report JSON đầy đủ: <out-dir>/<run>.<set>.json")
    a = ap.parse_args()
    sp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")))
    for run in a.run:
        have = {json.loads(l)["id"] for l in open(run, encoding="utf-8") if l.strip()}
        timings = None
        if not a.no_timings:
            timings = a.timings or os.path.splitext(run)[0] + ".timings.jsonl"
            if not os.path.exists(timings):
                print(f"(không có {os.path.basename(timings)} → chưa áp luật 30 s)")
                timings = None
            else:
                w = {json.loads(l).get("workers", 1) for l in open(timings, encoding="utf-8") if l.strip()}
                if w - {1}:
                    print(f"CẢNH BÁO: timings đo với workers={sorted(w)} (chen GPU) → số timeout chỉ để tham khảo")
        for name in a.sets:
            ids = set(sp[name]) & have
            if not ids:
                continue
            out_json = os.path.join(a.out_dir, f"{os.path.splitext(os.path.basename(run))[0]}.{name}.json") if a.out_dir else None
            if a.out_dir:
                os.makedirs(a.out_dir, exist_ok=True)
            rep = score(a.gold, run, ids, a.catalog, timings=timings, replay=not a.no_replay,
                        data_root=a.data_root, split=a.split, out_json=out_json)
            s = rep["summary"]
            flags = []
            if s.get("num_hash_discrepancies"):
                flags.append(f"HASH KHAI≠REPLAY {s['num_hash_discrepancies']}")
            if s.get("num_replay_timeouts"):
                flags.append(f"replay timeout {s['num_replay_timeouts']}")
            print(f"{os.path.basename(run):34} {name:8} n={s['num_samples']:4}  điểm {s['score']:5.1f}  "
                  f"F1 {s['tool_call_f1'] * 100:5.1f}  tie-break {s['tie_break_score']:5.1f}  state {s['state_match_rate']:.3f}  "
                  f"timeout {s['num_timeout_samples']:3}  lỗi {s['num_failed_samples'] - s['num_timeout_samples']:3}  "
                  f"hash={s['hash_verification']}{'  ' + ' | '.join(flags) if flags else ''}")


if __name__ == "__main__":
    main()
