"""Ráp bộ data FC v3 vòng 1 (PLAN_72h_v3 A1): bản build đầy đủ có nhiễu B0 → chọn tập con có coverage → thêm ví dụ Q1 app lạ
(build_newapps_q1.py) → kiểm tra (không rò val, đủ tool, thành phần) → ghi thư mục + meta + report.

    python agent/clef/make_v3.py --src agent/clef/data/fmtA_v3_full --newapps agent/clef/data/newapps_q1/train.jsonl \
        --out agent/clef/data/fmtA_v3 --target 11500
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COND = re.compile(r"\bnếu\b|\bthì\b|\btrừ\b|\bnhưng\b|\bkhi nào\b", re.I)
SELF = re.compile(r"à không|à thôi|à khoan|khoan đã|nhầm|à mà", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--newapps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=11500, help="số ví dụ của tập con (chưa tính ví dụ app lạ)")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    a = ap.parse_args()
    sub = a.out + "_sub"
    subprocess.run([sys.executable, os.path.join(HERE, "make_subset.py"), "--src", a.src, "--out", sub, "--target",
                    str(a.target), "--data", a.data], check=True)
    os.makedirs(a.out, exist_ok=True)
    ex = [json.loads(l) for l in open(f"{sub}/train.jsonl", encoding="utf-8")]
    na = [json.loads(l) for l in open(a.newapps, encoding="utf-8")]
    with open(f"{a.out}/train.jsonl", "w", encoding="utf-8") as f:
        for e in ex + na:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- kiểm tra ----
    splits = json.load(open(os.path.join(HERE, "splits.json")))
    val_in = set(splits["val_in"])
    base = lambda i: i.split("#")[0]
    leak = {base(e["id"]) for e in ex} & val_in
    na_held = {json.loads(l)["id"].rsplit("#na", 1)[0] for p in ("val_newapps", "test_newapps")
               for l in open(os.path.join(os.path.dirname(a.newapps), f"{p}.jsonl"), encoding="utf-8")}
    leak_na = {e["id"].rsplit("#na", 1)[0] for e in na} & na_held
    orig = {}
    for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8"):
        d = json.loads(l)
        orig[d["id"]] = d
    items = list(dict.fromkeys(base(e["id"]) for e in ex if base(e["id"]) in orig))
    c = collections.Counter()
    for i in items:
        d = orig[i]
        names = [t["name"] for t in d["label"].get("tool_calls", [])]
        c["cond"] += bool(COND.search(d.get("audio_text") or ""))
        c["selfc"] += bool(SELF.search(d.get("audio_text") or ""))
        c["ask"] += "sys.ask_missing_info" in names
        c["multi3"] += len(names) >= 3
    tools = {e["name"] for e in ex if e.get("name")} | {e["name"] for e in na if e.get("name")}
    meta = json.load(open(f"{sub}/train.jsonl.meta.json"))
    report = {"examples_subset": len(ex), "examples_newapps_q1": len(na), "examples_total": len(ex) + len(na),
              "items_subset": len(items), "leak_val_in": sorted(leak), "leak_newapps_val_test": sorted(leak_na),
              "tools_in_data": len(tools), "share_items": {k: round(100 * v / len(items), 1) for k, v in c.items()},
              "cat": dict(collections.Counter(e.get("cat") for e in ex + na)),
              "asr_noise": meta.get("asr_noise"), "subset_stats": meta.get("subset_stats")}
    meta.update(n_examples=len(ex) + len(na), newapps_q1=len(na), v3_report=report)
    json.dump(meta, open(f"{a.out}/train.jsonl.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(report, open(f"{a.out}/report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])
    if leak or leak_na:
        sys.exit("RÒ DỮ LIỆU: xem report.json")


if __name__ == "__main__":
    main()
