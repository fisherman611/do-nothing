"""Phân tích lỗi một file kết quả agent so với nhãn train: điểm khác biệt đầu tiên giữa chuỗi call nộp và chuỗi vàng.

    python agent/clef/analyze.py --data data/train+public_test --run agent/runs/clef/zs_train.jsonl [--run2 ...]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef import fmt as F   # noqa: E402


def tok_f1(a, b):
    a, b = str(a).lower().split(), str(b).lower().split()
    if not a or not b:
        return float(a == b)
    c = sum((collections.Counter(a) & collections.Counter(b)).values())
    return 0 if c == 0 else 2 * c / (len(a) + len(b))


def first_diff(sub, gold, catalog):
    """→ (loại lỗi, chi tiết) ở vị trí khác nhau đầu tiên; ('ok', None) nếu giống hệt."""
    for i in range(max(len(sub), len(gold))):
        if i >= len(sub):
            return ("dừng sớm" if sub else "không gọi gì"), gold[i]["name"]
        if i >= len(gold):
            return ("gọi thừa ở cuối" if gold else "gọi khi nhãn rỗng"), sub[i]["name"]
        s, g = sub[i], gold[i]
        if s["name"] != g["name"]:
            if g["name"] == F.ASK:
                return "không hỏi lại (ask)", s["name"]
            if s["name"] == F.ASK:
                return "hỏi lại thừa (ask)", g["name"]
            return "sai tool", f"{g['name']} → {s['name']}"
        props = catalog.get(g["name"], {}).get("parameters", {}).get("properties", {})
        req = set(catalog.get(g["name"], {}).get("parameters", {}).get("required", []))
        for k in g["arguments"]:
            if k not in s["arguments"]:
                return f"thiếu tham số {'bắt buộc' if k in req else 'tùy chọn'} ({F.param_kind(props.get(k, {}))})", f"{g['name']}.{k}"
        for k in s["arguments"]:
            if k not in g["arguments"]:
                return f"thừa tham số ({F.param_kind(props.get(k, {}))})", f"{g['name']}.{k}"
        for k, gv in g["arguments"].items():
            sv = s["arguments"][k]
            p = props.get(k, {})
            if p.get("x-match") == "fuzzy":
                if tok_f1(sv, gv) <= 0.6:
                    return "sai giá trị (text fuzzy)", f"{g['name']}.{k}"
            elif sv != gv:
                kind = F.param_kind(p)
                if kind == "free":
                    kind = "thời gian" if isinstance(gv, str) and gv[:2] == "20" and ("T" in gv or "-" in gv) else \
                        ("số" if isinstance(gv, (int, float)) else "chuỗi")
                return f"sai giá trị ({kind})", f"{g['name']}.{k}: {gv!r} → {sv!r}"
    return "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", nargs="+", required=True)
    a = ap.parse_args()
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    gold = {json.loads(l)["id"]: json.loads(l)["label"]["tool_calls"]
            for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")}
    res = {}
    for run in a.run:
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(run) if l.strip()}
        res[run] = {i: first_diff(r["tool_calls"], gold[i], catalog) for i, r in rows.items()}
    kinds = collections.Counter(k for r in res.values() for k, _ in r.values())
    names = [os.path.basename(r) for r in a.run]
    print(f"{'loại (khác biệt đầu tiên)':40}" + "".join(f"{n[:22]:>24}" for n in names))
    for k, _ in kinds.most_common():
        print(f"{k:40}" + "".join(f"{sum(v[0] == k for v in res[r].values()):>24}" for r in a.run))
    for r in a.run:
        det = collections.Counter(d for k, d in res[r].values() if k.startswith(("thiếu tham số tùy chọn", "dừng sớm")))
        print(f"\n{os.path.basename(r)}: hay gặp nhất (thiếu tùy chọn / dừng sớm):", det.most_common(10))


if __name__ == "__main__":
    main()
