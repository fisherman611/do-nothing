"""Gộp các eval_*.json (filter.py eval) thành bảng markdown: với mỗi model, recall ở các cấu hình chuẩn và
"k nhỏ nhất đạt ngưỡng recall" trên train.

    python agent/clef/app_filter/summarize.py agent/clef/app_filter/zs/*.json agent/clef/app_filter/eval_bm25_kw.json
"""
import glob
import json
import os
import sys

files = [f for p in sys.argv[1:] for f in sorted(glob.glob(p))]
rows = []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    name = d.get("emb") or "BM25"
    cfgs = d["configs"]

    def pick(prefix):
        return next((v for k, v in cfgs.items() if k.startswith(prefix + " ")), None)
    std = {"emb@5": pick("bm25@0 ∪ emb@5") or pick("bm25@0 ∪ rr@5"),
           "bm25@3∪emb@4": pick("bm25@3 ∪ emb@4") or pick("bm25@3 ∪ rr@4"),
           "bm25@4∪emb@5": pick("bm25@4 ∪ emb@5") or pick("bm25@4 ∪ rr@5"),
           "bm25@5": pick("bm25@5") if name == "BM25" else None,
           "bm25@8": pick("bm25@8") if name == "BM25" else None}
    # cấu hình nhỏ nhất (theo app giữ TB trên train) đạt recall ≥ 0.98 / 0.99 trên train
    best = {}
    for thr in (0.95, 0.98, 0.99):
        ok = [(v["train"]["kept_avg"], k, v) for k, v in cfgs.items() if v["train"]["recall"] >= thr]
        best[thr] = min(ok) if ok else None
    rows.append((name, std, best))

print("| Model | emb@5 (train / val_in / val_ood) | bm25@3 ∪ emb@4 | bm25@4 ∪ emb@5 | nhỏ nhất ≥95% train | ≥98% | ≥99% |")
print("|---|---|---|---|---|---|---|")


def cell(v):
    if not v:
        return "–"
    return f"{100 * v['train']['recall']:.1f} / {100 * v['val_in']['recall']:.1f} / {100 * v['val_ood']['recall']:.1f} ({v['train']['kept_avg']:.1f} app)"


def bcell(b):
    if not b:
        return "không đạt"
    kept, k, v = b
    return f"{k.replace(' +luật', '')}: {kept:.1f} app (val_in {100 * v['val_in']['recall']:.0f}, val_ood {100 * v['val_ood']['recall']:.0f})"


for name, std, best in rows:
    short = name.split("/")[-1]
    if name == "BM25":
        print(f"| {short} | {cell(std['bm25@5'])} (bm25@5) | – | {cell(std['bm25@8'])} (bm25@8) | {bcell(best[0.95])} | {bcell(best[0.98])} | {bcell(best[0.99])} |")
    else:
        print(f"| {short} | {cell(std['emb@5'])} | {cell(std['bm25@3∪emb@4'])} | {cell(std['bm25@4∪emb@5'])} | {bcell(best[0.95])} | {bcell(best[0.98])} | {bcell(best[0.99])} |")
