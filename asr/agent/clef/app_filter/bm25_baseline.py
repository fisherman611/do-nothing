"""Baseline lọc app bằng BM25 (không model): recall của tập app vàng theo k, trên val_in / val_ood của clef.

Tài liệu app = name_vi + mô tả app + mô tả các tool. Query = yêu cầu cuối (+ text assistant lượt trước).
Luật bổ sung: + app đã gọi trong history, + app cung cấp id theo x-ref của app đã chọn.
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter

D = sys.argv[1]
SPLITS = json.load(open(sys.argv[2], encoding="utf-8"))
cat = json.load(open(f"{D}/train/train_tools_catalog.json", encoding="utf-8"))
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/train/train.jsonl", encoding="utf-8")}


def toks(s):
    s = unicodedata.normalize("NFC", s.lower())
    w = re.findall(r"\w+", s)
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


docs, deps = {}, {}
for app in cat["apps"]:
    text = " ".join([app["name_vi"], app["description"]] + [t["description"] for t in app["tools"]])
    docs[app["id"]] = toks(text)
    deps[app["id"]] = {p["x-ref"].split(".")[0] for t in app["tools"]
                       for p in t["parameters"].get("properties", {}).values() if "x-ref" in p} - {app["id"]}
N = len(docs)
df = Counter(w for d in docs.values() for w in set(d))
avg = sum(map(len, docs.values())) / N


def bm25(q, d, k1=1.2, b=0.75):
    tf = Counter(d)
    return sum(math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
               (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in set(q) if w in tf)


def run(ids, k, rules):
    hit = kept = n = 0
    for i in ids:
        r = rows[i]
        gold = {c["name"].split(".")[0] for c in r["label"]["tool_calls"]} - {"sys"}
        inst = [a for a in r["context"]["installed_apps"] if a != "sys"]
        q = r["audio_text"] + " " + " ".join(t.get("text", "") for t in r["history"][-1:] if t["role"] == "assistant")
        rank = sorted(inst, key=lambda a: -bm25(toks(q), docs.get(a, [])))
        keep = set(rank[:k])
        if rules:
            keep |= {c["name"].split(".")[0] for t in r["history"] if t["role"] == "assistant"
                     for c in t.get("tool_calls") or []} & set(inst)
            keep |= {d for a in list(keep) for d in deps.get(a, set())} & set(inst)
        n += 1
        hit += gold <= keep
        kept += len(keep)
    return hit / n, kept / n


for split in ("val_in", "val_ood"):
    ids = SPLITS[split]
    print(split, len(ids), "mẫu")
    for k in (1, 2, 3, 4, 5, 6):
        r0, k0 = run(ids, k, False)
        r1, k1_ = run(ids, k, True)
        print(f"  top-{k}: recall {r0:.1%} (giữ {k0:.1f} app) | + luật history/x-ref: recall {r1:.1%} (giữ {k1_:.1f} app)")
