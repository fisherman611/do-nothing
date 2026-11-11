"""Chấm ASR trên manifest: WER/CER (chuẩn hoá chữ thường, bỏ dấu câu), lỗi số, lỗi tên riêng; xuất cặp (ref, hyp) có lỗi.
    python agent/asr_score.py --manifest agent/runs/asr/val.manifest.json --asr agent/runs/asr/val.asr.json [--out-pairs agent/runs/asr/val.pairs.jsonl]
"""
import argparse
import collections
import json
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFC", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]


DIGITS = re.compile(r"\d")
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--asr", required=True)
ap.add_argument("--out-pairs")
a = ap.parse_args()
items = json.load(open(a.manifest, encoding="utf-8"))["items"]
H = json.load(open(a.asr, encoding="utf-8"))["hyps"]
tot = collections.Counter()
by_kind = collections.defaultdict(collections.Counter)
pairs = []
for it in items:
    ref, hyp = norm(it.get("ref")), norm(H.get(it["uid"]))
    if not ref:
        continue
    w = edit(ref.split(), hyp.split())
    c = edit(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    for t in (tot, by_kind[it["kind"]]):
        t["n"] += 1; t["w_err"] += w; t["w_ref"] += len(ref.split()); t["c_err"] += c; t["c_ref"] += len(ref.replace(" ", ""))
        t["exact"] += ref == hyp
    rd, hd = DIGITS.findall(it["ref"] or ""), DIGITS.findall(H.get(it["uid"], ""))
    if rd or hd:
        tot["num_utts"] += 1
        tot["num_diff"] += norm(it["ref"]) != hyp and (rd != hd)
    if w:
        pairs.append({"uid": it["uid"], "kind": it["kind"], "ref": it["ref"], "hyp": H.get(it["uid"], ""), "wer": round(w / max(1, len(ref.split())), 3),
                      "cer": round(c / max(1, len(ref.replace(" ", ""))), 3)})
def show(name, t):
    print(f"{name:8} n={t['n']:5}  WER {100 * t['w_err'] / t['w_ref']:5.2f}%  CER {100 * t['c_err'] / t['c_ref']:5.2f}%  câu đúng hoàn toàn {100 * t['exact'] / t['n']:5.1f}%")
show("tất cả", tot)
for k, t in by_kind.items():
    show(k, t)
print(f"câu có chữ số trong ref/hyp: {tot['num_utts']}, trong đó lệch chữ số: {tot['num_diff']}")
if a.out_pairs:
    with open(a.out_pairs, "w", encoding="utf-8") as f:
        for p in sorted(pairs, key=lambda x: -x["cer"]):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} cặp có lỗi → {a.out_pairs}")
