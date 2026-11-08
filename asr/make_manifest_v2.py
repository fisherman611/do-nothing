"""Manifest v2 cho contextual biasing mở rộng (PLAN_ASR_bias_v2.md). Không ghi đè manifest.json của v1.

Cùng split với v1 (seed 42, 8% theo sample id) và cùng trường uid/audio/text/sid/kind/split/prev; thêm:
profile, apps, hist_ents (có lúc test), label_ents (chỉ train). Ghi thêm lexicon_v2.json dựng **chỉ từ split train**
(dùng cho tên gây nhiễu khi train và cho điều kiện eval "lexicon").

    python make_manifest_v2.py <root chứa train/train.jsonl>
    → modal/manifest_v2.json, modal/lexicon_v2.json
    modal volume put zaic-data modal/manifest_v2.json manifest_v2.json
    modal volume put zaic-data modal/lexicon_v2.json lexicon_v2.json
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_prompt import build_lexicon, items_from_row  # noqa: E402

root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(f"{root}/train/train.jsonl")]
ids = sorted(r["id"] for r in rows)
random.Random(42).shuffle(ids)
val = set(ids[: int(len(ids) * 0.08)])
out = []
for r in rows:
    out += items_from_row(r, split="val" if r["id"] in val else "train", with_label=True)
lex = build_lexicon([r for r in rows if r["id"] not in val])

# kiểm tra: các trường v1 giữ nguyên
v1p = os.path.join(here, "manifest.json")
if os.path.exists(v1p):
    v1 = {x["uid"]: x for x in json.load(open(v1p))}
    keys = ("uid", "audio", "text", "sid", "kind", "split", "prev")
    diff = [x["uid"] for x in out if {k: x[k] for k in keys} != {k: v1[x["uid"]][k] for k in keys}]
    assert len(v1) == len(out) and not diff, f"lệch manifest v1: {len(v1)} vs {len(out)}, {diff[:5]}"
    print("trường v1 khớp manifest.json")
json.dump(out, open(os.path.join(here, "manifest_v2.json"), "w"), ensure_ascii=False)
json.dump(lex, open(os.path.join(here, "lexicon_v2.json"), "w"), ensure_ascii=False)
print(Counter((o["split"], o["kind"]) for o in out))
print("utt có hist_ents:", sum(bool(o["hist_ents"]) for o in out), "| có label_ents:", sum(bool(o["label_ents"]) for o in out))
print("lexicon:", len(lex), "app,", sum(len(v) for f in lex.values() for v in f.values()), "mục")
