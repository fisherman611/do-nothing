"""Tạo manifest.json từ train.jsonl: mỗi audio (lượt cuối + lượt user trong history) là một utt.

Chia train/val theo sample id (seed 42, 8%). Trường `prev` giữ các lượt trước utt đó
(cả user lẫn assistant) để làm prompt (contextual biasing) khi train/infer:
    prev = [[role, text, uid_or_None], ...]   # uid chỉ có ở lượt user (để thay bằng transcript ASR lúc infer)

python make_manifest.py <root chứa train/train.jsonl>   → ghi modal/manifest.json
"""
import json, random, sys
from collections import Counter

root = sys.argv[1]
rows = [json.loads(l) for l in open(f"{root}/train/train.jsonl")]
ids = sorted(r["id"] for r in rows)
random.Random(42).shuffle(ids)
val = set(ids[: int(len(ids) * 0.08)])
out = []
for r in rows:
    sp = "val" if r["id"] in val else "train"
    prev = []
    for t in r["history"]:
        if t["role"] == "user":
            uid = t["audio"].split("/")[-1][:-4]
            out.append(dict(uid=uid, audio=t["audio"], text=t["audio_text"], sid=r["id"], kind="history",
                            split=sp, prev=list(prev)))
            prev.append(["user", t["audio_text"], uid])
        else:
            prev.append(["assistant", t.get("text", ""), None])
    out.append(dict(uid=r["id"], audio=r["audio"], text=r["audio_text"], sid=r["id"], kind="final",
                    split=sp, prev=prev))
json.dump(out, open("modal/manifest.json", "w"), ensure_ascii=False)
print(Counter((o["split"], o["kind"]) for o in out), "| utt có prev:", sum(bool(o["prev"]) for o in out))
