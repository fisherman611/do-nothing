"""Chia val cho SFT / Clef-token (PLAN_SFT_qwen35_text.md mục 3), cố định seed.

- val_ood: mẫu có GỌI tool của ~8 app giữ lại (proxy cho 11 app lạ ở public test). Các app giữ lại chọn trong nhóm
  tần suất vừa (1–6% số mẫu), không lấy app lõi; mọi ví dụ train (kể cả lượt history) gọi các app này đều bị loại.
- val_in: 10% ngẫu nhiên trong phần còn lại.
- train: phần còn lại.

    python agent/clef/splits.py data/train+public_test   → agent/clef/splits.json
"""
import collections
import json
import os
import random
import sys

CORE = {"contacts", "sms", "calendar", "smart_home", "weather", "phone", "sys"}


def main(root):
    rows = [json.loads(l) for l in open(f"{root}/train/train.jsonl", encoding="utf-8")]
    n = len(rows)
    called = {r["id"]: {c["name"].split(".")[0] for c in r["label"]["tool_calls"]} for r in rows}
    freq = collections.Counter(a for s in called.values() for a in s)
    pool = sorted(a for a, c in freq.items() if 0.01 * n <= c <= 0.06 * n and a not in CORE)
    rng = random.Random(0)
    rng.shuffle(pool)
    held, cover = [], set()
    for a in pool:
        if len(held) >= 8 or len(cover) >= 0.15 * n:
            break
        held.append(a)
        cover |= {i for i, s in called.items() if a in s}
    rest = sorted(i for i in called if i not in cover)
    rng = random.Random(1)
    rng.shuffle(rest)
    k = int(0.1 * len(rest))
    out = {"held_out_apps": sorted(held), "val_ood": sorted(cover), "val_in": sorted(rest[:k]),
           "train": sorted(rest[k:])}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")
    json.dump(out, open(path, "w"), indent=1)
    print({k: len(v) for k, v in out.items()}, "| app giữ lại:", held, {a: freq[a] for a in held})


if __name__ == "__main__":
    main(sys.argv[1])
