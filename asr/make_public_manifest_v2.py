"""Manifest v2 (prompt contextual biasing) cho public_test — chỉ dùng trường có lúc thi (history, context), không nhãn.
    python make_public_manifest_v2.py <root chứa public_test/public_test.jsonl> <out public_manifest_v2.json>
Không đọc public_test_environment/."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bias_prompt import items_from_row  # noqa: E402

root, out_path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{root}/public_test/public_test.jsonl", encoding="utf-8")]
out = []
for r in rows:
    for it in items_from_row(r, split="public_test", with_label=False):
        it["root"] = "public_test"
        out.append(it)
json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
kinds = {}
for o in out:
    kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
print(f"{len(rows)} mẫu → {len(out)} utt {kinds}; có hist_ents: {sum(bool(o['hist_ents']) for o in out)}")
