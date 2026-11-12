"""Ghép 6 shard train.s*.jsonl -> train.jsonl và hợp meta (cộng các trường số, lấy trung bình tier1_avg)."""
import glob, json, os, sys
d = sys.argv[1]
shards = sorted(glob.glob(f"{d}/train.s*.jsonl"))
assert len(shards) == 6, shards
with open(f"{d}/train.jsonl", "w", encoding="utf-8") as out:
    for p in shards:
        with open(p, encoding="utf-8") as f:
            for line in f:
                out.write(line)
metas = [json.load(open(p + ".meta.json")) for p in shards]
m = dict(metas[0])
def merge(key, vals):
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return round(sum(vals) / len(vals), 2) if key == "tier1_avg" else sum(vals)
    if all(isinstance(v, dict) for v in vals):
        return {k: merge(k, [v[k] for v in vals if k in v]) for k in dict.fromkeys(kk for v in vals for kk in v)}
    return vals[0]
for k in m:
    if k in ("spans", "asr_noise", "n_items", "n_examples", "skipped", "n_items_full_catalog", "tier1_avg"):
        m[k] = merge(k, [x[k] for x in metas if k in x])
m["shard"] = None
json.dump(m, open(f"{d}/train.jsonl.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = sum(1 for _ in open(f"{d}/train.jsonl", encoding="utf-8"))
print("lines", n, "meta n_examples", m["n_examples"], "asr_noise", m["asr_noise"])
