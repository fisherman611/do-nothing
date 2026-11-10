"""E4 verify: bộ lọc đóng gói (pack.py) chạy đúng đường code thật trên val_in / val_ood; đo recall tầng 1, số app giữ và
token của khối tools (đầy đủ vs hai tầng) + cả prompt đầu vào.

    python eval_pack.py --pack <dir pack> --data <data/train+public_test> [--limit 0] [--device cuda]
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from agent import agent as A                       # noqa: E402
from agent.clef import fmt as F                     # noqa: E402
from agent.clef.build_data import TOKENIZER         # noqa: E402
from pack import AppFilterPack                      # noqa: E402
import filter as FL                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--splits", default=os.path.join(ROOT, "agent/clef/splits.json"))
a = ap.parse_args()

F.set_style("v1", q2="all", q1_desc="short", sys="compact", rules="short")          # = fmtA_48
from transformers import AutoTokenizer              # noqa: E402
tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
catalog = A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
sp = json.load(open(a.splits))
pack = AppFilterPack.load(a.pack, device=a.device)
print("pack:", pack.cfg, flush=True)
for name in ("val_in", "val_ood"):
    ids = set(sp[name])
    rs = [r for r in rows if r["id"] in ids]
    if a.limit:
        rs = rs[:a.limit]
    t0 = time.time()
    E = pack.encode(rs)
    st = collections.defaultdict(list)
    miss = []
    for n, r in enumerate(rs):
        order, tier1 = pack.rank(r, E[n])
        gold = FL.gold_apps(r)
        inst = [x for x in r["context"]["installed_apps"] if x != "sys"]
        st["hit"].append(gold <= tier1 if gold else True)
        st["k"].append(len(tier1))
        st["n_inst"].append(len(inst))
        if gold and not gold <= tier1:
            miss.append((r["id"], sorted(gold - tier1)))
        specs = A.tool_specs(r, catalog)
        full_t = F.compact_tools(specs)
        two_t = F.compact_tools(specs, tier1)
        st["tok_full"].append(F.n_tokens(tok, full_t))
        st["tok_two"].append(F.n_tokens(tok, two_t))
        msgs = F.build_messages(r)
        st["prompt_full"].append(F.n_tokens(tok, F.render_prefix(tok, msgs, specs)))
        st["prompt_two"].append(F.n_tokens(tok, F.render_prefix(tok, msgs, specs, tier1)))
        # gold nằm trong 3 app đầu của order (xếp hạng)?
        st["gold_top3"].append(gold <= set(order[:3]) if gold else True)
    m = lambda k: statistics.mean(st[k])             # noqa: E731
    print(f"\n[{name}] n={len(rs)} ({time.time() - t0:.0f}s)  recall tầng 1 {100 * m('hit'):.1f}%  app giữ {m('k'):.2f}/{m('n_inst'):.1f} cài  "
          f"gold⊆top3 {100 * m('gold_top3'):.1f}%")
    print(f"  tokens khối tools: đầy đủ {m('tok_full'):.0f} → hai tầng {m('tok_two'):.0f}  (giảm {100 * (1 - m('tok_two') / m('tok_full')):.0f}%)")
    print(f"  tokens prompt đầu vào: đầy đủ {m('prompt_full'):.0f} → hai tầng {m('prompt_two'):.0f}  (giảm {100 * (1 - m('prompt_two') / m('prompt_full')):.0f}%)")
    print(f"  sót {len(miss)}: {miss[:8]}")
