"""Kiểm tra định dạng Clef-token trên dữ liệu train thật.

    python agent/clef/test_fmt.py --data data/train+public_test [--limit 300]

1. Mọi nhãn " X" là 1 token.
2. Tại mỗi vị trí choice: token hoá text[:pos] (prompt lúc suy luận) là tiền tố của token hoá cả chuỗi, và token kế
   tiếp đúng là nhãn vàng → softmax giới hạn đọc đúng chỗ, train = suy luận.
3. Tại mỗi đoạn lm: token hoá text[:start] là tiền tố của cả chuỗi (đếm số ca lệch ranh giới).
4. Tool call dựng bằng fmt giống hệt chat template của Qwen3.5 (cùng thứ tự tham số).
5. Tham số dựng lại (sau khi parse giá trị tự do) bằng tham số vàng.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef import fmt as F                     # noqa: E402
from agent.clef.build_data import TOKENIZER, examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--style", default="v1", choices=list(F.STYLES))
    ap.add_argument("--q2", default="all", choices=F.Q2_MODES)
    ap.add_argument("--q1-desc", default="full", choices=F.Q1_DESC_MODES)
    a = ap.parse_args()
    F.set_style(a.style, q2=a.q2, q1_desc=a.q1_desc)
    print("format:", F.fmt_opts())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    assert all(len(tok(" " + l, add_special_tokens=False).input_ids) == 1 for l in labels)
    assert all(len(tok(" " + w, add_special_tokens=False).input_ids) == 1 for w in (F.YES, F.NO))
    print("1. nhãn 1 token:", len(labels), "nhãn + có/không  OK")

    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[: a.limit]
    gold = {r["id"]: r["label"]["tool_calls"] for r in rows}
    stat = collections.Counter()
    bad = collections.defaultdict(list)
    for r in rows:
        exs, skipped = examples(r, tok, catalog, labels, random.Random(0))
        stat["bước bị bỏ"] += skipped
        for e in exs:
            stat["ví dụ"] += 1
            text = e["text"]
            full = tok(text, add_special_tokens=False).input_ids
            for sp in e["spans"]:
                if sp["type"] == "choice":
                    stat["choice"] += 1
                    pre = tok(text[: sp["pos"]], add_special_tokens=False).input_ids
                    want = tok(" " + sp["gold"], add_special_tokens=False).input_ids[0]
                    if full[: len(pre)] != pre or full[len(pre)] != want:
                        bad["2. choice lệch token"].append((e["id"], e["step"], sp["gold"]))
                else:
                    stat["lm"] += 1
                    pre = tok(text[: sp["start"]], add_special_tokens=False).input_ids
                    if full[: len(pre)] != pre:
                        bad["3. lm lệch ranh giới"].append((e["id"], e["step"], text[sp["start"]: sp["start"] + 20]))
            if e["name"] is None:
                continue
            schema = catalog[e["name"]]
            g = gold[r["id"]][e["step"]]
            # 4. so với chat template
            m = [{"role": "user", "content": "x"},
                 F.A._call_msg([{"name": e["name"], "arguments": F.canonical_args(g["arguments"], schema)}])]
            tpl = tok.apply_chat_template(m, tokenize=False, enable_thinking=False)
            tpl_call = tpl[tpl.index("<tool_call>"): tpl.index("</tool_call>") + len("</tool_call>")]
            ours = text[text.rindex("<tool_call>"):]
            if tpl_call != ours:
                bad["4. tool call khác template"].append((e["id"], e["step"], ours[-120:], tpl_call[-120:]))
            # 5. tham số
            if e["args"] != F.canonical_args(g["arguments"], schema) or list(e["args"]) != list(F.canonical_args(g["arguments"], schema)):
                bad["5. tham số khác nhãn"].append((e["id"], e["step"], e["args"], g["arguments"]))
    print("thống kê:", dict(stat))
    for k in sorted(bad):
        print(f"{k}: {len(bad[k])}", bad[k][:3])
    if not bad:
        print("2–5 OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
