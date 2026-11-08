"""Dựng dữ liệu train Clef-token theo từng bước từ nhãn train (PLAN_FC_clef_qwen35.md mục 4).

Mỗi bước có nhãn → một ví dụ {id, step, text, spans}: text = prompt chuẩn + Q1 + đáp án + Q2 + đáp án + tool call
(hoặc chỉ Q1 = DỪNG ở bước cuối). spans: vị trí ký tự cần tính loss (choice: nhãn 1 token trong tập nhãn của câu
hỏi; lm: giá trị tham số tự do + "\\n</parameter>"). Kết quả tool của các bước lấy từ `label.tool_results` (có sẵn
trong train.jsonl, không đọc *_environment/).

    python agent/clef/build_data.py --data data/train+public_test --split-name train --out agent/clef/data/train.jsonl \
        [--history] [--shuffle-seed 0]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef import fmt as F            # noqa: E402
from agent.clef.core import GoldOracle, SkipStep, run_step  # noqa: E402

TOKENIZER = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def history_items(item):
    """Mỗi lượt assistant có tool call trong history → một mẫu phụ: history cắt tới trước lượt user tương ứng."""
    out, hist = [], item["history"]
    for j, t in enumerate(hist):
        if t["role"] != "assistant" or not t.get("tool_calls") or j == 0 or hist[j - 1]["role"] != "user":
            continue
        out.append(dict(item, id=f"{item['id']}#h{j}", history=hist[: j - 1],
                        audio_text=hist[j - 1].get("audio_text", ""),
                        label={"tool_calls": t["tool_calls"], "tool_results": t.get("tool_results", [])}))
    return out


def examples(item, tok, catalog, labels, rng):
    """Các ví dụ theo bước của một mẫu + số bước bị bỏ (SkipStep)."""
    msgs = F.A.build_messages(item)
    tools = F.A.tool_specs(item, catalog)
    seen = [(c["name"], r) for t in item["history"] if t["role"] == "assistant"
            for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or [])]
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    out, skipped = [], 0
    hint = {"request": item.get("audio_text", "") or "", "done": []}
    for i in range(len(calls) + 1):
        if i == len(calls) and calls and calls[-1]["name"] == F.ASK:
            break                                   # vòng lặp tự dừng sau sys.ask_missing_info
        gold = calls[i] if i < len(calls) else None
        prefix = F.render_prefix(tok, msgs, tools)
        o = GoldOracle(gold)
        try:
            r = run_step(prefix, tools, catalog, seen, labels, o, tok, rng, hint={**hint, "done": list(hint["done"])})
            out.append({"id": item["id"], "step": i, "prefix_chars": len(prefix), "text": r["text"],
                        "spans": o.spans, "args": r["args"], "name": r["name"]})
        except SkipStep as e:
            skipped += 1
        if gold is not None:
            schema = catalog.get(gold["name"], {"parameters": {"properties": {}}})
            F.append_step(msgs, {"name": gold["name"], "arguments": F.canonical_args(gold["arguments"], schema)},
                          results[i])
            seen.append((gold["name"], results[i]))
            hint["done"].append(gold["name"])
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-name", default="train", help="khóa trong splits.json: train / val_in / val_ood")
    ap.add_argument("--out", required=True)
    ap.add_argument("--history", action="store_true", help="thêm mẫu phụ từ các lượt assistant trong history")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="<0: không xáo phương án")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--style", default="v1", choices=list(F.STYLES))
    a = ap.parse_args()
    F.set_style(a.style)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    splits = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")))
    keep = set(splits[a.split_name])
    held = set(splits["held_out_apps"])
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["id"] in keep][: a.limit]
    rng = random.Random(a.shuffle_seed) if a.shuffle_seed >= 0 else None
    items = []
    for r in rows:
        items.append(r)
        if a.history:
            items += [h for h in history_items(r)
                      if not ({c["name"].split(".")[0] for c in h["label"]["tool_calls"]} & held)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n, skip, kinds = 0, 0, {}
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            ex, s = examples(it, tok, catalog, labels, rng)
            skip += s
            for e in ex:
                for sp in e["spans"]:
                    kinds[sp["type"]] = kinds.get(sp["type"], 0) + 1
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.split_name}: {len(items)} mẫu → {n} ví dụ, bỏ {skip} bước | spans {kinds}")


if __name__ == "__main__":
    main()
