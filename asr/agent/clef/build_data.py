"""Dựng dữ liệu train Clef-token theo từng bước từ nhãn train (PLAN_FC_clef_qwen35.md mục 4).

Mỗi bước có nhãn → một ví dụ {id, step, text, spans}: text = prompt chuẩn + Q1 + đáp án + Q2 + đáp án + tool call
(hoặc chỉ Q1 = DỪNG ở bước cuối). spans: vị trí ký tự cần tính loss (choice: nhãn 1 token trong tập nhãn của câu
hỏi; lm: giá trị tham số tự do + "\\n</parameter>"). Kết quả tool của các bước lấy từ `label.tool_results` (có sẵn
trong train.jsonl, không đọc *_environment/).

    python agent/clef/build_data.py --data data/train+public_test --split-name train --out agent/clef/data/train.jsonl \
        [--history] [--shuffle-seed 0] [--q2 all|discrete|none] [--q1-desc full|short|none] [--sys json|compact] [--rules full|short]
Ghi kèm <out>.meta.json (format + thống kê); train.py chép vào config.json, clef_agent.py --fmt-from đọc lại.
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
    """Mỗi lượt assistant trong history → một mẫu phụ: history cắt tới trước lượt user tương ứng. Lượt không gọi tool
    (cảm ơn, điều kiện chưa đủ...) cũng giữ, thành mẫu DỪNG ngay (PLAN_prompt_format_v2.md 1.4)."""
    out, hist = [], item["history"]
    for j, t in enumerate(hist):
        if t["role"] != "assistant" or j == 0 or hist[j - 1]["role"] != "user":
            continue
        out.append(dict(item, id=f"{item['id']}#h{j}", history=hist[: j - 1],
                        audio_text=hist[j - 1].get("audio_text", ""),
                        label={"tool_calls": t.get("tool_calls") or [], "tool_results": t.get("tool_results") or []}))
    return out


def examples(item, tok, catalog, labels, rng, tier1=None):
    """Các ví dụ theo bước của một mẫu + số bước bị bỏ (SkipStep)."""
    if "catalog_overlay" in item and item["catalog_overlay"]:
        cat = dict(catalog)
        for t in item["catalog_overlay"]:
            cat[t["name"]] = t
        catalog = cat
    msgs = F.build_messages(item)
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
        prefix = F.render_prefix(tok, msgs, tools, tier1)
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
    ap.add_argument("--phonetic", action="append", default=[],
                    help="jsonl biến thể phiên âm (synth_phonetic/train_phonetic.jsonl); lấy theo split của id gốc. "
                         "Với --history: lượt history bị phiên âm cũng thành mẫu phụ (bỏ mẫu phụ trùng y hệt bản gốc)")
    ap.add_argument("--synth", action="append", default=[],
                    help="jsonl dữ liệu tổng hợp (synth/train_alias.jsonl, synth/train_iface.jsonl)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--style", default="v1", choices=list(F.STYLES))
    ap.add_argument("--q2", default="all", choices=F.Q2_MODES, help="all: hỏi Q2 mọi tham số; discrete: chỉ ref/enum/bool, "
                    "tham số tự do tuỳ chọn model tự sinh ở đuôi call; none: chỉ Q1")
    ap.add_argument("--q1-desc", default="full", choices=F.Q1_DESC_MODES, help="độ dài mô tả tool trong Q1")
    ap.add_argument("--sys", default="json", choices=F.SYS_MODES, help="tools trong system prompt: json (chat template) | compact")
    ap.add_argument("--rules", default="full", choices=F.RULES_MODES, help="quy tắc LifeMate: full | short")
    ap.add_argument("--app-filter", help="thư mục gói bộ lọc (app_filter/pack.py): catalog HAI TẦNG, tầng 1 = bộ lọc ∪ gold ∪ decoy (chỉ với --sys compact)")
    ap.add_argument("--decoy-apps", type=int, default=2, help="số app nhiễu ngẫu nhiên thêm vào tầng 1 khi dựng train")
    ap.add_argument("--filter-device", default="cpu")
    ap.add_argument("--full-catalog-p", type=float, default=0.0,
                    help="với --app-filter: tỉ lệ mẫu render catalog ĐẦY ĐỦ (không hai tầng) để model quen cả hai dạng")
    ap.add_argument("--asr-errors", help="A5b asr_errors_<tag>.jsonl: thay nguyên câu bằng bản ASR thật (chỉ train)")
    ap.add_argument("--asr-subst", help="A5b asr_subst_<tag>.json: bảng thay thế theo cụm từ")
    ap.add_argument("--asr-p", type=float, default=0.4, help="tỉ lệ mẫu bị nhiễu ASR (câu hiện tại)")
    ap.add_argument("--asr-hist-p", type=float, default=0.3, help="xác suất mỗi lượt user trong history bị nhiễu (trong mẫu bị nhiễu)")
    ap.add_argument("--shard", default="0/1", help="i/n: chỉ dựng các mẫu items[i::n] (chạy song song rồi cat)")
    a = ap.parse_args()
    F.set_style(a.style, q2=a.q2, q1_desc=a.q1_desc, sys=a.sys, rules=a.rules)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    splits = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")))
    keep = {i for s in a.split_name.split(",") for i in splits[s]}      # "train,val_ood": gộp nhiều split
    held = set(splits["held_out_apps"]) if "val_ood" not in a.split_name.split(",") else set()
    if not held:
        print("held_out_apps: KHÔNG loại (train có val_ood → 5 app held-out vào train; chỉ còn val_in để kiểm)")
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["id"] in keep][: a.limit]
    rng = random.Random(a.shuffle_seed) if a.shuffle_seed >= 0 else None
    noise = None
    if a.asr_errors:
        from agent.clef.asr_noise import AsrNoise
        n_user = {r["id"]: sum(t["role"] == "user" for t in r["history"]) for r in rows}
        noise = AsrNoise(a.asr_errors, a.asr_subst, random.Random(max(a.shuffle_seed, 0) + 2), p_item=a.asr_p,
                         p_hist=a.asr_hist_p, n_user_turns=n_user)
        print(f"nhiễu ASR: {len(noise.full)} câu có bản ASR sẵn, bảng {len(noise.table)} cụm, p={a.asr_p}/{a.asr_hist_p}")
    def hist_ok(h):
        return not ({c["name"].split(".")[0] for c in h["label"]["tool_calls"]} & held)

    items = []
    for r in rows:
        items.append(r)
        if a.history:
            items += [h for h in history_items(r) if hist_ok(h)]
    orig = {r["id"]: r for r in rows}
    for kind, paths in (("phonetic", a.phonetic), ("synth", a.synth)):
        n_main, n_hist = 0, 0
        for path in paths:
            for l in open(path, encoding="utf-8"):
                p = json.loads(l)
                base = orig.get(p["id"].split("#")[0])
                if base is None:                    # id gốc không thuộc split này (hoặc bị --limit cắt)
                    continue
                items.append(p)
                n_main += 1
                if a.history:                       # lượt history của bản biến thể → mẫu phụ, bỏ nếu trùng y hệt bản gốc
                    sig = {h["id"].rsplit("#", 1)[1]: (h["audio_text"], h["history"], h["label"])
                           for h in history_items(base)}
                    for h in history_items(p):
                        if hist_ok(h) and sig.get(h["id"].rsplit("#", 1)[1]) != (h["audio_text"], h["history"], h["label"]):
                            items.append(h)
                            n_hist += 1
        if paths:
            print(f"{kind}: +{n_main} mẫu, +{n_hist} mẫu phụ từ history ({paths})")
    si, sn = map(int, a.shard.split("/"))
    if sn > 1:
        items = items[si::sn]
        print(f"shard {si}/{sn}: {len(items)} mẫu")
    pack = None
    if a.app_filter:
        assert a.sys == "compact", "catalog hai tầng chỉ có ở --sys compact"
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_filter"))
        from pack import AppFilterPack, app_of, tier1_for
        pack = AppFilterPack.load(a.app_filter, device=a.filter_device)
        print(f"bộ lọc: {pack.cfg} — catalog hai tầng, decoy {a.decoy_apps}")
        frng = random.Random(a.shuffle_seed + 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n, skip, kinds = 0, 0, {}
    t1_sizes, n_inst_sum, n_fullcat = [], 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            if noise is not None:
                it = noise.apply(it)
            t1 = None
            if pack is not None and frng.random() < a.full_catalog_p:
                n_fullcat += 1
            elif pack is not None:
                gold = {app_of(c["name"]) for c in it["label"]["tool_calls"]} | {
                    app_of(c["name"]) for h in it["history"] if h["role"] == "assistant" for c in (h.get("tool_calls") or [])}
                t1 = tier1_for(pack, it, gold_apps=gold, decoys=a.decoy_apps, rng=frng)
                t1_sizes.append(len(t1))
                n_inst_sum += len([x for x in it["context"]["installed_apps"] if x != "sys"])
            ex, s = examples(it, tok, catalog, labels, rng, t1)
            skip += s
            for e in ex:
                e["cat"] = "full" if t1 is None else "tier"
                for sp in e["spans"]:
                    kinds[sp["type"]] = kinds.get(sp["type"], 0) + 1
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                n += 1
    meta = {**F.fmt_opts(), "split": a.split_name, "history": a.history, "phonetic": a.phonetic, "synth": a.synth,
            "app_filter": a.app_filter, "decoy_apps": a.decoy_apps if a.app_filter else None,
            "tier1_avg": round(sum(t1_sizes) / len(t1_sizes), 2) if t1_sizes else None,
            "full_catalog_p": a.full_catalog_p if a.app_filter else None, "n_items_full_catalog": n_fullcat,
            "asr_noise": noise.summary() if noise else None, "asr_errors": a.asr_errors, "asr_subst": a.asr_subst,
            "n_items": len(items), "n_examples": n, "skipped": skip, "spans": kinds}
    json.dump(meta, open(a.out + ".meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.split_name}: {len(items)} mẫu → {n} ví dụ, bỏ {skip} bước | spans {kinds} | fmt {F.fmt_opts()}")


if __name__ == "__main__":
    main()
