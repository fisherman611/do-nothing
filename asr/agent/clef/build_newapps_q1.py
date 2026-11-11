"""A1 (PLAN_72h_v3 §1e): ví dụ chỉ có BƯỚC Q1 (chọn tool) cho 28 app tổng hợp `synth/newapps`.

`newapps` chỉ có câu nói + tên tool đúng + danh sách app cài (không có tham số, không có môi trường tool), nên mỗi ví dụ
dừng ngay sau đáp án Q1 (loss chỉ ở nhãn chọn tool). Cùng định dạng dòng với build_data.py (id, step, text, spans, ...),
cat = "tier" | "full"; id có hậu tố `#na` để make_subset/phân tích nhận ra.

    python agent/clef/build_newapps_q1.py --data data/train+public_test --out agent/clef/data/newapps_q1_train.jsonl \
        --split train --repeat 3 --app-filter <pack_b> --full-catalog-p 0.4
Chỉ nhận split train | val_newapps | test_newapps. Khi --split train: bỏ khỏi `installed` mọi app thuộc val/test của newapps
(app lạ dành làm thước đo M3b), bỏ câu chéo app (thứ tự call không rõ).
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "synth", "newapps"))
from agent.clef import fmt as F                       # noqa: E402
from agent.clef.build_data import TOKENIZER           # noqa: E402
from agent.clef.core import GoldOracle, SkipStep, run_step  # noqa: E402


class EndQ1(Exception):
    pass


class Q1Oracle(GoldOracle):
    """Đáp án Q1 từ tên tool vàng; mọi câu hỏi sau Q1 → dừng. q1_text = chuỗi đến hết đáp án Q1."""

    def __init__(self, tool):
        super().__init__({"name": tool, "arguments": {}})
        self.q1_text = None

    def choose(self, text, labels, kind, info):
        if kind != "q1":
            raise EndQ1
        lab = super().choose(text, labels, kind, info)
        self.q1_text = text + " " + lab + "\n"
        return lab

    def choose_name(self, text, names):
        raise EndQ1

    def gen(self, text, k, s):
        raise EndQ1

    def gen_tail(self, text, params):
        raise EndQ1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val_newapps", "test_newapps"])
    ap.add_argument("--repeat", type=int, default=3, help="số biến thể (xáo phương án, ngữ cảnh, tầng) cho mỗi câu; val/test nên dùng 1")
    ap.add_argument("--app-filter")
    ap.add_argument("--full-catalog-p", type=float, default=0.4)
    ap.add_argument("--decoy-apps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    F.set_style("v1", q2="all", q1_desc="short", sys="compact", rules="short")      # = fmtA_48e

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    train_apps = {x["id"]: [t["name"] for t in x["tools"]] for x in json.load(
        open(f"{a.data}/train/train_tools_catalog.json", encoding="utf-8"))["apps"]}
    na = json.load(open(os.path.join(ROOT, "synth/newapps/newapps_catalog.json"), encoding="utf-8"))["apps"]
    na_tools = {x["id"]: [t for t in x["tools"]] for x in na}
    import apps_a, apps_b, apps_c, apps_d
    app_split = {x["id"]: x["split"] for x in apps_a.APPS + apps_b.APPS + apps_c.APPS + apps_d.APPS}
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "synth/newapps/newapps_utts.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["split"] == a.split and r["kind"] in ("single", "neg")][: a.limit]
    donors = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
    rng = random.Random(a.seed)
    pack = None
    if a.app_filter:
        sys.path.insert(0, os.path.join(HERE, "app_filter"))
        from pack import AppFilterPack, app_of, tier1_for
        pack = AppFilterPack.load(a.app_filter, device="cpu")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n, skipped, n_full, dropped_apps = 0, 0, 0, 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            gold = r["gold_tools"][0]
            for k in range(a.repeat):
                inst, overlay = {}, []
                for app in r["installed"]:
                    if a.split == "train" and app in app_split and app_split[app] != "train":
                        dropped_apps += 1
                        continue
                    if app in train_apps:
                        inst[app] = {"tools": train_apps[app]}
                    elif app in na_tools:
                        inst[app] = {"tools": [t["name"] for t in na_tools[app]]}
                        overlay += na_tools[app]
                if r["gold_apps"] and r["gold_apps"][0] not in inst and gold.rpartition(".")[0] not in inst:
                    skipped += 1
                    continue
                d = rng.choice(donors)
                ctx = {kk: d["context"][kk] for kk in ("current_time", "timezone", "location", "user_profile")}
                item = {"id": f"{r['id']}#na{k}", "audio_text": r["text"], "history": [],
                        "context": {**ctx, "installed_apps": inst},
                        "label": {"tool_calls": [{"name": gold, "arguments": {}}], "tool_results": []},
                        "catalog_overlay": overlay}
                cat = dict(catalog)
                for t in overlay:
                    cat[t["name"]] = t
                msgs = F.build_messages(item)
                tools = F.A.tool_specs(item, cat)
                if gold not in {t["function"]["name"] for t in tools}:
                    skipped += 1
                    continue
                t1 = None
                if pack is not None and a.split != "train":      # đo (M3b): tầng 1 do bộ lọc quyết định như lúc thi, không chèn gold/decoy
                    t1 = tier1_for(pack, item)
                elif pack is not None and rng.random() >= a.full_catalog_p:
                    t1 = tier1_for(pack, item, gold_apps={gold.rpartition(".")[0]}, decoys=a.decoy_apps, rng=rng)
                else:
                    n_full += 1
                prefix = F.render_prefix(tok, msgs, tools, t1)
                def q1_of(pfx):
                    oo = Q1Oracle(gold)
                    try:
                        run_step(pfx, tools, cat, [], labels, oo, tok, rng if a.split == "train" else None,
                                 hint={"request": r["text"], "done": []})      # đo: không xáo phương án (như lúc thi)
                    except (EndQ1, SkipStep):
                        pass
                    return oo
                o = q1_of(prefix)
                if o.q1_text is None:
                    skipped += 1
                    continue
                row = {"id": item["id"], "step": 0, "prefix_chars": len(prefix), "text": o.q1_text,
                       "spans": o.spans, "args": {}, "name": gold, "cat": "full" if t1 is None else "tier", "q1_only": True}
                if a.split != "train":          # cho N8: cùng câu hỏi với catalog đầy đủ (không hai tầng)
                    of = q1_of(F.render_prefix(tok, msgs, tools, None))
                    row.update(text_full=of.q1_text, spans_full=of.spans)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.split}: {len(rows)} câu × {a.repeat} → {n} ví dụ (full-catalog {n_full}), bỏ {skipped}, "
          f"gỡ {dropped_apps} lượt app val/test khỏi installed → {a.out}")


if __name__ == "__main__":
    main()
