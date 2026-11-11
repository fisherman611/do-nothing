"""M5 (PLAN_v3_improve §0): chẩn đoán FC trên public test bằng nhãn CÓ SẴN trong đầu vào — CHỈ ĐỂ HIỂU, không tune/train.

Mỗi lượt user trong history của public (lời nói = transcript ASR của mình) đi kèm lượt assistant với `tool_calls`. Dựng lại
đúng ngữ cảnh lúc đó (history cắt tới trước lượt), hỏi model câu Q1 bước đầu (chọn tool / hỏi lại / DỪNG) và so với call
đầu tiên của assistant. Không chạy tool nào (không đụng ToolEnv), không đọc *_environment/.

    python agent/diag_public_q1.py --url <vLLM> --fmt-from <meta.json> --app-filter <pack_b> \
        --asr agent/runs/asr/public.v2_medium.asr.json --out agent/runs/clef/diag_public_q1.jsonl
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import agent.clef_agent as CA                   # noqa: E402
from agent.clef import fmt as F                 # noqa: E402

NEW_APPS = {"ecommerce", "event_tickets", "grocery", "habit_tracker", "pantry", "parking", "public_transit", "recipes",
            "stocks", "train_bus_tickets", "translator"}


def pairs(row):
    H = row["history"]
    for j, t in enumerate(H):
        if t["role"] == "user" and j + 1 < len(H) and H[j + 1]["role"] == "assistant":
            calls = H[j + 1].get("tool_calls") or []
            yield j, os.path.basename(t["audio"])[:-4], (calls[0]["name"] if calls else F.STOP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--model-name", default="llm")
    ap.add_argument("--fmt-from", required=True)
    ap.add_argument("--app-filter")
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.split, a.log_probs, a.ask_tau = "public_test", True, None
    m = json.load(open(a.fmt_from, encoding="utf-8"))
    F.set_style_from(m.get("fmt", m))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CA.TOKENIZER[0], revision=CA.TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = CA.A.load_catalog(f"{a.data}/public_test/public_test_tools_catalog.json")
    asr = json.load(open(a.asr, encoding="utf-8"))
    rows = [json.loads(l) for l in open(f"{a.data}/public_test/public_test.jsonl", encoding="utf-8")]
    jobs = [(r, j, uid, gold) for r in rows for j, uid, gold in pairs(r)][: a.limit]
    pack = None
    if a.app_filter:
        pack, lock = CA.get_pack(a)
        from pack import tier1_for           # noqa: E402  (get_pack đã thêm app_filter vào sys.path)

    def one(job):
        r, j, uid, gold = job
        item = {**r, "id": uid, "history": r["history"][:j], "audio": r["history"][j]["audio"]}
        msgs = F.build_messages(item, asr)
        tools = CA.A.tool_specs(item, catalog)
        t1 = None
        if pack is not None:
            with lock:
                t1 = tier1_for(pack, item, asr)
        prefix = F.render_prefix(tok, msgs, tools, t1)
        q1, mp = F.q1_block(tools, labels, None, F.n_tokens(tok, prefix) > F.SHORT_Q1_AT,
                            {"request": asr.get(uid, ""), "done": []})
        o = CA.VLLMOracle(a, tok, labels)
        lab = o.choose(prefix + q1, list(mp), "q1", mp)
        pred = mp[lab]
        p = o.plog[-1] if o.plog else {}
        return {"sid": r["id"], "uid": uid, "turn": j, "gold": gold, "pred": pred, "ok": pred == gold,
                "gold_app": gold.split(".")[0], "new": gold.split(".")[0] in NEW_APPS,
                "gold_in_tier1": (t1 is None) or gold.split(".")[0] in t1 or gold in (F.STOP, F.ASK),
                "conf": p.get("conf"), "p_gold": (p.get("p") or {}).get(gold), "asr": asr.get(uid, "")}

    out = []
    with cf.ThreadPoolExecutor(a.workers) as ex, open(a.out, "w", encoding="utf-8") as f:
        for k, fu in enumerate(cf.as_completed([ex.submit(one, x) for x in jobs]), 1):
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                print("lỗi:", type(e).__name__, e, flush=True)
                continue
            out.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    def acc(L):
        return f"{100 * sum(x['ok'] for x in L) / len(L):.1f}% ({sum(x['ok'] for x in L)}/{len(L)})" if L else "-"
    g = collections.defaultdict(list)
    for x in out:
        g["DỪNG" if x["gold"] == F.STOP else ("hỏi lại" if x["gold"] == F.ASK else ("app mới" if x["new"] else "app cũ"))].append(x)
    print(f"Q1 bước đầu, đúng tool như assistant: tổng {acc(out)}")
    for k in ("app cũ", "app mới", "hỏi lại", "DỪNG"):
        print(f"  {k:8} {acc(g[k])}")
    conf = collections.Counter()
    for x in g["app mới"]:
        if not x["ok"]:
            conf[(x["gold"], x["pred"])] += 1
    print("Nhầm hay gặp ở app mới (gold → dự đoán):")
    for (gd, pr), n in conf.most_common(15):
        print(f"  {n:3}  {gd} → {pr}")
    by_app = collections.defaultdict(list)
    for x in out:
        if x["gold"] not in (F.STOP, F.ASK):
            by_app[x["gold_app"]].append(x)
    print("Theo app (≥5 lượt):")
    for ap_, L in sorted(by_app.items(), key=lambda kv: sum(y["ok"] for y in kv[1]) / len(kv[1])):
        if len(L) >= 5:
            print(f"  {'*' if ap_ in NEW_APPS else ' '} {ap_:20} {acc(L)}")


if __name__ == "__main__":
    main()
