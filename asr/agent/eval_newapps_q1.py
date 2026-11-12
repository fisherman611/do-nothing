"""M3b (PLAN_72h_v3): độ đúng bước Q1 (chọn tool) trên app LẠ — val_newapps / test_newapps từ build_newapps_q1.py.
Cùng một lượt gọi tính luôn N8 offline: Q1 với catalog hai tầng (text) và catalog đầy đủ (text_full); N8(τ) = nếu conf < τ và
hai lần lệch và lần đầy đủ có xác suất cao hơn thì lấy lần đầy đủ.

    python agent/eval_newapps_q1.py --url <vLLM> --file agent/clef/data/newapps_q1/val_newapps.jsonl --out <json>
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import agent.clef_agent as CA                    # noqa: E402
from agent.clef import fmt as F                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--model-name", default="llm")
    ap.add_argument("--file", action="append", required=True)
    ap.add_argument("--fmt-from", default=os.path.join(HERE, "clef/data/fmtA_48e_20k/train.jsonl.meta.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--debias", action="store_true", help="PriDe: thêm lượt hỏi với thứ tự phương án đảo ngược; báo độ đúng khi lấy trung bình và tỉ lệ chọn DỪNG theo vị trí")
    a = ap.parse_args()
    a.log_probs, a.ask_tau = True, None
    m = json.load(open(a.fmt_from, encoding="utf-8"))
    F.set_style_from(m.get("fmt", m))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CA.TOKENIZER[0], revision=CA.TOKENIZER[1])
    labels = F.make_labels(tok)
    rows = [json.loads(l) for p in a.file for l in open(p, encoding="utf-8")]

    def ask(text, sp):
        o = CA.VLLMOracle(a, tok, labels)
        ls = sp["labels"]
        text = text[:sp["pos"]]
        ids = [o.tid[l] for l in ls]
        d = o._req_json(text, 1, allowed_token_ids=ids, logprobs=len(ids), return_tokens_as_token_ids=True)
        p = CA.label_probs(d, ids)
        k = max(range(len(ls)), key=lambda i: p[i])
        return ls[k], p[k], CA.confidence(p)

    def probs(text, sp):
        o = CA.VLLMOracle(a, tok, labels)
        ids = [o.tid[l] for l in sp["labels"]]
        d = o._req_json(text, 1, allowed_token_ids=ids, logprobs=len(ids), return_tokens_as_token_ids=True)
        return CA.label_probs(d, ids)

    def opt_names(text, ls):
        k = text.rfind("[Q1]")
        return {l: next((x[len(l) + 2:].split(" – ")[0] for x in text[k:].split("\n") if x.startswith(f"{l}) ")), None) for l in ls}

    def one(r):
        sp, spf = r["spans"][0], r["spans_full"][0]
        l1, p1, c1 = ask(r["text"], sp)
        l2, p2, _ = ask(r["text_full"], spf)
        row = {"id": r["id"], "gold": sp["gold"], "gold_full": spf["gold"], "tier": r["cat"], "ok1": l1 == sp["gold"],
               "ok2": l2 == spf["gold"], "p1": p1, "p2": p2, "conf1": c1, "same": l1 == l2 and sp["labels"] == spf["labels"]}
        if a.debias:
            ls, t0 = sp["labels"], r["text"][:sp["pos"]]
            names = opt_names(t0, ls)
            rv = CA.reverse_q1(t0, ls)
            if rv is not None:
                pa, pb = probs(t0, sp), probs(rv[0], sp)
                q = [0.0] * len(ls)
                for i, j in enumerate(rv[1]):
                    q[j] = pb[i]
                avg = [(x + y) / 2 for x, y in zip(pa, q)]
                kb = max(range(len(ls)), key=lambda i: q[i]); ka = max(range(len(ls)), key=lambda i: avg[i])
                row.update(ok_rev=ls[kb] == sp["gold"], ok_avg=ls[ka] == sp["gold"],
                           stop_orig=names.get(l1) == F.STOP, stop_rev=names.get(ls[kb]) == F.STOP, stop_avg=names.get(ls[ka]) == F.STOP)
        return row

    with cf.ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(one, rows))
    n = len(out)
    res = {"n": n, "tier": sum(x["ok1"] for x in out) / n, "full": sum(x["ok2"] for x in out) / n, "n8": {}}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        ok = 0
        for x in out:
            use2 = x["conf1"] < tau and not x["ok1"] == x["ok2"] and x["p2"] > x["p1"]
            ok += x["ok2"] if use2 else x["ok1"]
        res["n8"][str(tau)] = ok / n
    res["low_conf"] = {"conf<0.7": sum(x["conf1"] < 0.7 for x in out),
                       "errors_in_low": sum(x["conf1"] < 0.7 and not x["ok1"] for x in out), "errors": sum(not x["ok1"] for x in out)}
    if a.debias:
        dd = [x for x in out if "ok_avg" in x]
        res["debias"] = {"n": len(dd), "acc_orig": sum(x["ok1"] for x in dd) / max(1, len(dd)),
                         "acc_rev": sum(x["ok_rev"] for x in dd) / max(1, len(dd)), "acc_avg": sum(x["ok_avg"] for x in dd) / max(1, len(dd)),
                         "stop_orig": sum(x["stop_orig"] for x in dd), "stop_rev": sum(x["stop_rev"] for x in dd), "stop_avg": sum(x["stop_avg"] for x in dd)}
    json.dump({"summary": res, "rows": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
