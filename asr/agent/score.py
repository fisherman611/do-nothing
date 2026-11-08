"""Score agent outputs on train against the labels: state match (state_hash) + call F1.

Gold state: replay label.tool_calls in a separate ToolEnv (scoring on train only; inference uses one env per sample).
Call matching (approximation of the official scorer, exact rules not published):
  same tool name; arguments equal after zaic_tools canonicalisation (lower-case, collapsed spaces, '700 cm'=='700cm',
  lists of scalars as sets); `x-match: fuzzy` arguments match when token-F1 >= 0.5 (diacritics folded).
Per-sample F1 = 2PR/(P+R) on a one-to-one matching; both empty -> 1.

    python score.py --data <dir> --split train runs/qwen3_4b_2507.jsonl [runs/other.jsonl ...]
"""
import argparse
import collections
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--fuzzy-th", type=float, default=0.5)
    ap.add_argument("--details", help="ghi chi tiết từng mẫu (jsonl) cho file đầu tiên")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.data, args.split))
    from zaic_tools import ToolEnv
    from zaic_tools.utils import _canon_scalar, _tokens

    cat = json.load(open(os.path.join(args.data, args.split, f"{args.split}_tools_catalog.json"), encoding="utf-8"))
    fuzzy = {t["name"]: {k for k, v in t.get("parameters", {}).get("properties", {}).items() if v.get("x-match") == "fuzzy"}
             for a in cat["apps"] for t in a["tools"]}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(args.data, args.split, f"{args.split}.jsonl"), encoding="utf-8")}

    cache_p = os.path.join(os.path.dirname(os.path.abspath(args.runs[0])), f"gold_hash_{args.split}.json")
    gold = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    for i, it in items.items():
        if i not in gold:
            env = ToolEnv(args.split, i, root=args.data)
            for c in it["label"]["tool_calls"]:
                env.call(c["name"], c["arguments"])
            gold[i] = env.get_state_hash()
    json.dump(gold, open(cache_p, "w"))

    def canon(v):
        if isinstance(v, dict):
            return {k: canon(x) for k, x in v.items()}
        if isinstance(v, list):
            out = [canon(x) for x in v]
            if all(not isinstance(x, (dict, list)) for x in out):
                return sorted(out, key=lambda x: (type(x).__name__, str(x)))
            return out
        return _canon_scalar(v)

    def tok_f1(a, b):
        ta, tb = _tokens(a), _tokens(b)
        if not ta and not tb:
            return 1.0
        inter = len(ta & tb)
        return 0.0 if inter == 0 else 2 * inter / (len(ta) + len(tb))

    def same(p, g):
        if p["name"] != g["name"]:
            return False
        pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
        if set(pa) != set(ga):
            return False
        fz = fuzzy.get(g["name"], set())
        for k in ga:
            if k in fz and isinstance(pa[k], str) and isinstance(ga[k], str):
                if tok_f1(pa[k], ga[k]) < args.fuzzy_th:
                    return False
            elif canon(pa[k]) != canon(ga[k]):
                return False
        return True

    def f1(pred, gold_calls):
        if not pred and not gold_calls:
            return 1.0, 0
        used, tp = set(), 0
        for p in pred:
            for j, g in enumerate(gold_calls):
                if j not in used and same(p, g):
                    used.add(j); tp += 1
                    break
        if tp == 0:
            return 0.0, 0
        pr, rc = tp / len(pred), tp / len(gold_calls)
        return 2 * pr * rc / (pr + rc), tp

    def groups(it):
        g = it["label"]["tool_calls"]
        names = [c["name"] for c in g]
        out = ["all"]
        out.append("0 call" if not g else ("1 call" if len(g) == 1 else ("2 calls" if len(g) == 2 else ">=3 calls")))
        if "sys.ask_missing_info" in names:
            out.append("ask_missing_info")
        if any(n.startswith("weather.") for n in names) or "mưa" in it.get("audio_text", ""):
            out.append("điều kiện thời tiết")
        out.append("có history" if it["history"] else "không history")
        return out

    print(f"{'run':28s} {'n':>5s} {'state':>6s} {'F1':>6s} {'exact':>6s} {'#pred':>6s} {'#gold':>6s} {'inval':>6s} {'err':>4s} {'lat':>5s} {'p95tok':>6s}")
    for ri, path in enumerate(args.runs):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        agg = collections.defaultdict(lambda: collections.defaultdict(float))
        det = []
        toks = []
        for r in rows:
            it = items[r["id"]]
            g = it["label"]["tool_calls"]
            fv, tp = f1(r["tool_calls"], g)
            st = float(r["state_hash"] == gold[r["id"]])
            ex = float(fv == 1.0 and len(r["tool_calls"]) == len(g))
            toks += [t for t in r.get("prompt_tokens", []) if t]
            for grp in groups(it):
                a = agg[grp]
                a["n"] += 1; a["state"] += st; a["f1"] += fv; a["exact"] += ex
                a["pred"] += len(r["tool_calls"]); a["gold"] += len(g); a["inval"] += r.get("n_invalid", 0)
                a["err"] += r.get("error") is not None; a["lat"] += r.get("latency", 0)
            det.append({"id": r["id"], "f1": round(fv, 3), "state": st, "text": it.get("audio_text"),
                        "gold": g, "pred": r["tool_calls"], "raw_last": (r.get("raw") or [""])[-1][:400], "error": r.get("error")})
        name = os.path.basename(path).replace(".jsonl", "")
        toks.sort()
        for grp in ["all", "0 call", "1 call", "2 calls", ">=3 calls", "ask_missing_info", "điều kiện thời tiết", "có history", "không history"]:
            a = agg.get(grp)
            if not a:
                continue
            n = a["n"]
            label = name if grp == "all" else f"  · {grp}"
            p95 = toks[int(0.95 * len(toks))] if (grp == "all" and toks) else 0
            print(f"{label:28s} {int(n):5d} {a['state']/n:6.3f} {a['f1']/n:6.3f} {a['exact']/n:6.3f} {a['pred']/n:6.2f} "
                  f"{a['gold']/n:6.2f} {a['inval']/n:6.2f} {int(a['err']):4d} {a['lat']/n:5.1f} {p95:6d}")
        if args.details and ri == 0:
            with open(args.details, "w", encoding="utf-8") as f:
                for d in sorted(det, key=lambda x: x["f1"]):
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
