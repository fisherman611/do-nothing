"""N1 (PLAN_v3_improve): mô phỏng offline "tìm rỗng → tìm lại với truy vấn nới" bằng ToolEnv, KHÔNG cần LLM.
Chạy lại tool_calls của một lần chạy clef_agent, tìm các call đọc trả danh sách rỗng, thử các cách nới và (nếu có nhãn)
kiểm tra kết quả nới có chứa giá trị mà call ghi của nhãn cần (id, tên...). Public chỉ để chẩn đoán, không chọn luật.

    python agent/sim_retry_empty.py --run agent/runs/clef/fmtA_48e_M2.val_in.jsonl --split train
    python agent/sim_retry_empty.py --run agent/runs/clef/public_fmtA_48e_e46f.jsonl --split public_test
"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def lists(res):
    return [v for v in res.values() if isinstance(v, list)] if isinstance(res, dict) and res.get("ok") else []


def is_empty(res):
    L = lists(res)
    return bool(L) and all(len(x) == 0 for x in L)


def relaxations(tool, args, spec):
    """Các cách nới (theo thứ tự thử). Tham số id không bao giờ bỏ (không do ASR nghe sai)."""
    P = spec.get("parameters", {}).get("properties", {})
    req = set(spec.get("parameters", {}).get("required", []))
    text = [k for k, v in args.items() if isinstance(v, str) and P.get(k, {}).get("type") == "string"
            and "enum" not in P.get(k, {}) and not k.endswith("id") and not k.endswith("_ids")]
    out = []
    opt = [k for k in text if k not in req]
    if opt:                                                  # (a) bỏ tham số chữ tuỳ chọn
        out.append(("drop_opt", {k: v for k, v in args.items() if k not in opt}))
    for k in text:                                           # (b) rút truy vấn chữ còn từ dài nhất / từng từ
        w = args[k].split()
        if len(w) >= 2:
            for t in sorted(w, key=len, reverse=True)[:3]:
                out.append((f"word:{k}", {**args, k: t}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["train", "public_test"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.data, a.split)); sys.path.insert(0, HERE)
    from zaic_tools import ToolEnv
    from agent import load_catalog
    cat = load_catalog(f"{a.data}/{a.split}/{a.split}_tools_catalog.json")
    gold = {}
    if a.split == "train":
        gold = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "runs", "clef", "gold_train.jsonl"), encoding="utf-8")}
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8")]
    stat, recs = collections.Counter(), []
    for r in rows:
        if not r.get("tool_calls"):
            continue
        env = ToolEnv(a.split, r["id"], root=a.data)
        g = gold.get(r["id"])
        need = set()
        if g:   # giá trị (chuỗi/số) trong call GHI của nhãn: mục cần tìm thấy
            for c in g.get("tool_calls", []):
                if cat.get(c["name"], {}).get("kind") != "read":
                    need |= {str(v) for v in (c.get("arguments") or {}).values() if isinstance(v, (str, int)) and str(v)}
        for i, c in enumerate(r["tool_calls"]):
            res = env.call(c["name"], c["arguments"])
            spec = cat.get(c["name"], {})
            if spec.get("kind") != "read" or not is_empty(res):
                continue
            stat["rỗng"] += 1
            last = i == len(r["tool_calls"]) - 1
            stat["rỗng_là_call_cuối"] += last
            rec = {"id": r["id"], "i": i, "tool": c["name"], "args": c["arguments"], "last": last, "tries": []}
            for kind, args2 in relaxations(c["name"], c["arguments"], spec):
                env2 = ToolEnv(a.split, r["id"], root=a.data)      # env riêng: không đụng state của lần chạy
                for c0 in r["tool_calls"][:i]:
                    env2.call(c0["name"], c0["arguments"])
                res2 = env2.call(c["name"], args2)
                n = sum(len(x) for x in lists(res2))
                hit = bool(need) and any(v in json.dumps(res2, ensure_ascii=False) for v in need)
                rec["tries"].append({"kind": kind, "args": args2, "n": n, "hit_gold": hit if g else None})
                if n:
                    break
            ok = [t for t in rec["tries"] if t["n"]]
            stat["nới_có_kết_quả"] += bool(ok)
            if g:
                stat["nới_chứa_mục_nhãn_cần"] += bool(ok and ok[0]["hit_gold"])
                stat["mẫu_có_nhãn_ghi"] += bool(need)
            recs.append(rec)
    by_tool = collections.Counter(x["tool"] for x in recs)
    by_tool_ok = collections.Counter(x["tool"] for x in recs if any(t["n"] for t in x["tries"]))
    print(dict(stat), "| mẫu có rỗng:", len({x["id"] for x in recs}), "/", len(rows))
    for t, n in by_tool.most_common():
        print(f"  {t:36} rỗng {n:3}  nới có kq {by_tool_ok[t]:3}")
    if a.out:
        json.dump(recs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
