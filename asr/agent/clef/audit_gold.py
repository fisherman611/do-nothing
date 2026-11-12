"""Kiểm toán GOLD nguồn của bộ train Clef: (G1) call gold có kết quả lỗi/không thành công; (G2) chạy lại gold của mẫu train gốc
trong ToolEnv chính thức (chỉ validate/call/get_state_hash — không mở *_environment/): validate ok, kết quả trùng bản lưu,
state_hash cuối trùng final_state; (G3) call gold lặp y hệt trong cùng lượt; (G4) nhãn rỗng (DỪNG ngay).

    python3 agent/clef/audit_gold.py --dir agent/clef/data/fmtA_v3 --out agent/runs/audit/fmtA_v3.gold.json
"""
import argparse, collections, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agent.clef.audit_v3 import load_sources, source_of, DUP      # noqa: E402


def failed(res):
    return not isinstance(res, dict) or res.get("ok") is False or res.get("done") is False or "error" in res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    synth = [os.path.join(ROOT, p) for p in ("synth/train_alias.jsonl", "synth/train_iface.jsonl", "synth_phonetic/train_phonetic.jsonl")]
    src, cache = load_sources(a.data, synth), {}
    ids = []
    for l in open(f"{a.dir}/train.jsonl", encoding="utf-8"):
        e = json.loads(l)
        if "#na" not in e["id"] and (not ids or ids[-1] != e["id"]):
            ids.append(e["id"])
    ids = list(dict.fromkeys(ids))
    cnt, rows = collections.Counter(), collections.defaultdict(list)
    for eid in ids:
        it = source_of(eid, src, cache)
        calls, results = it["label"]["tool_calls"], it["label"]["tool_results"]
        cnt["items"] += 1
        cnt["calls"] += len(calls)
        if len(calls) != len(results):
            rows["G0_calls_results_len"].append({"id": eid})
        if not calls:
            cnt["G4_empty_label"] += 1
        sig = collections.Counter(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)
        if any(v > 1 for v in sig.values()):
            rows["G3_repeated_call"].append({"id": eid, "call": [k[:120] for k, v in sig.items() if v > 1]})
        for i, (c, r) in enumerate(zip(calls, results)):
            if failed(r):
                rows["G1_failed_gold_call"].append({"id": eid, "step": i, "n_calls": len(calls), "call": c, "result": r})
    # ---- G2: chạy lại gold của mẫu train GỐC có trong bộ
    sys.path.insert(0, os.path.join(a.data, "train"))
    from zaic_tools import ToolEnv
    base = [i for i in ids if "#" not in i]
    for eid in base:
        it = src[eid]
        env = ToolEnv("train", eid, root=a.data)
        cnt["G2_replayed_items"] += 1
        for i, (c, r) in enumerate(zip(it["label"]["tool_calls"], it["label"]["tool_results"])):
            v = env.validate(c["name"], c["arguments"])
            if not v["ok"]:
                rows["G2_validate_fail"].append({"id": eid, "step": i, "call": c, "error": v.get("error")})
                continue
            got = env.call(c["name"], c["arguments"])
            cnt["G2_replayed_calls"] += 1
            if json.dumps(got, sort_keys=True, ensure_ascii=False) != json.dumps(r, sort_keys=True, ensure_ascii=False):
                rows["G2_result_mismatch"].append({"id": eid, "step": i, "call": c, "stored": json.dumps(r, ensure_ascii=False)[:200], "env": json.dumps(got, ensure_ascii=False)[:200]})
        fs = it["label"].get("final_state")
        try:
            h = env.get_state_hash()
            want = fs.get("state_hash") if isinstance(fs, dict) else fs
            if want is not None and isinstance(want, str) and h != want:
                rows["G2_state_hash_mismatch"].append({"id": eid, "env": h, "gold": want})
            elif want is None or not isinstance(want, str):
                cnt["G2_no_gold_hash"] += 1
        except Exception as ex:
            rows["G2_hash_error"].append({"id": eid, "err": str(ex)[:100]})
    summ = {**cnt, **{k: len(v) for k, v in rows.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    for k, v in rows.items():
        print(f"\n## {k} ({len(v)})")
        for x in v[:8]:
            print("  ", json.dumps(x, ensure_ascii=False)[:330])


if __name__ == "__main__":
    main()
