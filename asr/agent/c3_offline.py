"""Kiểm C3 (geo_fix) OFFLINE trên log một lần chạy public: chạy lại các tool call đã ghi trong ToolEnv chính thức (không mở
*_environment/), tìm call đọc có tham số nơi chốn trả rỗng, đề xuất call sửa và chạy thử xem có còn rỗng không.
Chỉ xem dự đoán + kết quả tool; KHÔNG dùng nhãn. In bảng + ghi JSON.

    python3 agent/c3_offline.py --run agent/runs/clef/public_fmtA_48e_e46f_n1.jsonl --asr agent/runs/asr/public.e46f.asr.json \
        --out agent/runs/c3/c3_offline.public_n1.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import agent as A          # noqa: E402
from agent import geo_fix as G        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--asr", help="JSON {hyps: {uid: text}} — câu cuối để tìm tên nơi khi mã IATA bịa")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.data, a.split))
    from zaic_tools import ToolEnv
    catalog = A.load_catalog(os.path.join(a.data, a.split, f"{a.split}_tools_catalog.json"))
    items = {x["id"]: x for x in map(json.loads, open(os.path.join(a.data, a.split, f"{a.split}.jsonl"), encoding="utf-8"))}
    asr = json.load(open(a.asr, encoding="utf-8")) if a.asr else {}
    asr = asr.get("hyps", asr)
    rows, n_place_calls, n_empty = [], 0, 0
    for line in open(a.run, encoding="utf-8"):
        r = json.loads(line)
        item = items[r["id"]]
        utter = asr.get(r["id"]) or item.get("audio_text", "")
        env = ToolEnv(a.split, r["id"], root=a.data)
        tried = []
        for call in r["tool_calls"]:
            pre = G.precheck_iata(call, item, utter) if call["name"] == "flights.search" else None
            if pre is not None:
                res_pre = env.call(pre["name"], pre["arguments"]) if env.validate(pre["name"], pre["arguments"])["ok"] else {"ok": False}
                rows.append({"id": r["id"], "kind": "iata_precheck", "call": call, "fixed": pre, "fixed_nonempty": not G.is_empty(res_pre) and bool(res_pre.get("ok")), "utter": utter})
                continue
            if not env.validate(call["name"], call["arguments"])["ok"]:
                continue
            res = env.call(call["name"], call["arguments"])
            if call["name"] in G.PLACE_TOOLS:
                n_place_calls += 1
                if G.is_empty(res):
                    n_empty += 1
                    fx = G.fix_empty(call, res, item, utter, catalog, tried=tried + [call["arguments"]])
                    row = {"id": r["id"], "kind": "empty", "call": call, "fixed": fx, "utter": utter}
                    if fx is not None and env.validate(fx["name"], fx["arguments"])["ok"]:
                        res2 = env.call(fx["name"], fx["arguments"])
                        row["fixed_nonempty"] = not G.is_empty(res2) and bool(res2.get("ok"))
                        tried.append(fx["arguments"])
                    rows.append(row)
            tried.append(call["arguments"])
    summ = {"samples": len(items), "place_calls": n_place_calls, "place_calls_empty": n_empty,
            "proposed": sum(1 for x in rows if x.get("fixed")), "fixed_nonempty": sum(1 for x in rows if x.get("fixed_nonempty")),
            "iata_precheck": sum(1 for x in rows if x["kind"] == "iata_precheck")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    for x in rows:
        fx = x.get("fixed")
        print(f"{x['id']} [{x['kind']}] {x['call']['name']} {json.dumps(x['call']['arguments'], ensure_ascii=False)}")
        print(f"    → {json.dumps(fx['arguments'], ensure_ascii=False) if fx else 'không sửa'}"
              + (f"  | có kết quả: {x['fixed_nonempty']}" if "fixed_nonempty" in x else ""))


if __name__ == "__main__":
    main()
