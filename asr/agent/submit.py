"""Gom output của clef_agent.py thành file nộp `submission.jsonl` (README dữ liệu, mục 4).

Mỗi dòng đúng 3 trường: id, tool_calls ([{"name", "arguments"}]), state_hash; đúng một dòng cho mỗi mẫu của split,
theo thứ tự trong <split>.jsonl. Dừng (không ghi file) nếu thiếu / thừa / trùng id hoặc có state_hash rỗng.
Không tạo lại ToolEnv để "vá" mẫu lỗi (mỗi mẫu chỉ một ToolEnv) — mẫu lỗi vẫn nộp đúng call + hash mà agent đã có.

    python agent/submit.py --run agent/runs/clef/public_fmtA_48e.jsonl --data data/train+public_test \
        --split public_test --out result/submission.jsonl
Sau đó kiểm schema bằng công cụ của BTC (chạy trong thư mục <data>/public_test):
    python -m zaic_tools --split public_test --root .. --validate <đường dẫn tuyệt đối submission.jsonl>
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="jsonl output của clef_agent.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = [json.loads(l)["id"] for l in open(f"{a.data}/{a.split}/{a.split}.jsonl", encoding="utf-8") if l.strip()]
    rows, dup = {}, []
    for l in open(a.run, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["id"] in rows:
            dup.append(r["id"])
        rows[r["id"]] = r
    missing = [i for i in ids if i not in rows]
    extra = [i for i in rows if i not in set(ids)]
    bad_hash = [i for i in ids if i in rows and not rows[i].get("state_hash")]
    bad_calls = [i for i in ids if i in rows and not isinstance(rows[i].get("tool_calls"), list)]
    errs = [i for i in ids if i in rows and rows[i].get("error")]
    print(f"split {a.split}: {len(ids)} mẫu | run: {len(rows)} dòng | thiếu {len(missing)} | thừa {len(extra)} | "
          f"trùng {len(dup)} | hash rỗng {len(bad_hash)} | tool_calls sai kiểu {len(bad_calls)} | mẫu có lỗi agent {len(errs)}")
    if errs:
        print("  mẫu lỗi (vẫn nộp call + hash đã có):", errs[:10])
    if missing or extra or dup or bad_hash or bad_calls:
        for name, lst in (("thiếu", missing), ("thừa", extra), ("trùng", dup), ("hash rỗng", bad_hash),
                          ("tool_calls sai", bad_calls)):
            if lst:
                print(f"  {name}: {lst[:10]}")
        sys.exit("KHÔNG ghi file nộp: sửa các lỗi trên trước")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in ids:
            r = rows[i]
            calls = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in r["tool_calls"]]
            f.write(json.dumps({"id": i, "tool_calls": calls, "state_hash": r["state_hash"]}, ensure_ascii=False) + "\n")
    n_calls = sum(len(rows[i]["tool_calls"]) for i in ids)
    n_empty = sum(not rows[i]["tool_calls"] for i in ids)
    print(f"→ {a.out}: {len(ids)} dòng, {n_calls} call, {n_empty} mẫu không gọi tool")


if __name__ == "__main__":
    main()
