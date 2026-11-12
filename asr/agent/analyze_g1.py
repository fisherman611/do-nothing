"""Phân tích G1 (I5b) + C3 theo từng mẫu trên val: so báo cáo điểm của lần chạy mới với lần tham chiếu, tách theo việc luật có
kích hoạt hay không và theo xác suất của tool ghi ở Q1 thu hẹp → chọn ngưỡng τ (không dùng public).

    python3 agent/analyze_g1.py --run agent/runs/clef/fmtA_48e_g1_M1.val_in.jsonl \
        --report agent/runs/clef/reports/fmtA_48e_g1_M1.val_in.json --ref agent/runs/clef/reports/fmtA_48e_notau.val_in.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    run = {r["id"]: r for r in map(json.loads, open(a.run, encoding="utf-8"))}
    new = {s["id"]: s for s in json.load(open(a.report, encoding="utf-8"))["samples"]}
    ref = {s["id"]: s for s in json.load(open(a.ref, encoding="utf-8"))["samples"]}
    ids = [i for i in new if i in ref]
    tot_new, tot_ref = sum(new[i]["score"] for i in ids), sum(ref[i]["score"] for i in ids)
    print(f"mẫu {len(ids)} | điểm mới {100 * tot_new / len(ids):.2f} | tham chiếu {100 * tot_ref / len(ids):.2f} | chênh {100 * (tot_new - tot_ref) / len(ids):+.2f}")
    for flag in ("n_i5b_probe", "n_i5b", "n_c3", "n_retry"):
        hit = [i for i in ids if run[i].get(flag)]
        d = sum(new[i]["score"] - ref[i]["score"] for i in hit)
        print(f"  {flag}: {len(hit)} mẫu, tổng chênh điểm {d:+.2f} (lên {sum(new[i]['score'] > ref[i]['score'] for i in hit)}, xuống {sum(new[i]['score'] < ref[i]['score'] for i in hit)})")
    other = [i for i in ids if not any(run[i].get(f) for f in ("n_i5b", "n_c3")) and abs(new[i]["score"] - ref[i]["score"]) > 1e-9]
    print(f"  mẫu đổi điểm mà KHÔNG có i5b/c3 kích hoạt: {len(other)} {[(i, round(ref[i]['score'], 2), round(new[i]['score'], 2)) for i in other[:8]]}")
    print("\nCác lần hỏi lại I5b (xếp theo xác suất tool ghi):")
    rows = []
    for i in ids:
        for e in run[i].get("probs") or []:
            if e.get("kind") == "q1_i5b":
                p = e["p"]
                best = max((k for k in p if k not in ("DỪNG", "sys.ask_missing_info")), key=lambda k: p[k], default=None)
                rows.append((p.get(best, 0) if best else 0, i, best, bool(run[i].get("n_i5b")), ref[i]["score"], new[i]["score"],
                             [c["name"] for c in run[i]["tool_calls"]][:3]))
    for pw, i, best, act, s0, s1, calls in sorted(rows, reverse=True):
        print(f"  p={pw:.2f} {'KÍCH HOẠT' if act else 'bỏ qua  '} {i} {best}  điểm {s0:.2f} → {s1:.2f}  calls={calls}")
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        act = [(s1 - s0) for pw, i, best, ac, s0, s1, _ in rows if ac and pw >= tau]
        print(f"  τ={tau}: kích hoạt {len(act)}, chênh {sum(act):+.2f}, làm xấu {sum(x < 0 for x in act)}")


if __name__ == "__main__":
    main()
