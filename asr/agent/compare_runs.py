"""So hai lần chạy agent trên cùng split KHÔNG có nhãn (public): chỉ chẩn đoán, không chọn kết quả theo từng mẫu.

Với mỗi mẫu khác state_hash giữa hai lần chạy: transcript câu cuối của mỗi bên, chuỗi call, tín hiệu không cần nhãn
(tool đọc trả rỗng / call không hợp lệ khi chạy lại trong ToolEnv chính thức, hỏi lại, không gọi tool, độ chắc Q1 đầu),
và loại khác biệt của transcript (giá trị số, cách viết số, tên riêng/từ nước ngoài, động từ hành động, khác vặt).

    python3 agent/compare_runs.py --a agent/runs/clef/public_fmtA_48e_e46f_n1.jsonl --asr-a agent/runs/asr/public.e46f.asr.flat.json \
        --b agent/runs/clef/public_fmtA_48e_L2_n1c3.jsonl --asr-b agent/runs/asr/public.L2.asr.flat.json \
        --out agent/runs/metrics/compare.e46f_vs_L2.public.json
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import agent as A                       # noqa: E402
from agent.clef.asr_noise_b0 import _nums          # noqa: E402

ASK = "sys.ask_missing_info"
ACTION = set("đặt gọi nhắn gửi mở tắt bật thêm xoá xóa huỷ hủy tạo sửa dời lưu mua trả chuyển nhắc ghi tìm coi xem đổi bán "
             "phát chơi dừng hẹn khoá khóa ghé lấy nộp nạp".split())
FILLER = set("à á ạ ơi nha nhé nhá nghe nghen hen hả ha đi với thì là mà ờ ừ ấy đó nhỉ nè luôn giùm dùm giúp em anh chị cô "
             "con tui tôi mình".split())


def words(s):
    return re.findall(r"\w+", unicodedata.normalize("NFC", (s or "").lower()))


def diff_kind(x, y):
    """Loại khác biệt giữa hai transcript (một nhãn, theo thứ tự ưu tiên)."""
    a, b = words(x), words(y)
    if a == b:
        return "giống"
    if _nums(x) != _nums(y):
        return "giá trị số"
    ops = [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op != "equal"]
    chg = [w for p, q in ops for w in p + q]
    if any(re.search(r"\d", w) for w in chg) and all(re.search(r"\d", w) or w in _NUMW for w in chg):
        return "cách viết số"
    raw_x, raw_y = x or "", y or ""
    caps = {w.lower() for w in re.findall(r"\b[A-ZĐ][\wÀ-ỹ]*", raw_x + " " + raw_y)}
    if any(w in caps or re.search(r"[a-z]", w) and not re.search(r"[à-ỹđ]", w) and len(w) > 2 and w not in FILLER for w in chg):
        return "tên riêng / từ nước ngoài"
    if any(w in ACTION for w in chg):
        return "động từ hành động"
    if all(w in FILLER for w in chg):
        return "từ đệm"
    return "từ thường khác"


_NUMW = set("không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi chục trăm nghìn ngàn triệu tỷ linh lẻ".split())


def replay(env_cls, split, sid, calls, catalog, root):
    """(số tool đọc trả rỗng, số call không hợp lệ) khi chạy lại chuỗi call trong ToolEnv."""
    env = env_cls(split, sid, root=root)
    empty = invalid = 0
    for c in calls:
        if not env.validate(c["name"], c["arguments"])["ok"]:
            invalid += 1
            continue
        r = env.call(c["name"], c["arguments"])
        if (catalog.get(c["name"]) or {}).get("kind") == "read" and isinstance(r, dict) and r.get("ok"):
            L = [v for v in r.values() if isinstance(v, list)]
            empty += bool(L) and not any(L)
    return empty, invalid


def q1_conf(run):
    e = next((x for x in run.get("probs") or [] if x.get("kind") == "q1"), None)
    return round(e["conf"], 3) if e and "conf" in e else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--asr-a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--asr-b", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.data, a.split))
    from zaic_tools import ToolEnv
    catalog = A.load_catalog(os.path.join(a.data, a.split, f"{a.split}_tools_catalog.json"))
    RA = {r["id"]: r for r in map(json.loads, open(a.a, encoding="utf-8"))}
    RB = {r["id"]: r for r in map(json.loads, open(a.b, encoding="utf-8"))}
    TA, TB = json.load(open(a.asr_a, encoding="utf-8")), json.load(open(a.asr_b, encoding="utf-8"))
    TA, TB = TA.get("hyps", TA), TB.get("hyps", TB)
    rows, cnt = [], collections.Counter()
    for i in RA:
        ra, rb = RA[i], RB[i]
        kind = diff_kind(TA.get(i), TB.get(i))
        cnt[f"câu cuối: {kind}"] += 1
        if ra["state_hash"] == rb["state_hash"]:
            cnt[f"state giống | {kind}"] += 1
            continue
        cnt[f"state KHÁC | {kind}"] += 1
        sig = {}
        for tag, r in (("a", ra), ("b", rb)):
            e, inv = replay(ToolEnv, a.split, i, r["tool_calls"], catalog, a.data)
            names = [c["name"] for c in r["tool_calls"]]
            sig[tag] = {"calls": names, "n_calls": len(names), "empty_reads": e, "invalid": inv, "ask": ASK in names,
                        "no_call": not names, "q1_conf": q1_conf(r)}
        rows.append({"id": i, "diff_kind": kind, "asr_a": TA.get(i), "asr_b": TB.get(i), "a": sig["a"], "b": sig["b"],
                     "calls_a": ra["tool_calls"], "calls_b": rb["tool_calls"]})
    # tổng hợp tín hiệu trên các mẫu khác state
    agg = collections.Counter()
    for r in rows:
        for t in ("a", "b"):
            s = r[t]
            agg[f"{t}: có tool đọc trả rỗng"] += s["empty_reads"] > 0
            agg[f"{t}: call không hợp lệ"] += s["invalid"] > 0
            agg[f"{t}: hỏi lại"] += s["ask"]
            agg[f"{t}: không gọi tool"] += s["no_call"]
            agg[f"{t}: tổng call"] += s["n_calls"]
        agg["cùng tool, khác tham số"] += r["a"]["calls"] == r["b"]["calls"]
        agg["khác chuỗi tool"] += r["a"]["calls"] != r["b"]["calls"]
    rep = {"counts": dict(cnt), "signals_on_diff": dict(agg), "n_diff": len(rows), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"mẫu khác state: {len(rows)}/{len(RA)}")
    for k in sorted(cnt):
        print(f"  {cnt[k]:4}  {k}")
    print("tín hiệu trên các mẫu khác state (a = lần chạy --a, b = --b):")
    for k in sorted(agg):
        print(f"  {agg[k]:4}  {k}")


if __name__ == "__main__":
    main()
