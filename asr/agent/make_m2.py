"""M2 (PLAN_v3_improve §0.1): transcript có lỗi ASR giống điều kiện thi, bằng cách "pha loãng" lỗi của PhoWhisper GỐC.

PhoWhisper gốc chưa nghe audio train → lỗi là nhầm lẫn âm học thật, nhưng quá dày (WER ~26%) và viết thường, không dấu câu.
Với mỗi đoạn: căn câu chuẩn và câu của model gốc theo từ (đã chuẩn hoá); đoạn khớp giữ nguyên TOKEN GỐC của câu chuẩn
(giữ hoa/thường, dấu câu); mỗi cụm lệch lấy bản của model gốc với xác suất q, không thì trả về câu chuẩn. Seed theo uid.

    python agent/make_m2.py --manifest agent/runs/asr/asr_heldout.manifest.json --base <heldout.base_medium.asr.json> \
        --q 0.45 --out <heldout.M2.asr.json>
Ra: {"hyps": {uid: câu}, "q": q, ...} (đọc được bằng agent/asr_score.py) và <out>.flat.json {uid: câu} cho clef_agent --asr.
"""
import argparse
import difflib
import hashlib
import json
import random
import re
import unicodedata

PUNCT = re.compile(r"[^\w]", re.U)


def ntok(t):
    return PUNCT.sub("", unicodedata.normalize("NFC", t).lower())


def gold_tokens(text):
    """[(raw, norm)]: token chỉ có dấu câu được gắn vào token trước để không mất khi ghép lại."""
    out = []
    for raw in (text or "").split():
        n = ntok(raw)
        if n:
            out.append([raw, n])
        elif out:
            out[-1][0] += " " + raw
    return out


def thin(gold, base, q, rng):
    G = gold_tokens(gold)
    B = [t for t in (ntok(x) for x in (base or "").split()) if t]
    sm = difflib.SequenceMatcher(a=[n for _, n in G], b=B, autojunk=False)
    out, n_spans, n_kept = [], 0, 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += [r for r, _ in G[i1:i2]]
            continue
        n_spans += 1
        if rng.random() < q:
            n_kept += 1
            out += B[j1:j2]
        else:
            out += [r for r, _ in G[i1:i2]]
    s = " ".join(out).strip()
    return (s[:1].upper() + s[1:]) if s else s, n_spans, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest có trường ref (câu chuẩn) theo uid")
    ap.add_argument("--base", required=True, help="transcript của PhoWhisper gốc ({'hyps': {uid: câu}})")
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--seed", default="m2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.load(open(a.manifest, encoding="utf-8"))
    items = m["items"] if isinstance(m, dict) else m
    base = json.load(open(a.base, encoding="utf-8"))["hyps"]
    hyps, spans, kept = {}, 0, 0
    for it in items:
        rng = random.Random(int(hashlib.md5(f"{a.seed}:{it['uid']}".encode()).hexdigest(), 16))
        hyps[it["uid"]], s, k = thin(it["ref"], base.get(it["uid"], ""), a.q, rng)
        spans, kept = spans + s, kept + k
    json.dump({"model": f"thinned({a.base}) q={a.q}", "q": a.q, "hyps": hyps}, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False)
    flat = a.out[:-5] + ".flat.json" if a.out.endswith(".json") else a.out + ".flat.json"
    json.dump(hyps, open(flat, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(hyps)} đoạn | cụm lệch {spans}, giữ lỗi {kept} ({100 * kept / max(1, spans):.0f}%) → {a.out}, {flat}")


if __name__ == "__main__":
    main()
