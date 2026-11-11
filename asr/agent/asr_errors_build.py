"""A5b: dựng dữ liệu lỗi ASR (của model ASR ĐƯỢC CHỌN) để trộn vào train FC.

Vào: n-best của ASR trên 405 đoạn giữ riêng (`asr_v2_infer_nbest.json` khoá `val`: nbest[0] = beam top-1, điểm None;
các phương án sau = mẫu T=0.7 kèm tỉ lệ), manifest giữ riêng (uid → sid / lượt), splits FC.

Ra:
  asr_errors_<tag>.jsonl : mỗi uid một dòng {uid, sid, fc_split, turn ("final" | số thứ tự lượt user 1-based), ref,
                            hyps: [[câu ASR sai, trọng số], ...]} — chỉ hyps khác ref (sau chuẩn hoá nhẹ).
                            Dùng thay NGUYÊN CÂU khi build FC train (chỉ sid thuộc FC train).
  asr_subst_<tag>.json   : {"stats": tỉ lệ lỗi theo câu/từ (top-1), "table": {cụm từ chuẩn: [[cụm ASR, trọng số], ...]}}
                            bảng thay thế theo từ (1–3 từ), gom từ mọi uid (kể cả sid val) — dùng cho câu không có
                            bản ASR sẵn; cụm ASR rỗng = mất từ.

    python agent/asr_errors_build.py --nbest agent/runs/asr/asr_v2_infer_nbest.json \
        --manifest agent/runs/asr/asr_heldout.manifest.json --tag v2
"""
import argparse
import collections
import difflib
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = re.compile(r"[^\w\s]", re.U)


def words(s):
    s = unicodedata.normalize("NFC", s or "").lower()
    return PUNCT.sub(" ", s).split()


def norm(s):
    return " ".join(words(s))


def align(ref, hyp, max_len=3):
    """Cặp (cụm ref → cụm hyp) từ difflib trên từ; cụm ref 1..max_len từ; thừa từ (ref rỗng) bỏ."""
    r, h = words(ref), words(hyp)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if op == "equal" or i2 == i1:
            continue
        if i2 - i1 > max_len:
            continue
        out.append((" ".join(r[i1:i2]), " ".join(h[j1:j2])))
    return out, len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", default=os.path.join(HERE, "clef", "splits.json"))
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "train+public_test"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs", "asr"))
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--alt-weight", type=float, default=0.5, help="trọng số tổng của các mẫu T=0.7 so với top-1 (=1)")
    a = ap.parse_args()

    nb = json.load(open(a.nbest, encoding="utf-8"))["val"]
    man = {x["uid"]: x for x in json.load(open(a.manifest, encoding="utf-8"))["items"]}
    sp = json.load(open(a.splits, encoding="utf-8"))
    fc_split = {i: k for k in ("train", "val_in", "val_ood") for i in sp[k]}

    rows, table = [], collections.defaultdict(lambda: collections.defaultdict(float))
    n_utt, n_utt_err, n_w, n_w_err, n_alt = 0, 0, 0, 0, 0
    for uid, v in nb.items():
        m = man[uid]
        ref = v["ref"]
        n_utt += 1
        hyps, seen = [], {norm(ref)}
        top1 = v["nbest"][0][0]
        pairs, nw = align(ref, top1)
        n_w += nw
        n_w_err += sum(max(len(p[0].split()), len(p[1].split())) for p in pairs)
        if norm(top1) != norm(ref):
            n_utt_err += 1
        for k, (h, w) in enumerate(v["nbest"]):
            key = norm(h)
            if key in seen:
                continue
            seen.add(key)
            wt = 1.0 if k == 0 else a.alt_weight * (w or 0.0)
            hyps.append([h, round(wt, 4)])
            n_alt += k > 0
            for src, dst in align(ref, h)[0]:
                table[src][dst] += wt
        turn = "final" if m["kind"] == "final" else int(uid.rsplit("_u", 1)[1])
        rows.append({"uid": uid, "sid": m["sid"], "fc_split": fc_split.get(m["sid"], "?"), "turn": turn,
                     "ref": ref, "hyps": hyps})

    os.makedirs(a.out_dir, exist_ok=True)
    pe = os.path.join(a.out_dir, f"asr_errors_{a.tag}.jsonl")
    with open(pe, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tab = {src: sorted(((d, round(w, 4)) for d, w in dsts.items()), key=lambda x: -x[1]) for src, dsts in table.items()}
    stats = {"asr": a.tag, "utts": n_utt, "utt_err_rate": round(n_utt_err / n_utt, 4), "words": n_w,
             "word_err_rate_top1": round(n_w_err / n_w, 4),
             "err_words_per_bad_utt": round(n_w_err / max(1, n_utt_err), 2), "n_alt_hyps": n_alt,
             "table_entries": len(tab), "by_fc_split": dict(collections.Counter(r["fc_split"] for r in rows)),
             "with_err_by_fc_split": dict(collections.Counter(r["fc_split"] for r in rows if r["hyps"]))}
    ps = os.path.join(a.out_dir, f"asr_subst_{a.tag}.json")
    json.dump({"stats": stats, "table": tab}, open(ps, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    top = sorted(tab.items(), key=lambda kv: -sum(w for _, w in kv[1]))[:25]
    for src, dsts in top:
        print(f"  {src!r:28} → {dsts[:3]}")
    print("→", pe, ps)


if __name__ == "__main__":
    main()
