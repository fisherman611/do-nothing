"""A4 (1)(4)(5) + B6 bước 1: đổi N-best của e46f (Kaggle `zaic-e46f-nbest-{a,b}` → nbest.json) thành nguồn nhiễu FC.

Ra (agent/runs/asr/):
  e46f_tts.asr.json        {"model", "hyps": {id: 1-best}}            — cùng định dạng b0.base_medium.asr.json
  e46f_tts.manifest.json   [{"uid": id, "text": nhãn, "split": "train"}] — ref_manifests cho AsrNoiseB0
  asr_subst_e46f_tts.json  {"stats", "table": {cụm đúng: [[cụm nghe, trọng số], ...]}} — cùng định dạng asr_subst_v2.json
  e46f_tts.nbest.json      {id: [5 bản]}                               — "cách nghe khác" cho A4 (5)
In: WER 1-best theo nguồn (v2 / v3 / phương ngữ Trung, Nam) và B6 bước 1 trên 405 held-out (bản 2–5 bù được bao nhiêu).
    python agent/clef/nbest_to_noise.py <nbest_A.json> <nbest_B.json>
"""
import collections, difflib, json, os, re, sys, unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, f"{ROOT}/synth/tts_text")
from vi_num import to_spoken
OUT = f"{ROOT}/agent/runs/asr"


def norm(s):
    s = to_spoken(unicodedata.normalize("NFC", (s or "").lower()))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def ed(r, h):
    r, h = r.split(), h.split(); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1]


def subs(ref, hyp, max_n=3):
    """Cụm thay thế theo căn chỉnh từ (bỏ khác biệt chỉ do số↔chữ vì đã chuẩn hoá)."""
    r, h = ref.split(), hyp.split(); out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if op == "equal" or (i2 - i1) > max_n or (j2 - j1) > max_n + 1: continue
        if i2 > i1: out.append((" ".join(r[i1:i2]), " ".join(h[j1:j2])))
    return out


def main(paths):
    R = {}
    for p in paths: R.update(json.load(open(p, encoding="utf-8")))
    hyps, man, nb = {}, [], {}
    table = collections.defaultdict(collections.Counter); ref_cnt = collections.Counter()
    grp = collections.defaultdict(lambda: [0, 0, 0, 0])      # lỗi, từ, câu, câu đúng hết
    for k, x in R.items():
        best = x["nbest"][0]["text"]; ref = norm(x["label"]); h1 = norm(best)
        g = "heldout" if x["set"] == "heldout" else f"{x['set']}/{x['region']}"
        e = ed(ref, h1); s = grp[g]; s[0] += e; s[1] += len(ref.split()); s[2] += 1; s[3] += (e == 0)
        if x["set"] == "heldout": continue
        hyps[k] = best; man.append({"uid": k, "text": x["label"], "split": "train"})
        nb[k] = [b["text"] for b in x["nbest"]]
        for w in set(ref.split()): ref_cnt[w] += 1
        for a, b in subs(ref, h1): table[a][b] += 1
    tab = {a: [[b, round(c / max(1, ref_cnt.get(a.split()[0], 1)), 4)] for b, c in cs.most_common(6)]
           for a, cs in table.items() if sum(cs.values()) >= 2 or len(a.split()) == 1}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"model": "fisherman611/pw-tmp-e46f@85081b30", "use_prompt": False, "num_beams": 5, "hyps": hyps},
              open(f"{OUT}/e46f_tts.asr.json", "w"), ensure_ascii=False)
    json.dump(man, open(f"{OUT}/e46f_tts.manifest.json", "w"), ensure_ascii=False)
    json.dump(nb, open(f"{OUT}/e46f_tts.nbest.json", "w"), ensure_ascii=False)
    stats = {"asr": "e46f", "audio": "TTS v2/v3/ElevenLabs đã xử lý (augment_v3)", "utts": len(hyps), "table_entries": len(tab),
             "wer_by_group": {g: round(s[0] / max(1, s[1]), 4) for g, s in grp.items()}}
    json.dump({"stats": stats, "table": tab}, open(f"{OUT}/asr_subst_e46f_tts.json", "w"), ensure_ascii=False)
    print(f"{'nhóm':22s} {'câu':>6s} {'WER':>7s} {'đúng hết':>9s}")
    for g, s in sorted(grp.items()): print(f"{g:22s} {s[2]:6d} {s[0] / max(1, s[1]):7.2%} {s[3] / max(1, s[2]):9.1%}")
    print("bảng thay thế:", len(tab), "cụm; ví dụ:", [(a, tab[a][0]) for a in list(tab)[:8]])
    H = {k: x for k, x in R.items() if x["set"] == "heldout"}
    if H:
        wrong = [k for k, x in H.items() if ed(norm(x["label"]), norm(x["nbest"][0]["text"])) > 0]
        fixed = [k for k in wrong if any(ed(norm(H[k]["label"]), norm(b["text"])) == 0 for b in H[k]["nbest"][1:])]
        orc = sum(min(ed(norm(x["label"]), norm(b["text"])) for b in x["nbest"]) for x in H.values())
        W = sum(len(norm(x["label"]).split()) for x in H.values())
        fin = [k for k in wrong if H[k]["kind"] == "final"]
        print(f"B6 bước 1 (held-out {len(H)}): 1-best sai {len(wrong)} câu; có bản 2–5 đúng hết: {len(fixed)} "
              f"(câu cuối {sum(k in fixed for k in fin)}/{len(fin)}); WER oracle N-best {orc / W:.2%}")


if __name__ == "__main__":
    main(sys.argv[1:])
