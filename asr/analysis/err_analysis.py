"""Phân tích lỗi ASR trên val: WER thường vs WER sau quy đổi số/chính tả, và lỗi theo thực thể."""
import json, re, sys, unicodedata, collections
import jiwer

UNITS = "không một hai ba bốn năm sáu bảy tám chín".split()

def n2w(n: int) -> str:
    if n < 10: return UNITS[n]
    if n < 20: return "mười " + ("lăm" if n % 10 == 5 else UNITS[n % 10]) if n % 10 else "mười"
    if n < 100:
        t, u = divmod(n, 10)
        s = UNITS[t] + " mươi"
        if u == 1: s += " mốt"
        elif u == 5: s += " lăm"
        elif u: s += " " + UNITS[u]
        return s
    if n < 1000:
        h, r = divmod(n, 100)
        s = UNITS[h] + " trăm"
        if r == 0: return s
        if r < 10: return s + " linh " + UNITS[r]
        return s + " " + n2w(r)
    if n < 1_000_000:
        k, r = divmod(n, 1000)
        s = n2w(k) + " nghìn"
        if r == 0: return s
        if r < 100: return s + " không trăm " + (("linh " + UNITS[r]) if r < 10 else n2w(r))
        return s + " " + n2w(r)
    m, r = divmod(n, 1_000_000)
    s = n2w(m) + " triệu"
    return s if r == 0 else s + " " + n2w(r)

def digits2words(s: str) -> str:
    s = re.sub(r"(\d+)[h:](\d+)", lambda m: f"{m.group(1)} giờ {m.group(2)}", s)
    s = re.sub(r"(\d+)h\b", r"\1 giờ", s)
    def rep(m):
        d = m.group(0)
        if len(d) >= 7 or d.startswith("0"):           # số điện thoại / mã: đọc từng chữ số
            return " ".join(UNITS[int(c)] for c in d)
        return n2w(int(d))
    return re.sub(r"\d+", rep, s)

# chính tả tương đương: vị trí dấu thanh (hoá/hóa), ngàn/nghìn, lẻ/linh, tư/bốn...
TONE = {"òa":"oà","óa":"oá","ỏa":"oả","õa":"oã","ọa":"oạ","òe":"oè","óe":"oé","ỏe":"oẻ","õe":"oẽ","ọe":"oẹ",
        "ùy":"uỳ","úy":"uý","ủy":"uỷ","ũy":"uỹ","ụy":"uỵ"}
SYN = {"ngàn":"nghìn","lẻ":"linh","tư":"bốn","mươi lăm":"mươi lăm","dùm":"giùm"}

def basic(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def full(s):
    s = basic(digits2words(s))
    for a, b in TONE.items(): s = s.replace(a, b)
    toks = [SYN.get(t, t) for t in s.split()]
    return " ".join(toks)

def name_tokens(ref_raw):
    """từ viết hoa không ở đầu câu → coi là tên riêng / thực thể"""
    out = set()
    for sent in re.split(r"[.!?]\s+", unicodedata.normalize("NFC", ref_raw)):
        ws = sent.split()
        for w in ws[1:]:
            w2 = re.sub(r"[^\w]", "", w)
            if w2 and w2[0].isupper() and not w2.isupper() or (w2.isupper() and len(w2) > 1):
                out.add(w2.lower())
    return out

NUMWORDS = set(UNITS) | {"mười","mươi","mốt","lăm","trăm","nghìn","triệu","linh","rưỡi","giờ","phút","kém","nửa"}

def main(path):
    pairs = json.load(open(path))["pairs"]
    refs_b = [basic(r) for _, r, _ in pairs]; hyps_b = [basic(h) for _, _, h in pairs]
    refs_f = [full(r) for _, r, _ in pairs]; hyps_f = [full(h) for _, _, h in pairs]
    print(f"utts: {len(pairs)}")
    print(f"WER thường      : {jiwer.wer(refs_b, hyps_b):.4f}  CER {jiwer.cer(refs_b, hyps_b):.4f}")
    print(f"WER sau quy đổi : {jiwer.wer(refs_f, hyps_f):.4f}  CER {jiwer.cer(refs_f, hyps_f):.4f}")
    exact_b = sum(r == h for r, h in zip(refs_b, hyps_b)); exact_f = sum(r == h for r, h in zip(refs_f, hyps_f))
    print(f"khớp hoàn toàn: thường {exact_b}, sau quy đổi {exact_f}, chỉ lệch kiểu viết: {exact_f - exact_b}")

    cat = collections.Counter(); name_tot = name_err = num_tot = num_err = 0
    examples = collections.defaultdict(list)
    for (i, r_raw, h_raw), r, h in zip(pairs, refs_f, hyps_f):
        if r == h: continue
        names = name_tokens(r_raw)
        out = jiwer.process_words(r, h)
        rt, ht = r.split(), h.split()
        bad_ref = set(); ins = []
        for ch in out.alignments[0]:
            if ch.type == "equal": continue
            bad_ref.update(range(ch.ref_start_idx, ch.ref_end_idx))
            if ch.type == "insert": ins += ht[ch.hyp_start_idx:ch.hyp_end_idx]
        kinds = set()
        for j, t in enumerate(rt):
            isname = t in names; isnum = t in NUMWORDS
            name_tot += isname; num_tot += isnum
            if j in bad_ref:
                if isname: name_err += 1; kinds.add("tên/thực thể")
                elif isnum: num_err += 1; kinds.add("số/giờ")
                else: kinds.add("từ thường")
        if not kinds: kinds.add("chèn thêm từ")
        key = " + ".join(sorted(kinds))
        cat[key] += 1
        if len(examples[key]) < 4: examples[key].append((r_raw, h_raw))
    print("\nphân loại câu còn sai sau quy đổi:")
    for k, v in cat.most_common(): print(f"  {v:3d}  {k}")
    print(f"\ntoken tên/thực thể: {name_tot}, sai {name_err} ({name_err/max(name_tot,1):.1%})")
    print(f"token số/giờ      : {num_tot}, sai {num_err} ({num_err/max(num_tot,1):.1%})")
    for k, ex in examples.items():
        print(f"\n== {k}")
        for r, h in ex: print("  R:", r); print("  H:", h)

if __name__ == "__main__":
    main(sys.argv[1])
