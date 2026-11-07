#!/usr/bin/env python3
"""Phân tích lỗi ASR từ prediction đã lưu (chỉ đọc train.jsonl + data/results, không đọc *_environment/).

Chạy:  python3 analysis/asr_errors.py [--results data/results/asr] [--train data/train+public_test/train/train.jsonl] [--out analysis/asr_errors]
Mỗi model (thư mục <run>/<model_slug>/<phase>/predictions.jsonl) được gộp theo id từ mọi lượt chạy, chỉ lấy status ok.
"""
import argparse, collections, csv, glob, json, os, re, statistics as st, unicodedata

# ---------------------------------------------------------------- chuẩn hóa
DIG = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5,
       "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
MULT = {"nghìn": 10**3, "ngàn": 10**3, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
NUMW = set(DIG) | set(MULT) | {"mười", "mươi", "trăm", "linh", "lẻ"}
TIME = set("giờ phút giây kém rưỡi sáng trưa chiều tối khuya đêm mai hôm qua kia nay tuần tháng ngày lúc".split())
NEG = set("không chưa đừng chẳng chả khỏi chớ".split())
PART = set("nha nhé nhỉ nghen hen hén hả à ạ ơi nè đi giùm giúp dùm với nữa luôn thôi chút cái những các là thì mà ừ ờ vậy hỉ hông heng".split())
VERBS = set("chuyển gọi nhắn đặt hủy huỷ xóa xoá tắt bật mở tìm thêm báo nhắc ghi sửa đổi dời gửi đọc xem kiểm tra mua đặt bấm phát trả thanh toán cập nhật lưu hẹn hạ tăng giảm khóa khoá bỏ chạy".split())
DIALECT = set("rứa chừ hỉ nớ răng hén hông hổng dô zô ổng bả ảnh dzậy vầy tui nghen heng mô tê ni".split())
HALLU = re.compile(r"subscribe|đăng ký kênh|đăng kí kênh|ghiền mì|cảm ơn (các )?bạn đã|hẹn gặp lại|chào mừng quý vị|like và share|bấm chuông", re.I)


def nfc(s): return unicodedata.normalize("NFC", s)


def strip_dia(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c))


def ref_tokens(orig):
    """[(token_lower, is_name)] — tên riêng = viết hoa, không đứng đầu câu."""
    orig = nfc(orig)
    out, last_end, start = [], 0, True
    for m in re.finditer(r"\w+", orig):
        gap = orig[last_end:m.start()]
        if re.search(r"[.!?]", gap): start = True
        w = m.group()
        out.append((w.lower(), (not start) and w[0].isupper() and not w.isdigit()))
        start = False
        last_end = m.end()
    return out


def is_phone(t): return t.startswith("#0") and len(t) >= 7


def parse_run(run):
    """run: token số (từ tiếng Việt hoặc '#N') -> list '#giá_trị'."""
    if all(t in DIG for t in run) and len(run) >= 3:
        return ["#" + "".join(str(DIG[t]) for t in run)]
    res, total, cur, pend, seen = [], 0, 0, None, False

    def flush():
        nonlocal total, cur, pend, seen
        if seen:
            res.append("#" + str(total + cur + (pend or 0)))
        total, cur, pend, seen = 0, 0, None, False

    for t in run:
        if is_phone(t):
            flush(); res.append(t)
        elif t.startswith("#"):
            if pend is not None: flush()
            pend, seen = int(t[1:]), True
        elif t in DIG:
            if t in ("mốt", "lăm") and pend is not None and 2 <= pend <= 9:
                pend = pend * 10 + DIG[t]; continue
            if pend is not None: flush()
            pend, seen = DIG[t], True
        elif t == "mười": cur += 10; seen = True
        elif t == "mươi": cur += (pend or 1) * 10; pend = None; seen = True
        elif t == "trăm": cur += (pend or 1) * 100; pend = None; seen = True
        elif t in MULT:
            cur += pend or 0; pend = None
            total += (cur or 1) * MULT[t]; cur = 0; seen = True
        # linh/lẻ: bỏ qua
    flush()
    return res


_TO = {"ò": "\u0300", "ó": "\u0301", "ỏ": "\u0309", "õ": "\u0303", "ọ": "\u0323"}
_TU = {"ù": "\u0300", "ú": "\u0301", "ủ": "\u0309", "ũ": "\u0303", "ụ": "\u0323"}


def canon(w):
    """Đưa vị trí dấu thanh về một kiểu (hòa = hoà, hủy = huỷ) để không tính là lỗi nghe."""
    m = re.fullmatch(r"([^aeiouyáàảãạăâéèẻẽẹêíìỉĩịóòỏõọôơúùủũụưýỳỷỹỵ]*)([òóỏõọ])([ae])", w)
    if m: return unicodedata.normalize("NFC", m.group(1) + "o" + m.group(3) + _TO[m.group(2)])
    m = re.fullmatch(r"([^aeiouyqáàảãạăâéèẻẽẹêíìỉĩịóòỏõọôơúùủũụưýỳỷỹỵ]*)([ùúủũụ])y", w)
    if m: return unicodedata.normalize("NFC", m.group(1) + "u" + "y" + _TU[m.group(2)])
    return w


def normalize_tokens(toks, flags):
    toks = [canon(t) for t in toks]
    # 1) "phần trăm" -> 1 token (không phải số 100)
    t2, f2, i = [], [], 0
    while i < len(toks):
        if toks[i] == "phần" and i + 1 < len(toks) and toks[i + 1] == "trăm":
            t2.append("phần_trăm"); f2.append(False); i += 2
        else:
            t2.append(toks[i]); f2.append(flags[i]); i += 1
    # 2) chữ số -> '#N'
    t3 = []
    for t in t2:
        if t.isdigit():
            t3.append("#" + (t if (t.startswith("0") and len(t) >= 6) else str(int(t))))
        else:
            t3.append(t)
    # 3) cụm số
    words, fl, i = [], [], 0
    isnum = lambda t: t in NUMW or t.startswith("#")
    while i < len(t3):
        if isnum(t3[i]):
            j = i
            while j < len(t3) and isnum(t3[j]): j += 1
            run = t3[i:j]
            if run == ["không"]:
                words.append("không"); fl.append(False)
            else:
                for n in parse_run(run): words.append(n); fl.append(False)
            i = j
        else:
            words.append(t3[i]); fl.append(f2[i]); i += 1
    return words, fl


def norm_ref(orig):
    toks = ref_tokens(orig)
    return normalize_tokens([t for t, _ in toks], [n for _, n in toks])


def norm_pred(text):
    s = nfc(text).lower()
    s = re.sub(r"(\d)h(\d{0,2})", lambda m: f"{m.group(1)} giờ {m.group(2)} ", s)
    s = re.sub(r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000), s)
    s = s.replace("%", " phần trăm ")
    s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)[.,](?=\d)", "_", s)
    s = re.sub(r"[^\w\s]", " ", s)
    toks = s.split()
    merged, i = [], 0
    while i < len(toks):  # gộp nhóm chữ số liền kề dạng số điện thoại
        if toks[i].isdigit() and toks[i].startswith("0") and len(toks[i]) <= 4:
            j, buf = i, ""
            while j < len(toks) and toks[j].isdigit() and len(buf) + len(toks[j]) <= 11:
                buf += toks[j]; j += 1
            if len(buf) >= 9: merged.append(buf); i = j; continue
        merged.append(toks[i]); i += 1
    return normalize_tokens(merged, [False] * len(merged))[0]


# ---------------------------------------------------------------- căn chỉnh
def align(r, h):
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1: ops.append(("D", i - 1, None)); i -= 1
        else: ops.append(("I", None, j - 1)); j -= 1
    return D[n][m], ops[::-1]


def loan(w):
    return bool(re.fullmatch(r"[a-z]+", w)) and (bool(re.search(r"[fjwz]", w)) or w[-1] in "bdfklrsvxz" or bool(re.search(r"ck|ll|ss|tt|ee|oo", w)))


def category(w, is_name):
    if w.startswith("#"): return "số (số lượng, tiền, SĐT…)"
    if is_name: return "tên riêng"
    if w in TIME: return "từ chỉ thời gian"
    if w in NEG: return "phủ định"
    if loan(w): return "từ mượn / tiếng Anh"
    if w in VERBS: return "động từ lệnh"
    if w in PART: return "trợ từ / từ đệm"
    return "từ nội dung khác"


# ---------------------------------------------------------------- tải dữ liệu
def load_models(results_dir):
    models = collections.defaultdict(dict)  # slug -> id -> {text, audio_seconds, src}
    refs = {}
    for pf in sorted(glob.glob(os.path.join(results_dir, "**", "predictions.jsonl"), recursive=True)):
        parts = pf.split(os.sep)
        slug = parts[-3]
        rf = os.path.join(os.path.dirname(pf), "references.jsonl")
        if os.path.exists(rf):
            for l in open(rf, encoding="utf8"):
                d = json.loads(l); refs[d["id"]] = d.get("audio_text") or d.get("text")
        for l in open(pf, encoding="utf8"):
            d = json.loads(l)
            if d.get("status", "ok") != "ok": continue
            models[slug][d["id"]] = {"text": d.get("text") or "", "audio": d.get("audio_seconds"), "src": pf}
    return models, refs


def analyze(slug, preds, refs, labels):
    rows, cat_ref, cat_err, pair, tone_pairs = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    tot = collections.Counter()
    num_ref = num_hit = name_ref = name_full = name_part = 0
    ins_num = ins_other = 0
    for i, p in preds.items():
        if i not in refs: continue
        rw, rf = norm_ref(refs[i]); hw = norm_pred(p["text"])
        e, ops = align(rw, hw)
        tot["err"] += e; tot["ref"] += len(rw)
        # WER không phân biệt dấu thanh
        rs, hs = [strip_dia(w) for w in rw], [strip_dia(w) for w in hw]
        tot["err_nodia"] += align(rs, hs)[0]
        # WER thô: không quy đổi số (chỉ lowercase + bỏ dấu câu)
        rraw = [t for t, _ in ref_tokens(refs[i])]
        hraw = re.sub(r"[^\w\s]", " ", nfc(p["text"]).lower()).split()
        tot["err_raw"] += align(rraw, hraw)[0]; tot["ref_raw"] += len(rraw)
        s = d_ = ins = 0
        for op, a, b in ops:
            if op == "ok": cat_ref[category(rw[a], rf[a])] += 1
            elif op == "S":
                s += 1; c = category(rw[a], rf[a]); cat_ref[c] += 1; cat_err[c] += 1
                pair[(rw[a], hw[b])] += 1
                if strip_dia(rw[a]) == strip_dia(hw[b]): tot["S_tone"] += 1; tone_pairs[(rw[a], hw[b])] += 1
            elif op == "D":
                d_ += 1; c = category(rw[a], rf[a]); cat_ref[c] += 1; cat_err[c] += 1
            else:
                ins += 1
                if hw[b].startswith("#"): ins_num += 1
                else: ins_other += 1
        tot["S"] += s; tot["D"] += d_; tot["I"] += ins
        # thực thể
        nums = [w for w in rw if w.startswith("#")]
        hc = collections.Counter(w for w in hw if w.startswith("#"))
        for n in nums:
            num_ref += 1
            if hc[n] > 0: hc[n] -= 1; num_hit += 1
        j, hj = 0, " ".join(hw)
        while j < len(rw):
            if rf[j]:
                k = j
                while k < len(rw) and rf[k]: k += 1
                phrase = rw[j:k]; name_ref += 1
                if " ".join(phrase) in hj: name_full += 1
                if sum(w in hw for w in phrase) >= max(1, (len(phrase) + 1) // 2): name_part += 1
                j = k
            else: j += 1
        wer = e / max(len(rw), 1)
        flags = []
        if HALLU.search(p["text"]): flags.append("ảo giác kiểu video/YouTube")
        if len(hw) < 0.5 * len(rw) and wer > 0.5: flags.append("cắt cụt / thiếu nội dung")
        if len(hr_ := re.findall(r"\w+", nfc(p["text"]))) > 1.6 * len(rw) and wer > 0.5: flags.append("thừa nội dung")
        hr = re.findall(r"\w+", nfc(p["text"]).lower())
        tri = collections.Counter(zip(hr, hr[1:], hr[2:]))
        if tri and max(tri.values()) >= 3: flags.append("lặp / vòng lặp sinh token")
        if not hw: flags.append("rỗng")
        if wer >= 0.8 and not flags: flags.append("sai gần hết (không rõ nguyên nhân)")
        dia = bool(set(rw) & DIALECT)
        lab = labels.get(i) or {}
        calls = [c["name"] for c in (lab.get("tool_calls") or [])]
        dom = calls[0].split(".")[0] if calls else "(không gọi tool)"
        rows.append(dict(id=i, wer=wer, ref=refs[i], pred=p["text"], audio=p["audio"] or 0, n_ref=len(rw), errors=e,
                         flags=flags, dialect=dia, domain=dom, first_tool=calls[0] if calls else "",
                         has_num=any(w.startswith("#") for w in rw), has_name=any(rf),
                         multi=len(calls) >= 2, ask=("sys.ask_missing_info" in calls)))
    return dict(rows=rows, tot=tot, cat_ref=cat_ref, cat_err=cat_err, pair=pair, tone_pairs=tone_pairs,
                num=(num_ref, num_hit), name=(name_ref, name_full, name_part), ins=(ins_num, ins_other))


def pct(a, b): return f"{100 * a / b:.1f}%" if b else "n/a"


def grp(rows, key, minn=15):
    g = collections.defaultdict(list)
    for r in rows: g[key(r)].append(r)
    out = []
    for k, v in g.items():
        if len(v) < minn: continue
        out.append((k, len(v), sum(x["errors"] for x in v) / max(sum(x["n_ref"] for x in v), 1)))
    return sorted(out, key=lambda x: -x[2])


def report(slug, a, f):
    rows, tot = a["rows"], a["tot"]
    n = len(rows); wer = tot["err"] / tot["ref"]
    w = [r["wer"] for r in rows]
    f.write(f"\n## {slug}\n\n")
    f.write(f"**{n} mẫu** (status ok, đã có nhãn). Chuẩn hóa số/giờ ở cả hai phía trước khi chấm.\n\n")
    f.write("| Chỉ số | Giá trị |\n| --- | ---: |\n")
    f.write(f"| WER thô (chỉ lowercase + bỏ dấu câu) | {100 * tot['err_raw'] / tot['ref_raw']:.2f}% |\n")
    f.write(f"| WER sau quy đổi số/giờ | {100 * wer:.2f}% |\n")
    f.write(f"| WER bỏ dấu thanh/dấu mũ (chỉ xét âm tiết) | {100 * tot['err_nodia'] / tot['ref']:.2f}% |\n")
    f.write(f"| Thay thế / xoá / chèn (từ) | {tot['S']} / {tot['D']} / {tot['I']} |\n")
    f.write(f"| Trong thay thế: chỉ sai dấu thanh/dấu mũ | {tot['S_tone']} ({pct(tot['S_tone'], tot['S'])}) |\n")
    f.write(f"| Câu đúng hoàn toàn | {sum(x == 0 for x in w)} ({pct(sum(x == 0 for x in w), n)}) |\n")
    f.write(f"| Câu WER ≥ 0,3 / ≥ 0,5 / ≥ 0,8 | {pct(sum(x >= .3 for x in w), n)} / {pct(sum(x >= .5 for x in w), n)} / {pct(sum(x >= .8 for x in w), n)} |\n")
    f.write(f"| Trung vị WER theo câu | {st.median(w):.3f} |\n\n")
    d = 100 * (tot['err_raw'] / tot['ref_raw'] - wer)
    f.write(f"- Chênh WER thô và WER sau quy đổi số/giờ: **{d:.1f} điểm %** là lỗi do *cách viết* (số chữ vs chữ số), không phải nghe sai.\n")
    f.write(f"- Chênh WER sau quy đổi và WER bỏ dấu: **{100 * (wer - tot['err_nodia'] / tot['ref']):.1f} điểm %** là lỗi chỉ ở dấu thanh/dấu mũ (đúng âm tiết nền).\n\n")

    f.write("### Lỗi theo loại từ tham chiếu\n\n")
    f.write("Tỷ lệ lỗi = (thay thế + xoá) / số từ loại đó trong nhãn. Chèn thừa không tính vào đây.\n\n")
    f.write("| Loại từ | Số từ nhãn | Sai | Tỷ lệ lỗi | Đóng góp vào tổng lỗi (S+D) |\n| --- | ---: | ---: | ---: | ---: |\n")
    sd = sum(a["cat_err"].values())
    for c, k in sorted(a["cat_ref"].items(), key=lambda x: -a["cat_err"][x[0]]):
        f.write(f"| {c} | {k} | {a['cat_err'][c]} | {pct(a['cat_err'][c], k)} | {pct(a['cat_err'][c], sd)} |\n")
    f.write(f"\nChèn thừa: {tot['I']} từ, trong đó {a['ins'][0]} là token số (sau quy đổi).\n\n")

    nr, nh = a["num"]; r_, full, part = a["name"]
    f.write("### Thực thể quan trọng cho tool call\n\n")
    f.write(f"- Số (sau quy đổi): giữ đúng **{pct(nh, nr)}** ({nh}/{nr}).\n")
    f.write(f"- Tên riêng (cụm viết hoa): khớp nguyên cụm **{pct(full, r_)}** ({full}/{r_}); khớp ít nhất nửa số từ {pct(part, r_)}.\n\n")

    f.write("### Nhóm lỗi cấp câu\n\n| Nhóm | Số câu | Tỷ lệ |\n| --- | ---: | ---: |\n")
    fc = collections.Counter(x for r in rows for x in r["flags"])
    for k, v in fc.most_common(): f.write(f"| {k} | {v} | {pct(v, n)} |\n")
    if not fc: f.write("| (không có) | 0 | 0% |\n")
    f.write("\n")

    f.write("### WER theo nhóm (WER corpus của nhóm; nhóm < 15 mẫu bị bỏ)\n\n| Nhóm | Số mẫu | WER |\n| --- | ---: | ---: |\n")
    def addg(title, items):
        for k, c, v in items: f.write(f"| {title}: {k} | {c} | {100 * v:.1f}% |\n")
    addg("độ dài audio", grp(rows, lambda r: "<4s" if r["audio"] < 4 else "4-6s" if r["audio"] < 6 else "6-9s" if r["audio"] < 9 else "9-13s" if r["audio"] < 13 else ">=13s"))
    addg("có số", grp(rows, lambda r: "có số" if r["has_num"] else "không số"))
    addg("có tên riêng", grp(rows, lambda r: "có tên riêng" if r["has_name"] else "không tên riêng"))
    addg("phương ngữ/khẩu ngữ (heuristic)", grp(rows, lambda r: "có từ phương ngữ" if r["dialect"] else "không"))
    addg("chuỗi nhiều tool", grp(rows, lambda r: "≥2 tool call" if r["multi"] else "≤1 tool call"))
    addg("hỏi thiếu thông tin", grp(rows, lambda r: "có ask_missing_info" if r["ask"] else "không"))
    f.write("\n**Theo ứng dụng của tool call đầu tiên (10 nhóm tệ nhất, ≥15 mẫu):**\n\n| App | Số mẫu | WER |\n| --- | ---: | ---: |\n")
    for k, c, v in grp(rows, lambda r: r["domain"])[:10]: f.write(f"| {k} | {c} | {100 * v:.1f}% |\n")
    f.write("\n**Tốt nhất (5 nhóm):**\n\n| App | Số mẫu | WER |\n| --- | ---: | ---: |\n")
    for k, c, v in grp(rows, lambda r: r["domain"])[-5:]: f.write(f"| {k} | {c} | {100 * v:.1f}% |\n")

    f.write("\n### Cặp nhầm hay gặp (nhãn → model), top 25\n\n| Nhãn | Model | Lần |\n| --- | --- | ---: |\n")
    for (x, y), k in a["pair"].most_common(25): f.write(f"| {x} | {y} | {k} |\n")
    f.write("\n### Cặp chỉ sai dấu thanh/dấu mũ, top 15\n\n| Nhãn | Model | Lần |\n| --- | --- | ---: |\n")
    for (x, y), k in a["tone_pairs"].most_common(15): f.write(f"| {x} | {y} | {k} |\n")

    f.write("\n### 20 câu tệ nhất\n\n")
    for r in sorted(rows, key=lambda r: (-r["wer"], r["id"]))[:20]:
        f.write(f"- `{r['id']}` WER {r['wer']:.2f} [{', '.join(r['flags']) or '—'}] — tool: `{r['first_tool'] or '—'}`\n  - nhãn: {r['ref']}\n  - model: {r['pred']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results/asr")
    ap.add_argument("--train", default="data/train+public_test/train/train.jsonl")
    ap.add_argument("--out", default="analysis/asr_errors")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = {}
    for l in open(a.train, encoding="utf8"):
        d = json.loads(l); labels[d["id"]] = d.get("label")
    models, refs = load_models(a.results)
    # nhãn từ train.jsonl là nguồn chuẩn
    for l in open(a.train, encoding="utf8"):
        d = json.loads(l); refs[d["id"]] = d["audio_text"]
    summ = {}
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf8") as f:
        f.write("# Phân tích lỗi ASR\n\nSinh bởi `analysis/asr_errors.py` từ prediction đã lưu trong `data/results/asr/`. "
                "Chỉ đọc `train.jsonl` (transcript + tên tool), không đọc `*_environment/`.\n")
        f.write("\nQuy đổi trước khi chấm (cả hai phía): bỏ khác biệt vị trí dấu thanh kiểu cũ/mới (hòa = hoà, hủy = huỷ);  số/giờ tiếng Việt → giá trị số ở nhãn; chữ số, `50k`, `9h35`, `420.000`, `50 nghìn`, SĐT cách nhóm → giá trị số ở prediction. "
                "Loại từ (tên riêng, thời gian, phủ định, từ mượn, động từ lệnh) là heuristic theo danh sách từ, không phải gán nhãn thủ công.\n")
        for slug, preds in sorted(models.items()):
            res = analyze(slug, preds, refs, labels)
            if not res["rows"]: continue
            report(slug, res, f)
            t = res["tot"]
            summ[slug] = dict(n=len(res["rows"]), wer_norm=t["err"] / t["ref"], wer_raw=t["err_raw"] / t["ref_raw"],
                              wer_nodia=t["err_nodia"] / t["ref"], S=t["S"], D=t["D"], I=t["I"], S_tone=t["S_tone"],
                              number_recall=res["num"], name_recall=res["name"],
                              cat_err=dict(res["cat_err"]), cat_ref=dict(res["cat_ref"]))
            with open(os.path.join(a.out, f"{slug}_samples.csv"), "w", newline="", encoding="utf8") as c:
                wr = csv.writer(c)
                wr.writerow(["id", "wer", "audio_s", "first_tool", "flags", "dialect", "ref", "pred"])
                for r in sorted(res["rows"], key=lambda r: (-r["wer"], r["id"])):
                    wr.writerow([r["id"], f"{r['wer']:.3f}", f"{r['audio']:.2f}", r["first_tool"], ";".join(r["flags"]), int(r["dialect"]), r["ref"], r["pred"]])
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf8"), ensure_ascii=False, indent=2)
    print("wrote", a.out, {k: v["n"] for k, v in summ.items()})


if __name__ == "__main__":
    main()
