"""Phân tích lỗi ASR theo từ (A1): căn chỉnh ref/hyp ở mức từ, phân loại mỗi lỗi, và đo **trần của biasing** — từ ref bị
sai có nằm sẵn trong ngữ cảnh được phép lúc thi không (theo từng nguồn của prompt V2).

    python agent/asr_analyze.py --manifest agent/runs/asr/asr_heldout.manifest.json --asr agent/runs/asr/heldout.asr.json \
        --data data/train+public_test --out agent/runs/asr/REPORT_asr_v1_heldout.md

Nguồn ngữ cảnh (chỉ thứ có lúc thi; lượt user trước dùng transcript ASR, không dùng nhãn):
  conv     = transcript ASR các lượt user trước + text assistant
  results  = mọi giá trị chuỗi trong tool_results của history
  calls    = giá trị chuỗi trong tham số tool_calls của history
  profile  = context.user_profile
  catalog  = name_vi / mô tả / enum của các app đã cài (train + public_test catalog)
Không đọc *_environment/.
"""
import argparse
import collections
import json
import re
import unicodedata


TONES = {"\u0300": "\u2082", "\u0301": "\u2081", "\u0303": "\u2084", "\u0309": "\u2083", "\u0323": "\u2085"}   # ký tự chỉ số dưới (không phải \\d)


def syl_key(w):
    """'khoá' và 'khóa' (hai cách đặt dấu thanh) → cùng khoá: âm tiết bỏ dấu thanh + mã thanh."""
    d = unicodedata.normalize("NFD", w)
    t = "".join(TONES[c] for c in d if c in TONES)
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES)) + t


def norm(s):
    s = unicodedata.normalize("NFC", (s or "").lower())
    s = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()
    return " ".join(syl_key(w) for w in s.split())


def align(r, h):
    """Levenshtein ở mức từ → list (op, ref_span, hyp_span); gộp các op lỗi liền nhau thành một cụm."""
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("=" if r[i - 1] == h[j - 1] else "S", [r[i - 1]], [h[j - 1]]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", [r[i - 1]], []))
            i -= 1
        else:
            ops.append(("I", [], [h[j - 1]]))
            j -= 1
    ops.reverse()
    out = []
    for op in ops:
        if op[0] != "=" and out and out[-1][0] != "=":
            out[-1] = ("E", out[-1][1] + op[1], out[-1][2] + op[2])
        else:
            out.append(op if op[0] == "=" else ("E", op[1], op[2]))
    return [x for x in out if x[0] == "E"]


NUMW = set("không một mốt hai ba bốn tư năm lăm nhăm sáu bảy tám chín mười mươi trăm nghìn ngàn triệu tỷ tỉ linh lẻ rưỡi phẩy".split())


def strip_acc(s):
    s = re.sub("[\u2081-\u2085]", "", s)
    s = unicodedata.normalize("NFD", s.replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def is_numtok(w):
    return bool(re.search(r"\d", w)) or w in NUMW_K


def kind_of(ref_w, hyp_w, ref_raw):
    rs, hs = " ".join(ref_w), " ".join(hyp_w)
    allnum = lambda ws: bool(ws) and all(is_numtok(w) for w in ws) and (len(ws) >= 2 or any(re.search(r"\d", w) or w in STRONG_K for w in ws))  # noqa: E731
    dig = lambda ws: bool(ws) and all(re.fullmatch(r"[\d.,]+", w) for w in ws)        # noqa: E731
    wrd = lambda ws: bool(ws) and all(w in NUMW_K for w in ws)                         # noqa: E731
    if (dig(ref_w) and wrd(hyp_w)) or (wrd(ref_w) and dig(hyp_w)):
        return "định dạng số (chữ↔số)"
    if any(re.search(r"\d", w) for w in ref_w + hyp_w) or allnum(ref_w) or allnum(hyp_w):
        return "sai số/tiền/giờ"
    if ref_w and hyp_w and strip_acc(rs) == strip_acc(hs):
        return "sai dấu"
    if ref_w and hyp_w and rs.replace(" ", "") == hs.replace(" ", ""):
        return "tách/ghép từ"
    capsent = {syl_key(m_.group(1).lower()) for m_ in re.finditer(r"(?<=[a-zà-ỹđ0-9,] )([A-ZĐÀ-Ỹ][\wà-ỹ]*)", ref_raw or "")}
    if ref_w and sum(w in capsent and w in PROPER for w in ref_w) * 2 >= len(ref_w) or (ref_w and not all(w in ENG for w in ref_w) and any(re.fullmatch(r"[a-z]+", w) and w in capsent for w in ref_w)):
        return "tên riêng"
    if any(w in ENG for w in ref_w + hyp_w):
        return "từ mượn/tiếng Anh"
    if not hyp_w:
        return "mất từ"
    if not ref_w:
        return "thừa từ"
    if all(w in FILLER for w in ref_w + hyp_w):
        return "từ đệm/khẩu ngữ"
    return "từ thường khác"


FILLER = set()
STRONG_K = set()
PROPER = set()
NUMW_K = set()
ENG = set()


def strings(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            strings(v, out)
    elif isinstance(o, list):
        for v in o:
            strings(v, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.manifest, encoding="utf-8"))["items"]
    H = json.load(open(a.asr, encoding="utf-8"))["hyps"]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")}
    cat = {}
    for f in ("train/train_tools_catalog.json", "public_test/public_test_tools_catalog.json"):
        for x in json.load(open(f"{a.data}/{f}", encoding="utf-8"))["apps"]:
            cat.setdefault(x["id"], x)
    # từ tiếng Anh/mượn: từ không dấu xuất hiện trong ref mà không phải âm tiết tiếng Việt thường gặp → xấp xỉ bằng danh sách
    global ENG, FILLER, PROPER, NUMW_K
    NUMW_K = {syl_key(w) for w in NUMW}
    global STRONG_K
    STRONG_K = {syl_key(w) for w in "mười mươi trăm nghìn ngàn triệu tỷ tỉ linh lẻ rưỡi phẩy".split()}
    FILLER = {syl_key(w) for w in ("à ạ ơi ờ ừ ừm ừ ờm nha nhé nhá nghen nghe nhỉ đi thôi vậy thế hả hông không ha đó đấy "
                                   "nè á ấy chứ mà luôn giùm dùm giúp với cái cho em anh chị tôi tui mình bạn coi xem rứa nớ").split()}
    # tên riêng: từ viết hoa KHÔNG ở đầu câu/đầu mệnh đề trong mọi transcript train (đủ thống kê)
    caps, low = collections.Counter(), collections.Counter()
    for rr in rows.values():
        for t in [rr] + [x for x in rr["history"] if x["role"] == "user"]:
            txt = t.get("audio_text") or ""
            for m_ in re.finditer(r"(?<=[a-zà-ỹđ0-9] )([A-ZĐÀ-Ỹ][\wà-ỹ]*)", txt):
                caps[syl_key(m_.group(1).lower())] += 1
            for w in re.findall(r"(?<![.?!] )\b([a-zà-ỹđ][\wà-ỹ]*)", txt):
                low[syl_key(w)] += 1
    PROPER = {w for w, c in caps.items() if c >= 1 and low[w] <= c}
    # độ hiếm: số mẫu train chứa từ đó (cho trần biasing: chỉ tính từ nội dung)
    df = collections.Counter()
    for rr in rows.values():
        ws = set(norm(" ".join([rr.get("audio_text") or ""] + [x.get("audio_text") or "" for x in rr["history"] if x["role"] == "user"])).split())
        df.update(ws)
    N_ROWS = len(rows)
    ENG = set("ok okay email mail zalo grab momo wifi app online check in out tivi tv vtv htv sms ron95 e5 diesel usd euro playlist "
              "podcast radio spotify youtube facebook messenger telegram zoom meet teams team meeting deadline voucher shopee lazada tiki "
              "booking taxi bus metro shipper ship cafe coffee pizza burger sale menu combo order ok code pin qr robot".split())
    errs, per_src = [], collections.Counter()
    n_err_words = 0
    for it in items:
        ref_raw = it.get("ref") or ""
        r, h = norm(ref_raw).split(), norm(H.get(it["uid"], "")).split()
        row = rows[it["sid"]]
        # ngữ cảnh lúc thi cho utt này
        conv = " ".join((H.get(uid, "") if role == "user" and uid else (t or "")) for role, t, uid in it["prev"])
        # history trước utt này: các lượt assistant đứng trước
        n_prev_user = sum(1 for role, _, uid in it["prev"] if role == "user")
        hist, seen_user = [], 0
        for t in row["history"]:
            if t["role"] == "user":
                if seen_user >= n_prev_user:
                    break
                seen_user += 1
            hist.append(t)
        if it["kind"] == "final":
            hist = row["history"]
        res = " ".join(strings([x.get("tool_results") for x in hist if x["role"] == "assistant"], []))
        calls = " ".join(strings([[c.get("arguments") for c in (x.get("tool_calls") or [])] for x in hist if x["role"] == "assistant"], []))
        prof = " ".join(strings(row["context"].get("user_profile") or {}, []))
        apps = [cat[x] for x in row["context"]["installed_apps"] if x in cat]
        catt = " ".join([x.get("name_vi", "") + " " + x.get("description", "") for x in apps] +
                        [str(e) for x in apps for t in x["tools"] for p in (t.get("parameters") or {}).get("properties", {}).values() for e in p.get("enum", [])])
        src = {k: " " + norm(v) + " " for k, v in (("conv", conv), ("results", res), ("calls", calls), ("profile", prof), ("catalog", catt))}
        for _, rw, hw in align(r, h):
            k = kind_of(rw, hw, ref_raw)
            n_err_words += max(len(rw), len(hw))
            content = [w for w in rw if w not in FILLER and w not in NUMW_K and df[w] / N_ROWS < 0.02]
            content_any = [w for w in rw if w not in FILLER and w not in NUMW_K]           # không lọc độ hiếm
            phrase = " " + " ".join(rw) + " "
            found = [s for s, txt in src.items() if content and phrase in txt]              # chặt: cả cụm liền nhau, từ hiếm
            found_w = [s for s, txt in src.items() if any(f" {w} " in txt for w in content)]           # theo từ, từ hiếm
            found_any = [s for s, txt in src.items() if any(f" {w} " in txt for w in content_any)]     # theo từ, không lọc
            in_prompt = bool(content_any) and any(f" {w} " in src["conv"] for w in content_any)
            errs.append({"uid": it["uid"], "kind": it["kind"], "type": k, "ref": " ".join(rw), "hyp": " ".join(hw), "in": found, "in_word": found_w, "in_any": found_any, "content": bool(content), "content_any": bool(content_any),
                         "in_prompt_v1": in_prompt, "sid": it["sid"],
                         "sent_ref": ref_raw, "sent_hyp": H.get(it["uid"], "")})
    by_type = collections.Counter(e["type"] for e in errs)
    fixable = [e for e in errs if e["in"]]
    src_c = collections.Counter(s for e in errs for s in e["in"])
    only = collections.Counter(e["in"][0] for e in errs if len(e["in"]) == 1)
    L = ["# Phân tích lỗi ASR v1 (`pw-tmp-c3d9`, prompt hội thoại) trên 405 đoạn ASR giữ riêng\n",
         "Tập: 123 mẫu train mà ASR v1 **không** học (seed 42, 8% theo sample id), 405 đoạn (123 lượt cuối + 282 history).",
         "Chạy Kaggle 2×T4, beam 5, fp16, 475 s. Prompt = transcript ASR các lượt user trước + text assistant (đúng điều kiện thi).\n",
         "## 1. Tổng\n"]
    return errs, by_type, fixable, src_c, only, L, n_err_words


if __name__ == "__main__":
    import sys
    errs, by_type, fixable, src_c, only, L, n_err_words = main()
    a = sys.argv
    out = a[a.index("--out") + 1]
    json.dump(errs, open(out.replace(".md", ".errors.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    real = [e for e in errs if e["type"] != "định dạng số (chữ↔số)"]
    print(len(errs), "cụm lỗi (", len(real), "không tính định dạng số );", n_err_words, "từ lỗi")
    print(f"{'loại':24}{'cụm':>5}{'% cụm':>7}{'từ nội dung':>12}{'có trong ngữ cảnh':>19}")
    for t, c in by_type.most_common():
        sub = [e for e in errs if e["type"] == t]
        print(f"{t:24}{c:5}{100 * c / len(errs):7.1f}{sum(e['content'] for e in sub):12}{sum(bool(e['in']) for e in sub):19}")
    cont = [e for e in errs if e["content"]]
    ca = [e for e in errs if e["content_any"]]
    print(f"TRẦN biasing: chặt {sum(bool(e['in']) for e in cont)}/{len(cont)} | theo từ (từ hiếm) {sum(bool(e['in_word']) for e in cont)}/{len(cont)} "
          f"| theo từ, không lọc độ hiếm {sum(bool(e['in_any']) for e in ca)}/{len(ca)} | từ đúng đã nằm trong prompt v1 (conv) mà vẫn sai: {sum(e['in_prompt_v1'] for e in ca)}")
    print(f"\nlỗi chứa từ nội dung: {len(cont)}; trong đó có sẵn trong ngữ cảnh: {len(fixable)} ({100 * len(fixable) / max(1, len(cont)):.0f}%)")
    print("theo nguồn:", src_c.most_common(), "| chỉ 1 nguồn:", only.most_common())
    for e in fixable:
        print(f"   [{e['ref']}] → [{e['hyp']}]  {e['in']}  ({e['type']})")
