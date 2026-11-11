"""Bắt số tiền / số lượng nói bằng tiếng Việt trong một câu → tập ứng viên số nguyên (đồng).

Mục đích (REPORT_errors_fmtA_mid_e1 §2.4): model sinh "một trăm hai mươi nghìn" → 1200000 (×10), "ba trăm" → 3000000,
"hai trăm" → 200. Hậu kiểm: nếu số model sinh không khớp ứng viên nào nhưng lệch đúng 10^k so với một ứng viên → thay.

    >>> candidates("chuyển cho Hòa một trăm hai mươi nghìn qua ví")
    {120000}
    >>> candidates("chuyển thằng Út ba trăm bằng tài khoản")        # tiền nói tắt: 'ba trăm' = 300 nghìn
    {300, 300000}
    >>> candidates("nạp 50 ngàn, hai triệu tám, 200k, 1tr5, một trăm rưỡi")
    {50000, 2800000, 200000, 1500000, 150, 150000}
Không đụng tới giờ/ngày ("8 giờ", "ngày 11") — bỏ số đứng trước 'giờ', 'phút', 'tháng', 'ngày', 'tuổi', 'người', 'chỗ'...
"""
import re
import unicodedata

UNITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6,
         "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10}
SCALE = {"nghìn": 1000, "ngàn": 1000, "k": 1000, "triệu": 10 ** 6, "tr": 10 ** 6, "củ": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
NOT_MONEY_AFTER = {"giờ", "h", "phút", "tiếng", "ngày", "tháng", "năm", "tuổi", "người", "chỗ", "lần", "phần", "cái", "chiếc",
                   "viên", "tập", "kênh", "số", "độ", "phần_trăm", "%", "km", "m", "kg", "g", "ly", "cốc", "vé", "lít", "bước"}
WORD = re.compile(r"[0-9][0-9.,]*|[^\W\d_]+|%", re.U)


def _norm(s):
    return unicodedata.normalize("NFC", s.lower())


def _digits(tok):
    t = tok.replace(".", "").replace(",", "")
    return int(t) if t.isdigit() else None


def candidates(text, money_only=True):
    """Tập số nguyên có thể là số tiền trong câu (kể cả cách nói tắt: 'ba trăm' → 300 và 300000)."""
    toks = WORD.findall(_norm(text))
    out, i, n = set(), 0, len(toks)
    while i < n:
        val, j, saw_scale, had_word = _parse_number(toks, i)
        if j > i and val is not None and val > 0:
            nxt = toks[j] if j < n else ""
            nxt2 = toks[j + 1] if j + 1 < n else ""
            is_time_like = nxt in NOT_MONEY_AFTER or (nxt == "rưỡi" and nxt2 in NOT_MONEY_AFTER)
            if not (money_only and is_time_like):
                if nxt == "rưỡi":                              # hai triệu rưỡi, một trăm rưỡi, 1tr rưỡi
                    val = int(val * 1.5) if _is_round(val) else val
                    j += 1
                out.add(val)
                if not saw_scale and had_word and val < 1000 and val % 10 == 0 and val >= 20:   # nói tắt: 'ba trăm' = 300 nghìn
                    out.add(val * 1000)
                if not saw_scale and not had_word and val < 1000 and val % 10 == 0 and val >= 20:  # '200' ở câu tiền → 200 nghìn
                    out.add(val * 1000)
            i = j
        else:
            i += 1
    return out


def _is_round(v):
    return v >= 10 and v % 10 == 0


def _parse_number(toks, i):
    """Đọc một cụm số bắt đầu ở toks[i]. Trả (giá trị, chỉ số sau cụm, có đơn vị scale, có chữ số bằng lời).
    Hỗ trợ: chữ số ("120.000", "2tr", "200k", "1tr5"), lời ("một trăm hai mươi nghìn", "hai triệu tám", "ba trăm mốt")."""
    n = len(toks)
    total, cur, j, saw_scale, had_word = 0, 0, i, False, False
    started = False
    while j < n:
        t = toks[j]
        d = _digits(t)
        if d is not None:
            m = re.match(r"^([0-9][0-9.,]*)(k|tr|nghìn|ngàn|triệu|tỷ|tỉ)$", t) if False else None
            cur += d
            started = True
            j += 1
            # dạng dính: "2tr5", "1tr", "200k" được WORD tách thành ['2','tr','5'] / ['200','k']
            continue
        if t in UNITS and not (t == "không" and not started):
            v = UNITS[t]
            if t == "mười":
                cur = cur * 10 if cur and cur < 10 else 10
                # "hai mươi" đọc ở nhánh 'mươi'
            elif t in ("mốt", "lăm", "nhăm", "tư") and cur >= 10:
                cur += v
            else:
                cur += v
            started = had_word = True
            j += 1
            continue
        if t == "mươi" and started:
            cur = cur * 10 if cur < 10 else cur
            had_word = True
            j += 1
            continue
        if t == "trăm" and started:
            cur = cur * 100
            had_word = True
            j += 1
            # "hai trăm hai" = 220, "hai trăm linh năm" = 205
            if j < n and toks[j] in ("linh", "lẻ"):
                j += 1
            continue
        if t in SCALE and started:
            total += (cur or 1) * SCALE[t]
            cur, saw_scale = 0, True
            j += 1
            # "hai triệu tám" = 2.8 triệu; "1tr5" = 1.5 triệu: số đơn ngay sau scale lớn là phần lẻ
            if j < n and (toks[j] in UNITS or _digits(toks[j]) is not None) and (j + 1 >= n or toks[j + 1] not in ("trăm", "mươi", "nghìn", "ngàn", "triệu", "k", "tr")):
                frac = UNITS.get(toks[j], _digits(toks[j]))
                if frac is not None and 0 < frac < 10:
                    total += frac * SCALE[t] // 10
                    j += 1
            continue
        break
    if not started:
        return None, i, False, False
    return total + cur, j, saw_scale, had_word


def fix_amount(value, text, name=""):
    """value: số model sinh; trả (giá trị sửa, lý do) hoặc (value, None). Chỉ sửa khi lệch đúng 10^k (k∈{1,2,3,4})
    so với một ứng viên duy nhất gần nhất theo log, và value không trùng ứng viên nào."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return value, None
    cands = candidates(text)
    if not cands or value in cands:
        return value, None
    import math
    best = None
    for c in cands:
        r = math.log10(value / c)
        if abs(r - round(r)) < 1e-9 and 1 <= abs(round(r)) <= 4:
            d = abs(round(r))
            if best is None or d < best[0]:
                best = (d, c)
    if best is None:
        return value, None
    return best[1], f"{value}→{best[1]} (lệch 10^{best[0]}, ứng viên {sorted(cands)})"


if __name__ == "__main__":
    tests = [
        ("Chuyển cho Hòa một trăm hai mươi nghìn qua ví giúp anh cái.", {120000}),
        ("Làm thủ tục chuyến mai luôn, rồi chuyển thằng Út ba trăm bằng tài khoản thanh toán.", {300, 300000}),
        ("Chuyển cho Huy một trăm rưỡi qua ví", {150, 150000}),
        ("Bạn ơi, chuyển cho chủ nhà Phạm Hùng hai triệu tám và nạp giúp sếp Dũng hai trăm nghìn nhé.", {2800000, 200000}),
        ("Nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha.", {50000}),
        ("Đặt báo thức 8 giờ sáng mai, chuyển 200k cho Lan", {200000}),
        ("nạp 1tr5 và chuyển 2 triệu rưỡi", {1500000, 2500000}),
        ("chuyển bà Sáu hai trăm tiền vé, con Thảo ba trăm nghe", {200, 200000, 300, 300000}),
        ("Dời lịch họp sang 14 giờ 30 ngày 11", set()),
    ]
    ok = True
    for s, want in tests:
        got = candidates(s)
        flag = "OK " if got == want else "SAI"
        ok &= got == want
        print(flag, s[:60], "→", sorted(got), "" if got == want else f"(muốn {sorted(want)})")
    for v, s in ((1200000, tests[0][0]), (3000000, tests[1][0]), (200, tests[7][0]), (120000, tests[0][0])):
        print("fix_amount", v, "→", fix_amount(v, s))
    print("ALL OK" if ok else "CÓ LỖI")
