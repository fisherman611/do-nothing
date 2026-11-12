"""C3 / N4 (PLAN_72h_v3 §1l): sửa ĐỊA DANH ở mức tham số khi tool ĐỌC có tham số nơi chốn trả danh sách rỗng.

An toàn như N1: chỉ chạy SAU khi đã gọi và kết quả rỗng; không bao giờ sửa tool ghi (dùng id từ kết quả tìm kiếm);
không thay bằng giá trị đã thử. Riêng mã IATA bịa (3 chữ in, không có trong bảng sân bay) kiểm NGAY TRƯỚC call vì chắc chắn sai.

Danh bạ ứng viên: `agent/data/places_vn.json` (tĩnh, nguồn công khai — không lấy gì từ public/private test) + thành phố
trong `user_profile` của mẫu. So khớp theo âm: bỏ dấu, tách âm tiết, quy đổi các cặp nhầm lẫn hay gặp của ASR tiếng Việt
(l/n, s/x, tr/ch, d/gi/r; vần ng→n, t→c) rồi đo khoảng cách chuỗi. Nhận khi điểm ≥ 0.75 và hơn ứng viên thứ hai ≥ 0.15.

Dùng: fix = fix_empty(call, res, item, utter, catalog, tried) -> call mới hoặc None; precheck_iata(call, item, utter) -> call mới hoặc None.
"""
import difflib
import json
import os
import re
import unicodedata

_PLACES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "places_vn.json"), encoding="utf-8"))
AIRPORTS = _PLACES["airports"]
CITIES = _PLACES["cities"]
STATIONS = _PLACES["stations"]

# tool đọc có tham số nơi chốn -> {tham số: loại ứng viên}
PLACE_TOOLS = {
    "flights.search": {"origin": "airport", "destination": "airport"},
    "train_bus_tickets.search": {"origin": "city", "destination": "city"},
    "parking.find": {"near": "any"},
    "maps.search_place": {"near": "any"},
    "bike_rental.find_bikes": {"near": "any"},
    "weather.get_current": {"location": "city"},
    "weather.get_forecast": {"location": "city"},
}
PAIR_TOOLS = ("flights.search", "train_bus_tickets.search")     # luật điểm đi = điểm đến
ACCEPT, MARGIN, IATA_ACCEPT = 0.75, 0.15, 0.85
_GENERIC = {"thanh", "pho", "tp", "san", "bay", "city", "airport", "station"}     # không bỏ "tỉnh/ga/bến": có trong tên (Hà Tĩnh)


def fold(s):
    """thường, bỏ dấu, đ→d, gộp khoảng trắng (như zaic_tools.utils._fold)."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def syls(s):
    return [w for w in re.findall(r"[a-z0-9]+", fold(s)) if w not in _GENERIC]


_ONSET = re.compile(r"^(ngh|ng|nh|tr|ch|th|ph|kh|gh|gi|qu|[bcdfghklmnprstvxz])?(.*?)(ng|nh|ch|[cmnpt])?$")
_ONSET_MAP = {"n": "l", "x": "s", "ch": "tr", "gi": "d", "r": "d", "ngh": "ng", "gh": "g", "k": "c", "qu": "cu"}
_FINAL_MAP = {"ng": "n", "nh": "n", "t": "c", "ch": "c", "p": "c"}


def canon(syl):
    """Quy đổi một âm tiết (đã bỏ dấu) về dạng gộp các cặp ASR hay nhầm."""
    m = _ONSET.match(syl)
    if not m:
        return syl
    on, nu, fi = m.group(1) or "", m.group(2), m.group(3) or ""
    return _ONSET_MAP.get(on, on) + nu + _FINAL_MAP.get(fi, fi)


def score(q, name):
    """Điểm giống nhau theo âm của hai tên (0..1)."""
    a, b = syls(q), syls(name)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if abs(len(a) - len(b)) > 1:
        return 0.0
    ca, cb = " ".join(canon(x) for x in a), " ".join(canon(x) for x in b)
    if ca == cb:
        return 0.95
    return 0.95 * difflib.SequenceMatcher(None, ca, cb, autojunk=False).ratio()


def candidates(kind, item=None):
    """Danh sách tên ứng viên theo loại + thành phố trong hồ sơ người dùng."""
    out = []
    if kind == "airport":
        out += [n for names in AIRPORTS.values() for n in names]
    if kind in ("city", "any", "airport"):
        out += CITIES
    if kind in ("city", "any"):
        out += STATIONS
    city = ((item or {}).get("context", {}).get("user_profile") or {}).get("city")
    if city:
        out.append(city)
    seen, uniq = set(), []
    for n in out:
        if fold(n) not in seen:
            seen.add(fold(n))
            uniq.append(n)
    return uniq


def is_known(value, kind, item=None):
    """Giá trị đã là một địa danh hợp lệ (khớp hẳn) → không sửa (rỗng vì lý do khác)."""
    if kind == "airport" and re.fullmatch(r"[A-Z]{3}", value or "") and value in AIRPORTS:
        return True
    return any(score(value, n) >= 1.0 for n in candidates(kind, item))


def best_match(value, kind, item=None, tried=()):
    """Ứng viên tốt nhất theo âm, hoặc None nếu không chắc."""
    tried_f = {fold(t) for t in tried}
    sc = sorted(((score(value, n), n) for n in candidates(kind, item) if fold(n) not in tried_f and fold(n) != fold(value)),
                reverse=True)
    if not sc or sc[0][0] < ACCEPT:
        return None
    if len(sc) > 1 and sc[0][0] - sc[1][0] < MARGIN and fold(sc[1][1]) != fold(sc[0][1]):
        return None
    return sc[0][1]


def place_from_utterance(utter, exclude=(), kind="airport"):
    """Tìm tên sân bay/thành phố nói trong câu (n-gram 1–3 âm tiết) — dùng khi model bịa mã IATA."""
    w = syls(utter)
    ex = {fold(e) for e in exclude}
    best = (0.0, None)
    for n in (3, 2, 1):                      # ưu tiên cụm dài hơn khi bằng điểm
        for i in range(len(w) - n + 1):
            frag = " ".join(w[i:i + n])
            for name in candidates(kind):
                if fold(name) in ex:
                    continue
                s = score(frag, name)
                if s > best[0]:
                    best = (s, name)
    return best[1] if best[0] >= IATA_ACCEPT else None


def _exclude_for(call, param, item):
    """Tên của các tham số nơi chốn khác trong cùng call (để không đổi điểm đến thành điểm đi)."""
    out = []
    for p in PLACE_TOOLS.get(call["name"], {}):
        if p != param and isinstance(call["arguments"].get(p), str):
            v = call["arguments"][p]
            out.append(v)
            out += AIRPORTS.get(v, [])
    return out


def precheck_iata(call, item, utter):
    """flights.search với mã 3 chữ in KHÔNG có trong bảng sân bay → thay bằng tên nơi nói trong câu. None nếu không sửa."""
    if call["name"] != "flights.search":
        return None
    args = dict(call["arguments"])
    changed = False
    for p in ("origin", "destination"):
        v = args.get(p)
        if isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v) and v not in AIRPORTS:
            name = place_from_utterance(utter or "", exclude=_exclude_for(call, p, item))
            if name:
                args[p] = name
                changed = True
    return {"name": call["name"], "arguments": args} if changed else None


def is_empty(res):
    """Kết quả tool đọc là danh sách rỗng (cùng định nghĩa với N1 relax_empty)."""
    L = [v for v in res.values() if isinstance(v, list)] if isinstance(res, dict) and res.get("ok") else []
    return bool(L) and not any(L)


def fix_empty(call, res, item, utter, catalog, tried=()):
    """Tool đọc có tham số nơi chốn trả rỗng → call mới với địa danh đã sửa. `tried`: các bộ tham số đã gọi (dict). None nếu không sửa được."""
    spec = (catalog or {}).get(call["name"]) or {}
    if spec.get("kind", "read") != "read" or call["name"] not in PLACE_TOOLS or not is_empty(res):
        return None
    params = PLACE_TOOLS[call["name"]]
    args = dict(call["arguments"])
    tried_vals = {p: [t.get(p) for t in tried if isinstance(t.get(p), str)] for p in params}
    changed = []
    for p, kind in params.items():
        v = args.get(p)
        if not isinstance(v, str) or not v.strip():
            continue
        if kind == "airport" and re.fullmatch(r"[A-Z]{3}", v):
            if v in AIRPORTS:
                continue
            name = place_from_utterance(utter or "", exclude=_exclude_for(call, p, item))
            if name:
                args[p] = name
                changed.append(p)
            continue
        if is_known(v, kind, item):
            continue
        name = best_match(v, kind, item, tried=tried_vals.get(p, ()))
        if name:
            args[p] = name
            changed.append(p)
    # luật điểm đi = điểm đến → điểm đi lấy thành phố trong hồ sơ
    if call["name"] in PAIR_TOOLS and isinstance(args.get("origin"), str) and isinstance(args.get("destination"), str) \
            and fold(args["origin"]) == fold(args["destination"]):
        city = (item.get("context", {}).get("user_profile") or {}).get("city")
        if city and fold(city) != fold(args["destination"]) and "origin" not in changed:
            args["origin"] = city
            changed.append("origin")
    if not changed or any({**t} == args for t in tried):
        return None
    return {"name": call["name"], "arguments": args}
