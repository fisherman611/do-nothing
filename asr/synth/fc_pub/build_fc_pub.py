"""FC v4 (PLAN_public_text_augment §3, chủ repo 13/11): biến thể FC từ mẫu public "chắc chắn", giữ phân phối test.
    python synth/fc_pub/build_fc_pub.py [--limit N] [--seed 0]
→ data/fc_pub/fc_pub_items.jsonl (cùng schema train.jsonl + catalog_overlay, dùng với build_data.py --extra) + fc_pub_report.json

Mẫu gốc: 634 public − nghi ngờ (uncertain.json) − có ghi chú review = 584 (giữ 98 câu "cần nghe": chỉ lệch số giữa 3 ASR,
nhãn đã được probe P1 xác nhận). Số biến thể theo nhóm để khớp test 83/11/6: có call ×2, hỏi lại ×3, rỗng ×2.
Mỗi biến thể:
  1. đổi tên NHẤT QUÁN trên state + history + nhãn + lời nói: id có chữ số (evt_04 → evt_731), tên người (theo từng token tên riêng,
     để "anh Dũng" và "Sếp Dũng" cùng đổi); KHÔNG đổi số, từ xưng hô, thiết bị;
  2. KIỂM: chạy lại call câu cuối trên state mới bằng Executor của public; mọi kết quả phải ok và bằng đổi_tên(kết quả gốc), lệch → bỏ;
  3. 11 app chỉ có ở test → tên app mới ngẫu nhiên theo từng biến thể (catalog_overlay, installed_apps, call) để không học thuộc tên;
  4. lời người dùng = nhãn P1 (đã sửa thực thể); từ địa phương cho profile Trung/Nam; phiên âm tên/mã nước ngoài ở một phần câu.
Không bỏ dấu câu ở đây (build_data.py --strip-punct-p làm sau nhiễu ASR).
"""
import argparse, collections, copy, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PT = os.path.join(ROOT, "data", "train+public_test", "public_test")
OUT = os.path.join(ROOT, "data", "fc_pub")
sys.path.insert(0, PT)
from zaic_tools import Executor  # noqa: E402

REGION = {"Hà Nội": "Bac", "Đà Nẵng": "Trung", "Huế": "Trung", "TP.HCM": "Nam", "Cần Thơ": "Nam"}
N_VAR = {"call": 2, "ask": 3, "empty": 2}
ID_RE = re.compile(r"(?<![\w.])([a-z]{1,12})_([0-9a-z]{1,10})(?![\w])")
TITLES = {"Anh", "Chị", "Em", "Cô", "Chú", "Bác", "Ông", "Bà", "Bố", "Mẹ", "Ba", "Má", "Mạ", "Vợ", "Chồng", "Con", "Cháu", "Thầy",
          "Sếp", "Chủ", "Nhà", "Mr.", "Mrs.", "Ms.", "Dr.", "Bé", "Thằng", "Cu", "Đội", "Team", "Nhóm", "Gia", "Đình", "Hàng", "Xóm",
          "Bạn", "Cậu", "Dì", "Dượng", "Mợ", "Thím", "Ngoại", "Nội", "O", "Mệ", "Bọ", "Lớp", "Phòng", "Công", "Ty", "Shop", "Cửa",
          "Ngân", "Giáo", "Trưởng", "Nhân", "Sự", "Tổ", "Hội", "Câu", "Lạc", "Bộ", "Khách", "Chủ", "Tiệm", "Quán", "Group", "Family"}
VN_ASCII = {"Minh", "Lan", "Linh", "Nam", "Hung", "Huy", "Tuan", "Long", "Quan", "Anh", "Mai", "Lam", "Nhi", "Vy", "Thu", "Ha", "Hoa",
            "Son", "Phong", "Trang", "Duc", "Khoa", "Kha", "Tam", "Hai", "Binh", "Thanh", "Hien", "Vinh", "Phuong", "Ngoc", "Thao"}
VN_POOL = ["Khải", "Diệp", "Thục", "Quế", "Khuê", "Tửng", "Uyên", "Hoành", "Liễu", "Tuệ", "Mận", "Xuyến", "Lữ", "Ngải", "Thiện",
           "Nhạn", "Bích", "Cẩm", "Đăng", "Hạ", "Kiều", "Lụa", "Mến", "Nguyệt", "Phấn", "Quyên", "Sương", "Tấn", "Ước", "Vỹ",
           "Châu", "Đoan", "Giang", "Hậu", "Khánh", "Lực", "Nghĩa", "Oanh", "Phúc", "Quỳnh", "Sang", "Thắm", "Triều", "Việt", "Yến"]
FOREIGN_POOL = ["Olivia", "Daniel", "Hannah", "Lucas", "Emma", "Noah", "Chloe", "Ethan", "Grace", "Oscar", "Ivy", "Leo", "Nora",
                "Felix", "Ruby", "Hugo", "Mia", "Isaac", "Zoe", "Victor", "Clara", "Adrian", "Stella", "Mason", "Aria",
                "Bennett", "Carter", "Dawson", "Ellis", "Fischer", "Garcia", "Holmes", "Ingram", "Jensen", "Keller", "Lawson",
                "Moreau", "Nakamura", "Ortega", "Park", "Quinn", "Romero", "Sato", "Tanner", "Weber", "Yamada", "Zhou"]
APP_ALIAS = {
    "stocks": ["chung_khoan", "co_phieu", "trade_app", "san_dau_tu", "portfolio"],
    "translator": ["dich_thuat", "phien_dich", "trans_app", "ngon_ngu"],
    "grocery": ["di_cho_online", "sieu_thi", "thuc_pham", "cho_nha"],
    "recipes": ["cong_thuc", "nau_an", "bep_nha", "mon_ngon"],
    "public_transit": ["xe_buyt", "giao_thong_cong_cong", "metro_bus", "transit"],
    "parking": ["gui_xe", "bai_do", "park_app", "do_xe"],
    "train_bus_tickets": ["ve_tau_xe", "dat_ve_xe", "ticket_go", "ve_lien_tinh"],
    "ecommerce": ["mua_sam_online", "san_tmdt", "shop_app", "cho_mang"],
    "event_tickets": ["ve_su_kien", "ticketbox", "ve_show", "su_kien"],
    "habit_tracker": ["thoi_quen", "habit_app", "nhat_ky_thoi_quen", "theo_doi_thoi_quen"],
    "pantry": ["tu_bep", "kho_bep", "do_trong_nha", "pantry_app"],
}
LETTER = {"A": "a", "B": "bê", "C": "xê", "D": "đê", "E": "e", "F": "ép", "G": "giê", "H": "hát", "I": "i", "J": "giây", "K": "ca",
          "L": "eo", "M": "em", "N": "en", "O": "ô", "P": "pê", "Q": "quy", "R": "e rờ", "S": "ét", "T": "tê", "U": "u", "V": "vê",
          "W": "vê kép", "X": "ích", "Y": "i dài", "Z": "dét"}
DIALECT = {
    "Trung": [(r"\blàm gì\b", "mần chi"), (r"\bở đâu\b", "ở mô"), (r"\bthế này\b", "ri"), (r"\bnhư vậy\b", "như rứa"),
              (r"\bgì\b", "chi"), (r"\bđâu\b", "mô"), (r"\bnày\b", "ni"), (r"\bbây giờ\b", "chừ"), (r"\bvậy\b", "rứa"),
              (r"\bthế nào\b", "răng"), (r"\btại sao\b", "răng"), (r"\bnhé\b", "nghe"), (r"\bnha\b", "nghe"), (r"\btôi\b", "tui"),
              (r"\bkia\b", "tê"), (r"\bnhỉ\b", "hè"), (r"\blàm gì\b", "mần chi"), (r"\bở đâu\b", "ở mô"),
              (r"\bthế này\b", "ri"), (r"\bnhư vậy\b", "như rứa"),
              (r"\bsao\b(?=\s+(?:rồi|vậy|thế|không|em|anh|chị|con|ta))", "răng"), (r"\bthế\b", "rứa")],
    "Nam": [(r"\bkhông\b(?=\s*(?:$|em\b|anh\b|chị\b|con\b|vậy\b|nhỉ\b|ta\b|ạ\b|[?.!,]))", "hông"), (r"\bvậy\b", "dzậy"),
            (r"\btôi\b", "tui"), (r"\bnhé\b", "nghen"), (r"\bthế\b", "vậy"), (r"\bđấy\b", "đó"), (r"\bbao nhiêu\b", "nhiêu"),
            (r"\bnày\b(?=\s*(?:$|[?.!,]))", "nè"), (r"\bthôi\b(?=\s*$)", "thôi nghen")],
}
P_DIALECT, P_PHON = {"Trung": 0.5, "Nam": 0.6}, 0.3   # câu "đậm" (bộ từ chặt) mục tiêu cao hơn public một chút: Trung ~15–20% (public 11%), Nam ~6–8% (public 0,2%)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load():
    items = {x["id"]: x for x in jl(os.path.join(PT, "public_test.jsonl"))}
    labels = {x["id"]: x["tool_calls"] for x in jl(os.path.join(ROOT, "result", "claude_submission_final.jsonl"))}
    p1 = {r["uid"]: r["label"] for r in jl(os.path.join(ROOT, "synth", "tts_text_pub", "tts_text_pub.jsonl")) if r["src"] == "orig"}
    unc = set(json.load(open(os.path.join(ROOT, "tools", "test_review", "uncertain.json"), encoding="utf-8")))
    rv = json.load(open(os.path.join(ROOT, "result", "test_review.json"), encoding="utf-8"))
    rvbad = {k for k, v in rv.items() if v.get("comment") and v.get("status") != "ok"}
    cat = json.load(open(os.path.join(PT, "public_test_tools_catalog.json"), encoding="utf-8"))
    tools = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
    trcat = json.load(open(os.path.join(ROOT, "data", "train+public_test", "train", "train_tools_catalog.json"), encoding="utf-8"))
    trnames = {t["name"] for a in trcat["apps"] for t in a["tools"]}
    test_only = {a["id"] for a in cat["apps"] if not any(t["name"] in trnames for t in a["tools"])}
    phon = collections.defaultdict(set)
    for r in jl(os.path.join(ROOT, "synth_phonetic", "train_phonetic.jsonl")):
        for x in (r.get("aug") or {}).get("replacements", []):
            if x.get("from") and x.get("to") and x["from"] != x["to"]: phon[x["from"]].add(x["to"])
    return items, labels, p1, unc | rvbad, tools, test_only, {k: sorted(v) for k, v in phon.items()}


def category(calls):
    n = [c["name"] for c in calls]
    return "empty" if not n else ("ask" if "sys.ask_missing_info" in n else "call")


# ---------- đổi tên nhất quán ----------
def collect_ids(text):
    return {m.group(0) for m in ID_RE.finditer(text) if any(ch.isdigit() for ch in m.group(2))}


FOREIGN_LAST = ["Bennett", "Carter", "Dawson", "Ellis", "Fischer", "Garcia", "Holmes", "Ingram", "Jensen", "Keller", "Lawson",
                "Moreau", "Nakamura", "Ortega", "Park", "Quinn", "Romero", "Sato", "Tanner", "Weber", "Yamada", "Zhou"]
FOREIGN_FIRST = [p for p in FOREIGN_POOL if p not in FOREIGN_LAST]
ADDR_KEYS = {"address", "location", "pickup", "dropoff", "home_address", "work_address", "destination", "origin", "near", "venue",
             "board_at", "alight_at", "stop", "area"}
COMMON = set()   # từ thông dụng (chữ thường) trong lời nói train → không coi là tên để đổi (vd "Mai" = ngày mai)


def load_common():
    c = collections.Counter()
    for l in open(os.path.join(ROOT, "data", "train+public_test", "train", "train.jsonl"), encoding="utf-8"):
        s = json.loads(l)
        for t in [s["audio_text"]] + [h.get("audio_text", "") for h in s["history"] if h["role"] == "user"]:
            c.update(w for w in re.findall(r"\w+", t) if w.islower())
    return {w for w, n in c.items() if n >= 3}


def person_tokens(state):
    """→ {token: vai}: người Việt chỉ đổi TÊN (token cuối); tên nước ngoài đổi cả tên lẫn họ."""
    names = []
    for c in (state.get("contacts") or {}).get("contacts", []): names.append(c.get("name", ""))
    for b in (state.get("banking") or {}).get("beneficiaries", []): names.append(b.get("name", ""))
    for ch in (state.get("chat_ott") or {}).get("chats", []):
        if ch.get("type") == "direct": names.append(ch.get("name", ""))
    for e in (state.get("email") or {}).get("emails", []): names.append(e.get("from_name", ""))
    out = {}
    for n in names:
        toks = [t for t in re.split(r"[\s\-–,]+", n or "") if len(t) >= 2 and t[0].isupper() and t not in TITLES
                and t.isalpha() and t.lower() not in COMMON]
        if not toks: continue
        foreign = all(t.isascii() and t not in VN_ASCII for t in toks) and len(toks) >= 1
        if foreign:
            out.setdefault(toks[0], "first_f")
            for t in toks[1:]: out.setdefault(t, "last_f")
        else:
            out.setdefault(toks[-1], "given_vn")
    return out


def address_runs(state):
    """cụm chữ hoa trong địa chỉ/địa điểm (vd "Nguyễn Thị Minh Khai") → bảo vệ, không đổi token bên trong."""
    runs = set()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ADDR_KEYS and isinstance(v, str):
                    for r in re.findall(r"(?:[A-ZÀ-Ỹ][\wÀ-ỹ]*\s+){1,5}[A-ZÀ-Ỹ][\wÀ-ỹ]*", v): runs.add(r.strip())
                walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(state)
    return runs


def build_map(blob, state, rng):
    low = blob.lower(); used = set(re.findall(r"\w+", low)); m = {}
    for old in sorted(collect_ids(blob)):
        pre, suf = old.split("_", 1)
        for _ in range(50):
            new_suf = "".join(rng.choice("0123456789") for _ in suf) if suf.isdigit() else \
                "".join(rng.choice("0123456789abcdefghijklmnopqrstuvwxyz") if ch.isalnum() else ch for ch in suf)
            if suf.isdigit() and len(suf) > 1 and new_suf[0] == "0": continue
            new = f"{pre}_{new_suf}"
            if new != old and new.lower() not in used and new not in m.values(): m[old] = new; break
    pools = {"given_vn": VN_POOL, "first_f": FOREIGN_FIRST, "last_f": FOREIGN_LAST}
    for tok, role in sorted(person_tokens(state).items()):
        pool = [p for p in pools[role] if p.lower() not in used and p not in m.values()]
        if pool: m[tok] = rng.choice(pool)
    return m


PROTECT = []   # cụm địa chỉ của mẫu hiện tại (đặt trong make_variant)


def apply_map(obj, m, pat):
    s = json.dumps(obj, ensure_ascii=False)
    if not m: return json.loads(s)
    keep = {}
    for i, r in enumerate(sorted(PROTECT, key=len, reverse=True)):
        if r in s:
            ph = f"\u0000{i}\u0000"; keep[ph] = r; s = s.replace(r, ph)
    s = pat.sub(lambda x: m[x.group(0)], s)
    for ph, r in keep.items(): s = s.replace(ph, r)
    return json.loads(s)


def map_pattern(m):
    if not m: return None
    keys = sorted(m, key=len, reverse=True)
    return re.compile(r"(?<![\w.])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")


# ---------- lời nói ----------
def protected_words(calls):
    out = set()
    for c in calls:
        for v in (c.get("arguments") or {}).values():
            if isinstance(v, str): out |= set(re.findall(r"\w+", v.lower()))
    return out


def dialect(text, region, rng, protect):
    if region not in DIALECT or rng.random() >= P_DIALECT[region]: return text, 0
    n = 0
    for pat, rep in DIALECT[region]:
        def sub(mm):
            nonlocal n
            if any(w in protect for w in re.findall(r"\w+", mm.group(0).lower())): return mm.group(0)
            n += 1
            r = rep
            return r[0].upper() + r[1:] if mm.group(0)[0].isupper() else r
        text = re.sub(pat, sub, text, flags=re.I)
    return text, n


def phonetic(text, rng, phon):
    n = 0
    def acro(mm):
        nonlocal n
        w = mm.group(0)
        if rng.random() >= P_PHON: return w
        n += 1
        return " ".join(LETTER.get(ch, ch) for ch in w)
    text = re.sub(r"(?<!\w)[A-Z]{2,5}(?!\w)", acro, text)
    for k in sorted(phon, key=len, reverse=True):
        if k in text and rng.random() < P_PHON:
            text = text.replace(k, rng.choice(phon[k]), 1); n += 1
    return text, n


# ---------- một biến thể ----------
def make_variant(sid, it, calls, p1, tools, test_only, phon, exe, base_res, k, rng):
    state = exe.db[sid]
    region = REGION.get(it["context"]["user_profile"].get("city"), "Bac")
    hist = copy.deepcopy(it["history"]); u = 0
    for t in hist:
        if t["role"] == "user":
            u += 1; t["audio_text"] = p1.get(f"{sid}_u{u}", "")
    final_text = p1.get(sid, "")
    blob = json.dumps([state, hist, calls, final_text, it["context"]], ensure_ascii=False)
    PROTECT[:] = sorted(address_runs(state) | address_runs(it["context"]))
    m = build_map(blob, state, rng); pat = map_pattern(m)
    st2, hist2, calls2 = (apply_map(x, m, pat) for x in (state, hist, calls))
    final2 = apply_map(final_text, m, pat); ctx2 = apply_map(it["context"], m, pat)
    vid = f"{sid}#pv{k}"
    exe.db[vid] = st2
    sess = exe.open(vid)
    res2 = [sess.call(c["name"], c.get("arguments") or {}) for c in calls2]
    del exe.db[vid]
    if any(not isinstance(r, dict) or r.get("ok") is False or "error" in r for r in res2): return None, "call lỗi trên state mới"
    if json.dumps(res2, ensure_ascii=False, sort_keys=True) != json.dumps(apply_map(base_res, m, pat), ensure_ascii=False, sort_keys=True):
        return None, "kết quả lệch sau đổi tên"
    # 11 app chỉ có ở test → tên mới theo biến thể
    inst = ctx2["installed_apps"]; ren = {}
    for app in list(inst):
        if app in test_only: ren[app] = rng.choice(APP_ALIAS.get(app, [app + "_app"]))
    def rn(name):
        a, _, t = name.partition(".")
        return f"{ren[a]}.{t}" if a in ren else name
    ctx2["installed_apps"] = {ren.get(a, a): {"tools": [rn(t) for t in v["tools"]]} for a, v in inst.items()}
    for t in hist2:
        for c in t.get("tool_calls") or []: c["name"] = rn(c["name"])
    for c in calls2: c["name"] = rn(c["name"])
    overlay = []
    for a, v in inst.items():
        for tn in v["tools"]:
            if tn in tools:
                spec = copy.deepcopy(tools[tn]); spec["name"] = rn(tn); spec["app"] = ren.get(spec.get("app", a), spec.get("app", a))
                overlay.append(spec)
    if ren:   # còn tên app gốc trong mô tả ("lấy từ recipes.search"), trong tham số ask_missing_info.tool_name (là NHÃN) và kết quả → đổi hết
        def rj(obj):
            js = json.dumps(obj, ensure_ascii=False)
            for app, al in ren.items(): js = re.sub(r"(?<![\w])" + re.escape(app) + r"\.", al + ".", js)
            return json.loads(js)
        overlay, calls2, res2, hist2 = rj(overlay), rj(calls2), rj(res2), rj(hist2)
    # lời nói: từ địa phương + phiên âm (chỉ lượt user)
    n_dia = n_ph = 0
    turn_calls = []
    for j, t in enumerate(hist2):
        if t["role"] == "user":
            nxt = next((x for x in hist2[j + 1:] if x["role"] == "assistant"), {})
            turn_calls.append((t, nxt.get("tool_calls") or []))
    for t, cs in turn_calls:
        t["audio_text"], a1 = dialect(t["audio_text"], region, rng, protected_words(cs)); n_dia += a1
        t["audio_text"], b1 = phonetic(t["audio_text"], rng, phon); n_ph += b1
    final2, a1 = dialect(final2, region, rng, protected_words(calls2)); n_dia += a1
    final2, b1 = phonetic(final2, rng, phon); n_ph += b1
    item = {"id": vid, "audio": it["audio"], "audio_text": final2, "context": ctx2, "history": hist2,
            "label": {"tool_calls": calls2, "tool_results": res2}, "catalog_overlay": overlay,
            "meta": {"src": "pub_variant", "seed": sid, "region": region, "category": category(calls), "renamed": len(m),
                     "app_alias": ren, "dialect_subs": n_dia, "phon_subs": n_ph}}
    return item, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items, labels, p1, drop, tools, test_only, phon = load()
    COMMON.update(load_common())
    seeds = [s for s in items if s not in drop]
    if a.limit: seeds = seeds[: a.limit]
    exe = Executor(os.path.join(PT, "public_test_environment"), os.path.join(PT, "public_test_tools_catalog.json"))
    os.makedirs(OUT, exist_ok=True)
    out, fails, cnt = [], collections.Counter(), collections.Counter()
    for sid in seeds:
        calls = labels[sid]; cat_ = category(calls)
        sess = exe.open(sid); base_res = [sess.call(c["name"], c.get("arguments") or {}) for c in calls]
        want, got, tries = N_VAR[cat_], 0, 0
        while got < want and tries < want * 3:
            rng = random.Random(f"{a.seed}:{sid}:{tries}"); tries += 1
            item, why = make_variant(sid, items[sid], calls, p1, tools, test_only, phon, exe, base_res, got, rng)
            if item is None: fails[why] += 1; continue
            out.append(item); got += 1; cnt[cat_] += 1
    with open(os.path.join(OUT, "fc_pub_items.jsonl"), "w", encoding="utf-8") as f:
        for x in out: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    n = len(out)
    app_share = sum(any(c["name"].split(".")[0] in {v for x in [o["meta"]["app_alias"]] for v in x.values()} for c in o["label"]["tool_calls"]) for o in out)
    def strong(t, reg):
        T = r"\b(mô|răng|rứa|chừ|ni|nớ|tê|hè|mi|tau|mạ|chi|nghe)\b"; N = r"\b(nè|hông|dzậy|nghen|tui|nhiêu)\b"
        return len(set(re.findall(T if reg == "Trung" else N, t.lower()))) >= 2
    rep = {"seeds": len(seeds), "variants": n, "category": {k: round(v / n, 3) for k, v in cnt.items()},
           "region": dict(collections.Counter(o["meta"]["region"] for o in out)),
           "fails": dict(fails), "with_test_only_app_alias": round(app_share / n, 3),
           "dialect_strong": {reg: round(sum(strong(o["audio_text"], reg) for o in out if o["meta"]["region"] == reg) /
                                         max(1, sum(o["meta"]["region"] == reg for o in out)), 3) for reg in ("Trung", "Nam")},
           "phonetic_items": round(sum(o["meta"]["phon_subs"] > 0 for o in out) / n, 3),
           "renamed_avg": round(sum(o["meta"]["renamed"] for o in out) / n, 1),
           "final_steps": sum(len(o["label"]["tool_calls"]) + 1 for o in out),
           "hist_steps": sum(len(t.get("tool_calls") or []) + 1 for o in out for t in o["history"] if t["role"] == "assistant")}
    json.dump(rep, open(os.path.join(OUT, "fc_pub_report.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
