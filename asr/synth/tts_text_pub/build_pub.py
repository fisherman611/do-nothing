"""P1 (PLAN_public_text_augment §1): text ASR từ public test → cặp (lời gốc = spoken, nhãn đúng ý = label) + biến thể.
    python synth/tts_text_pub/build_pub.py   → synth/tts_text_pub/tts_text_pub.jsonl, review_needed.jsonl, REPORT_pub.md

Mỗi dòng: id, uid, sid, kind(final|history), spoken (lời đọc cho TTS, không chữ số), label (nhãn ASR), label_src (auto|same),
region (Bac|Trung|Nam), group, split(train), src (orig|variant), calls (tool_calls của lượt đó), fixes (các sửa đã áp).
Nguồn: transcript L2/e46f/v2 (đa số; ưu tiên bản khớp số trong nhãn v4), state public (tên thực thể), text assistant trong
history, nhãn result/claude_submission_final.jsonl, bảng lỗi agent/runs/asr/asr_subst_a4.json.
Biến thể: chỉ câu cuối, đổi số (tiền/giờ/độ/số lượng) và tên người bằng num_v3/entities_v3, giữ khung câu và phương ngữ.
"""
import collections, difflib, json, os, random, re, sys, unicodedata, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "synth", "tts_text_v3"))
import num_v3 as N                      # noqa: E402
from entities_v3 import VN_NAMES        # noqa: E402

PT = os.path.join(ROOT, "data", "train+public_test", "public_test")
ASR = [("L2", "agent/runs/asr/public.L2.asr.flat.json"), ("e46f", "agent/runs/asr/public.e46f.asr.flat.json"),
       ("v2", "agent/runs/asr/public.v2_medium.asr.json")]
LABELS = os.path.join(ROOT, "result", "claude_submission_final.jsonl")
SUBST = os.path.join(ROOT, "agent", "runs", "asr", "asr_subst_a4.json")
REGION = {"Hà Nội": "Bac", "Đà Nẵng": "Trung", "Huế": "Trung", "TP.HCM": "Nam", "Cần Thơ": "Nam"}
NAME_KEYS = ("name", "title", "label", "operator", "from_name", "provider", "venue", "restaurant", "airline")
NUMW = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm|trăm|nghìn|ngàn|triệu|linh|lẻ|rưỡi|hăm)"
NUM_SEQ = re.compile(rf"\b{NUMW}(?:\s+{NUMW})*\b")
KIN = {"bố", "mẹ", "ba", "má", "vợ", "chồng", "anh", "chị", "em", "cô", "chú", "bác", "ông", "bà", "thầy", "sếp", "con", "mạ", "o"}
COND = re.compile(r"\bnếu\b|\bthì\b|\bkhông thì\b|\bcòn không\b", re.I)
SELF = re.compile(r"à không|à thôi|à mà|à khoan|khoan đã|nhầm|à quên", re.I)


def strip_acc(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", s).lower())).strip()


def strip_punct(s):
    """nhãn ASR không dấu câu (yêu cầu chủ repo 13/11): bỏ . , ; : ! ? … " ' ( ) và gạch ngang đứng riêng; giữ gạch nối trong từ (Stand-up)."""
    s = re.sub(r"[.,;:!?…\"“”‘’()\[\]]+", " ", s)
    s = re.sub(r"(?<!\w)-(?!\w)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else s


def n2w(n, region="Bac"):
    """số nguyên → lời đọc (đến < 1 tỷ)."""
    if n < 1000: return N.three(n, region)
    th = N.thousand_word(region)
    parts = []
    m, r = divmod(n, 1_000_000)
    if m: parts.append(N.three(m, region) + " triệu")
    k, r2 = divmod(r, 1000)
    if k: parts.append((N.three(k, region) if (k >= 100 or not m) else "không trăm " + N.two(k)) + f" {th}")
    if r2: parts.append(N.three(r2, region) if (r2 >= 100 or not (m or k)) else "không trăm " + N.two(r2))
    return " ".join(parts)


def digits_to_words(text, region="Bac"):
    def rep(m):
        s = m.group(0)
        if s.startswith("0") or len(s) >= 9: return " ".join(N.D[int(c)] for c in s)
        return n2w(int(s), region)
    text = re.sub(r"(\d)[.,](\d{3})\b", r"\1\2", text)          # 1.500 → 1500
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])(?=\d)|(?<=\d)(?=[A-Za-zÀ-ỹ])", " ", text)   # V5 → V 5, 8h → 8 h
    return re.sub(r"\d+", rep, text)


def num_forms(v, kind, region):
    """các cách đọc của một giá trị trong nhãn (để chọn transcript khớp)."""
    out = set()
    regs = ("Bac", "Nam") if region == "Nam" else ("Bac",)
    if kind == "money":
        for rg in regs:
            th = N.thousand_word(rg)
            if v % 1_000_000 == 0: out.add(f"{N.three(v // 1_000_000, rg)} triệu")
            elif v % 100_000 == 0 and v > 1_000_000: out.add(f"{N.three(v // 1_000_000, rg)} triệu {N.D[(v % 1_000_000) // 100_000]}")
            if v % 1000 == 0 and v < 1_000_000:
                k = v // 1000; out.add(f"{N.three(k, rg)} {th}"); out.add(f"{N.three(k, rg, True)} {th}")
                if k % 100 == 0: out.add(f"{N.D[k // 100]} trăm {th}" if k < 1000 else "")
            if v % 500_000 == 0 and v % 1_000_000: out.add(f"{N.D[v // 1_000_000]} triệu rưỡi")
            out.add(n2w(v, rg))
    elif kind == "qty":
        out |= {n2w(v)} | ({N.two(v, region, True)} if v < 100 else set())
    elif kind == "temp":
        out |= {f"{n2w(v)} độ"} | ({f"{N.two(v, region, True)} độ"} if v < 100 else set())
    elif kind == "time":
        h, m = v
        for hh in {h, h % 12 or 12}:
            if m == 0: out.add(f"{N.two(hh)} giờ")
            elif m == 30: out |= {f"{N.two(hh)} rưỡi", f"{N.two(hh)} giờ rưỡi", f"{N.two(hh)} giờ ba mươi"}
            else: out |= {f"{N.two(hh)} giờ {N.two(m)}", f"{N.two(hh)} giờ {N.two(m, region, True)}"}
            if m >= 45: out.add(f"{N.two(hh % 12 + 1)} giờ kém {N.two(60 - m)}")
    return {x for x in out if x}


def label_values(calls):
    """(kind, value) từ tham số nhãn: tiền, số lượng, độ, giờ."""
    vals = []
    for c in calls:
        for k, v in (c.get("arguments") or {}).items():
            if isinstance(v, bool): continue
            if isinstance(v, (int, float)):
                if k in ("amount", "price", "max_price", "total", "budget"): vals.append(("money", int(v)))
                elif k == "temperature": vals.append(("temp", int(v)))
                elif k in ("quantity", "passengers", "seats", "hours", "people", "count", "speed", "brightness", "volume"): vals.append(("qty", int(v)))
            elif isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d", v):
                vals.append(("time", (int(v[11:13]), int(v[14:16]))))
            elif isinstance(v, list):
                for it in v:
                    if isinstance(it, dict) and isinstance(it.get("quantity"), int): vals.append(("qty", it["quantity"]))
    return vals


def walk_names(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in NAME_KEYS and isinstance(v, str) and 2 <= len(v) <= 60: acc.add(v.strip())
            walk_names(v, acc)
    elif isinstance(o, list):
        for x in o: walk_names(x, acc)


def find_by_id(o, ident):
    if isinstance(o, dict):
        if o.get("id") == ident: return o
        for v in o.values():
            r = find_by_id(v, ident)
            if r: return r
    elif isinstance(o, list):
        for x in o:
            r = find_by_id(x, ident)
            if r: return r


def cap_ngrams(text):
    """cụm từ viết hoa liên tiếp trong text assistant (tên riêng)."""
    toks = re.findall(r"[\w.'’\-]+", text)
    out, cur, first = set(), [], True
    for t in toks:
        if t[:1].isupper() and not first: cur.append(t)
        else:
            if len(cur) >= 1: out.add(" ".join(cur))
            cur = [t] if (t[:1].isupper() and not first) else []
        first = t.endswith((".", "!", "?"))
    if cur: out.add(" ".join(cur))
    out = {re.sub(r"[.,;:!?…]+$", "", x).strip() for x in out}
    return {x for x in out if len(x) >= 3 and not re.fullmatch(r"[A-Z]{1,3}", x)}


def load():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(PT, "public_test.jsonl"), encoding="utf-8")}
    asr = [(n, json.load(open(os.path.join(ROOT, p), encoding="utf-8"))) for n, p in ASR]
    labels = {json.loads(l)["id"]: json.loads(l)["tool_calls"] for l in open(LABELS, encoding="utf-8")}
    tab = json.load(open(SUBST, encoding="utf-8"))["table"]
    # bảng: key = cụm ASR nghe ra (hyp), value = [(cụm đúng trong nhãn, xác suất)]. Chỉ dùng cụm có tên riêng / từ ngoại
    # (chữ hoa hoặc f j w z) để tránh thay từ thường theo ngữ cảnh sai.
    # 13/11: TẮT. Bảng này đo trên held-out, key đã hạ chữ thường và phần lớn là từ thường theo ngữ cảnh ("không"↔"hông",
    # "sáng"↔"sớm"); áp lên public làm hỏng nhãn (thử 2 chiều đều vậy). Giữ code để bật lại nếu có bảng riêng cho tên riêng.
    subst = {}
    _ = tab
    # từ thường của train (bảo vệ khỏi bị thay bằng tên riêng na ná: mình→Minh, tuần→Tuấn)
    common = collections.Counter()
    for l in open(os.path.join(ROOT, "data", "train+public_test", "train", "train.jsonl"), encoding="utf-8"):
        s = json.loads(l)
        for t in [s["audio_text"]] + [h.get("audio_text", "") for h in s["history"] if h["role"] == "user"]:
            common.update(norm(t).split())
    common = {w for w, c in common.items() if c >= 3}
    return items, asr, labels, subst, common


class Builder:
    def __init__(self):
        self.items, self.asr, self.labels, self.subst, self.common = load()
        self.review, self.rows, self.fixcount = [], [], collections.Counter()

    def state(self, sid):
        return json.load(open(os.path.join(PT, "public_test_environment", sid, "state.json"), encoding="utf-8"))

    # ---------- chọn lời gốc ----------
    def pick_spoken(self, uid, region, vals):
        hyps = [(n, d[uid]) for n, d in self.asr if d.get(uid)]
        if not hyps: return None, "none", False
        normed = [norm(digits_to_words(t, region)) for _, t in hyps]
        forms = {f for k, v in vals for f in num_forms(v, k, region)}
        score = [sum(f" {f} " in f" {nt} " for f in forms) for nt in normed]
        cnt = collections.Counter(normed)
        disagree = len({tuple(NUM_SEQ.findall(nt)) for nt in normed}) > 1   # lệch nhau ở phần số
        best = None
        if forms and max(score) > 0 and score.count(max(score)) < len(score):
            best = score.index(max(score)); how = "num_match"
        else:
            top, c = cnt.most_common(1)[0]
            if c >= 2: best = normed.index(top); how = "majority"
            else: best = next(i for i, (n, _) in enumerate(hyps) if n == "e46f"); how = "e46f"
        return digits_to_words(hyps[best][1], region), how, disagree

    # ---------- sửa nhãn ----------
    def entity_vocab(self, sid, it, calls, upto_turn):
        st = self.state(sid)
        voc = set(); walk_names(st, voc)
        prio = set()
        for c in calls:
            for k, v in (c.get("arguments") or {}).items():
                if isinstance(v, str):
                    if re.fullmatch(r"[a-z]{2,6}_[0-9a-z]{2,8}", v):
                        e = find_by_id(st, v)
                        if e:
                            for nk in NAME_KEYS:
                                if isinstance(e.get(nk), str): prio.add(e[nk])
                    elif k in ("query", "symbol", "near", "origin", "destination", "stop", "contact", "name") and 2 <= len(v) <= 40:
                        prio.add(v)
        for t in it["history"][:upto_turn]:
            if t["role"] == "assistant" and t.get("text"): voc |= cap_ngrams(t["text"])
        prof = it["context"].get("user_profile", {})
        voc |= cap_ngrams(" ".join(str(prof.get(k, "")) for k in ("home_address", "work_address")))   # không lấy tên người dùng
        voc -= {prof.get("name", "")}
        return prio, voc - prio

    def fix_entities(self, text, prio, voc, fixes):
        toks = re.findall(r"\S+", text)
        bare = [re.sub(r"^[^\w]+|[^\w]+$", "", t) for t in toks]
        for ent, is_prio in [(e, True) for e in sorted(prio, key=len, reverse=True)] + [(e, False) for e in sorted(voc, key=len, reverse=True)]:
            et = ent.split(); L = len(et)
            if L == 1 and (len(ent) < 4 or ent.lower() in KIN) and not is_prio: continue
            best = (0, None)
            widths = {L} | ({L + 1, L - 1} if L >= 3 else set())
            for w in widths:
                for i in range(0, len(bare) - w + 1):
                    wt = bare[i:i + w]
                    if not all(wt): continue
                    sm = difflib.SequenceMatcher(None, [strip_acc(x) for x in wt], [strip_acc(x) for x in et])
                    # ghép token theo thứ tự rồi đo từng cặp
                    pairs = []
                    a = b = 0
                    while a < w and b < L:
                        pairs.append(difflib.SequenceMatcher(None, strip_acc(wt[a]), strip_acc(et[b])).ratio()); a += 1; b += 1
                    if not pairs: continue
                    whole = difflib.SequenceMatcher(None, strip_acc(" ".join(wt)), strip_acc(ent)).ratio()
                    score = min(whole, sum(pairs) / len(pairs))
                    if min(pairs) < 0.5: score = 0
                    if score > best[0]: best = (score, (i, w, " ".join(wt), min(pairs)))
            if not best[1]: continue
            r, (i, w, win, pmin) = best
            if win.lower() == ent.lower():
                if win != ent and (L >= 2 or is_prio or ent[0].isupper()):
                    self._replace(toks, bare, i, w, ent); fixes.append(("case", win, ent))
                continue
            if L == 1:
                # một từ: chỉ sửa khi là thực thể của nhãn và từ nghe được không phải từ thường
                if is_prio and r >= 0.8 and win.lower() not in self.common:
                    self._replace(toks, bare, i, w, ent); fixes.append(("ent", win, ent))
                elif is_prio and r >= 0.75: fixes.append(("maybe", win, ent))
                continue
            thr = 0.84 if is_prio else 0.88
            if r >= thr:
                self._replace(toks, bare, i, w, ent); fixes.append(("ent", win, ent))
            elif r >= 0.72 and is_prio:
                fixes.append(("maybe", win, ent))
        return " ".join(toks)

    @staticmethod
    def _replace(toks, bare, i, w, ent):
        lead = re.match(r"^[^\w]*", toks[i]).group(0); trail = re.search(r"[^\w]*$", toks[i + w - 1]).group(0)
        toks[i:i + w] = [lead + ent + trail]; bare[i:i + w] = [ent]

    def fix_subst(self, text, fixes):
        low = text
        for bad, good in self.subst.items():
            pat = re.compile(r"(?<!\w)" + re.escape(bad) + r"(?!\w)", re.I)
            if pat.search(low):
                low = pat.sub(good, low); fixes.append(("subst", bad, good))
        return low

    # ---------- biến thể ----------
    def variants(self, spoken, label, region, rng, names):
        out = []
        for k in range(2):
            s, lb, changed = spoken, label, []
            # số
            for m in list(NUM_SEQ.finditer(s))[::-1]:
                seq = m.group(0); after = s[m.end():m.end() + 12].strip().split()[:1]
                nxt = after[0].strip(".,!?") if after else ""
                if rng.random() < 0.6:
                    if re.search(r"nghìn|ngàn|triệu", seq): rep = N.money(rng, region)[0]
                    elif nxt == "giờ" or seq.endswith("rưỡi"): rep = re.sub(r" (sáng|trưa|chiều|tối)$", "", N.clock(rng, region)[0])
                    elif nxt == "độ": rep = N.temp(rng, region)[0].replace(" độ", "")
                    elif nxt == "cổ": rep = N.three(rng.choice([100, 200, 300, 500]))
                    elif nxt in ("vé", "khay", "gói", "túi", "hộp", "ký", "chai", "phần", "ly", "người", "ghế", "chỗ", "cái", "quả", "bó"):
                        rep = N.two(rng.choice([1, 2, 3, 4, 5, 6, 10]))
                    else: continue
                    if nxt == "giờ" and "giờ" in rep: s2 = s[:m.start()] + rep + s[m.end():].replace(" giờ", "", 1); lb2 = lb.replace(seq + " giờ", rep, 1)
                    else: s2 = s[:m.start()] + rep + s[m.end():]; lb2 = lb.replace(seq, rep, 1)
                    if lb2 != lb: s, lb = s2, lb2; changed.append("num")
            # tên người
            for nm in names:
                if nm in lb and nm in s and rng.random() < 0.7:
                    new = rng.choice(VN_NAMES); new = new if nm.split()[0].lower() in KIN else new.split()[-1]
                    s, lb = s.replace(nm, new), lb.replace(nm, new); changed.append("name")
            if changed and lb != label and s != spoken and (s, lb) not in [(a, b) for a, b, _ in out]:
                out.append((s, lb, "+".join(sorted(set(changed)))))
        return out

    # ---------- chạy ----------
    def run(self):
        for sid, it in self.items.items():
            region = REGION.get(it["context"]["user_profile"].get("city"), "Bac")
            turns, u = [], 0
            H = it["history"]
            for i, t in enumerate(H):
                if t["role"] != "user": continue
                u += 1
                nxt = next((j for j in range(i + 1, len(H)) if H[j]["role"] == "assistant"), None)
                turns.append((f"{sid}_u{u}", "history", (H[nxt].get("tool_calls") or []) if nxt is not None else [], (nxt + 1) if nxt is not None else i + 1))
            turns.append((sid, "final", self.labels.get(sid, []), len(H)))
            for uid, kind, calls, upto in turns:
                vals = label_values(calls)
                spoken, how, disagree = self.pick_spoken(uid, region, vals)
                if not spoken: continue
                fixes = []
                prio, voc = self.entity_vocab(sid, it, calls, upto)
                label = self.fix_subst(spoken, fixes)
                label = self.fix_entities(label, prio, voc, fixes)
                label = strip_punct(label)
                grp = []
                if NUM_SEQ.search(spoken): grp.append("num")
                if any(f[0] in ("ent", "case") for f in fixes): grp.append("name")
                if COND.search(spoken): grp.append("cond")
                if SELF.search(spoken): grp.append("selfcorr")
                names = [f[2] for f in fixes if f[0] in ("ent", "case") and f[2].split()[0].lower() in KIN or (f[0] in ("ent", "case") and len(f[2].split()) <= 2 and f[2][0].isupper())]
                row = {"id": f"pub_{uid}", "uid": uid, "sid": sid, "kind": kind, "spoken": spoken, "label": label,
                       "label_src": "auto" if any(f[0] != "maybe" for f in fixes) else "same", "pick": how, "region": region,
                       "group": "+".join(grp) or "plain", "split": "train", "src": "orig", "calls": calls, "fixes": fixes}
                self.rows.append(row)
                for f in fixes: self.fixcount[f[0]] += 1
                if disagree and how in ("majority", "e46f") and "num" in grp or any(f[0] == "maybe" for f in fixes):
                    self.review.append({"uid": uid, "why": ("số lệch, chưa có tham số nhãn để chốt" if disagree else "") + ("; thực thể nghi: " + "; ".join(f"{a}→{b}" for k, a, b in fixes if k == "maybe") if any(f[0] == "maybe" for f in fixes) else ""),
                                        "hyps": [d.get(uid) for _, d in self.asr], "spoken": spoken, "label": label})
                if kind == "final":
                    rng = random.Random(uid)
                    for k, (s, lb, ch) in enumerate(self.variants(spoken, label, region, rng, names)):
                        self.rows.append({**row, "id": f"pubv_{uid}_{k}", "spoken": s, "label": lb, "src": "variant", "group": row["group"] + "|var:" + ch, "fixes": []})
        self.write()

    def write(self):
        assert all(not re.search(r"\d", r["spoken"]) for r in self.rows), "spoken còn chữ số"
        with open(os.path.join(HERE, "tts_text_pub.jsonl"), "w", encoding="utf-8") as f:
            for r in self.rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(HERE, "review_needed.jsonl"), "w", encoding="utf-8") as f:
            for r in self.review: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = collections.Counter((r["src"], r["kind"]) for r in self.rows); rg = collections.Counter(r["region"] for r in self.rows if r["src"] == "orig")
        gp = collections.Counter(g for r in self.rows if r["src"] == "orig" for g in r["group"].split("+"))
        pk = collections.Counter(r["pick"] for r in self.rows if r["src"] == "orig")
        chars = sum(len(r["spoken"]) for r in self.rows)
        rep = ["# P1 — text ASR từ public (tự sinh bởi build_pub.py)", "",
               f"Tổng {len(self.rows)} dòng, {chars:,} ký tự lời đọc. Gốc: {sum(c[k] for k in c if k[0]=='orig')} (cuối {c[('orig','final')]}, history {c[('orig','history')]}); biến thể: {sum(c[k] for k in c if k[0]=='variant')}.",
               "", "| Miền | Gốc |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rg.most_common()] + \
              ["", "| Nhóm | Gốc |", "|---|---|"] + [f"| {k} | {v} |" for k, v in gp.most_common()] + \
              ["", "| Cách chọn lời gốc | Số câu |", "|---|---|"] + [f"| {k} | {v} |" for k, v in pk.most_common()] + \
              ["", "| Loại sửa nhãn | Số lần |", "|---|---|"] + [f"| {k} | {v} |" for k, v in self.fixcount.most_common()] + \
              ["", f"Cần xem tay: {len(self.review)} câu → `review_needed.jsonl`."]
        open(os.path.join(HERE, "REPORT_pub.md"), "w", encoding="utf-8").write("\n".join(rep) + "\n")
        print("\n".join(rep))


if __name__ == "__main__":
    Builder().run()
