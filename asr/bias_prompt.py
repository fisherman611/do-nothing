"""Contextual biasing v2 cho PhoWhisper: dựng prompt từ nguồn có sẵn lúc test (xem PLAN_ASR_bias_v2.md).

Dùng chung cho manifest (make_manifest_v2.py), train/eval (modal_finetune_phowhisper.py) và infer trong Docker sau
này, để train và test dựng prompt giống hệt nhau. Chỉ đọc các trường của `train.jsonl` / `public_test.jsonl`
(context, history, label khi train); không đọc `*_environment/`.

Prompt (sau <|startofprev|>), mỗi phần một ngân sách token riêng:
    [hội thoại: transcript lượt user trước + text assistant, giữ đuôi] [hồ sơ người dùng] [danh sách tên]
Danh sách tên lúc test = tên trong `tool_results` của history (+ tùy chọn: mục lexicon train chọn theo app đã cài
và độ giống với bản nháp ASR). Lúc train có thể thêm tên trong label + tên gây nhiễu (chỉ train).
"""
import re
import unicodedata

# Trường trong tool_results mang tên riêng mà người dùng hay nói ra (thống kê trên train). Bỏ text dài
# (content, subject, snippet, message), id, giờ, số.
ENT_FIELDS = ("name", "title", "members", "channels", "channel", "artist", "location", "provider", "from_name",
              "from", "destination", "origin", "airline", "station", "bank", "assignee", "carrier", "author")
BUDGET = {"conv": 110, "profile": 40, "ents": 70}   # 1 (SOP) + 220 + 4 token ép + 200 token sinh = 425 < 448
SOP = "<|startofprev|>"
_SENT_END = re.compile(r"[.?!…]\s*$")


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- trích dữ liệu từ một dòng jsonl ----------
def ents_of(results, app):
    """[[app, field, value], ...] theo thứ tự xuất hiện, từ một list tool_results."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ENT_FIELDS and isinstance(v, str):
                    out.append([app, k, v])
                elif k in ENT_FIELDS and isinstance(v, list):
                    out.extend([app, k, s] for s in v if isinstance(s, str))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(results)
    return [e for e in out if e[2].strip() and not re.match(r"^[\d\W]", e[2])]


def _turn_ents(turn):
    calls, res = turn.get("tool_calls") or [], turn.get("tool_results") or []
    out = []
    for c, r in zip(calls, res):
        out += ents_of(r, c["name"].split(".")[0])
    return out


def profile_text(ctx):
    p = ctx.get("user_profile") or {}
    parts = [p.get(k, "") for k in ("name", "home_address", "work_address")]
    city = p.get("city", "")
    if city and not any(city in x for x in parts):
        parts.append(city)
    return ", ".join(x for x in parts if x)


def items_from_row(row, split=None, with_label=False):
    """Mỗi audio của mẫu (lượt user trong history + lượt cuối) thành một utt.

    prev: [[role, text, uid_or_None], ...] như manifest v1. hist_ents: tên trong tool_results các lượt assistant
    trước utt (có lúc test). label_ents (chỉ khi with_label): tên trong kết quả của lượt trả lời utt đó
    (lượt assistant ngay sau với utt history, `label.tool_results` với lượt cuối)."""
    ctx = row["context"]
    base = dict(sid=row["id"], split=split, profile=profile_text(ctx), apps=sorted(ctx.get("installed_apps", {})))
    out, prev, hist = [], [], []
    turns = row.get("history", [])
    for i, t in enumerate(turns):
        if t["role"] == "user":
            uid = t["audio"].split("/")[-1][:-4]
            it = dict(base, uid=uid, audio=t["audio"], text=t.get("audio_text"), kind="history",
                      prev=list(prev), hist_ents=list(hist))
            if with_label:
                nxt = turns[i + 1] if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant" else {}
                it["label_ents"] = _turn_ents(nxt)
            out.append(it)
            prev.append(["user", t.get("audio_text"), uid])
        else:
            prev.append(["assistant", t.get("text", ""), None])
            hist += _turn_ents(t)
    it = dict(base, uid=row["id"], audio=row["audio"], text=row.get("audio_text"), kind="final",
              prev=list(prev), hist_ents=list(hist))
    if with_label:
        lab = row["label"]
        it["label_ents"] = [e for c, r in zip(lab["tool_calls"], lab["tool_results"])
                            for e in ents_of(r, c["name"].split(".")[0])]
    out.append(it)
    return out


def build_lexicon(rows):
    """{app: {field: [value, ...]}} từ tool_results (history + label) của các dòng train (chỉ split train)."""
    lex = {}
    for row in rows:
        ents = [e for t in row.get("history", []) if t["role"] == "assistant" for e in _turn_ents(t)]
        lab = row.get("label") or {}
        ents += [e for c, r in zip(lab.get("tool_calls", []), lab.get("tool_results", []))
                 for e in ents_of(r, c["name"].split(".")[0])]
        for app, field, v in ents:
            vals = lex.setdefault(app, {}).setdefault(field, [])
            if v not in vals:
                vals.append(v)
    return lex


# ---------- dựng prompt ----------
def _dedup_keep_last(vals):
    seen, out = set(), []
    for v in reversed(vals):
        k = norm(v)
        if k and k not in seen:
            seen.add(k)
            out.append(v)
    return out[::-1]


def conv_text(it, hyps=None):
    """Như prev_text_of của v1: hyps (uid -> transcript ASR) thay nhãn thật lượt user (điều kiện thi)."""
    parts = []
    for role, text, uid in it.get("prev", []):
        if role == "user" and hyps is not None:
            text = hyps.get(uid, "")
        if text:
            parts.append(text.strip())
    return " ".join(parts)


def _ids(tok, text):
    return tok(" " + text.strip(), add_special_tokens=False).input_ids if text and text.strip() else []


def _fit_list(tok, vals, budget):
    """Giữ các tên từ cuối danh sách (gần nhất / quan trọng nhất) cho tới khi hết ngân sách token."""
    keep, used = [], 0
    for v in reversed(vals):
        n = len(_ids(tok, v + ","))
        if used + n > budget:
            continue
        keep.append(v)
        used += n
    return keep[::-1]


def build_prompt(tok, it, hyps=None, extra_ents=(), train=False, rng=None, cfg=None):
    """Trả về (prompt_ids, ents_used). Eval/test (train=False): tất định, chỉ dùng hội thoại (hyps nếu có),
    hồ sơ, hist_ents và extra_ents (lexicon). Train: drop ngẫu nhiên từng phần, thêm label_ents với xác suất
    cfg['label_ent_prob'], tên gây nhiễu từ cfg['pool'] (field -> values, chỉ split train)."""
    cfg = cfg or {}
    conv = conv_text(it, hyps)
    prof = it.get("profile", "")
    ents = [e[2] for e in it.get("hist_ents", [])] + list(extra_ents)
    if train:
        sd = cfg.get("section_drop", 0.3)
        if rng.random() < sd:
            prof = ""
        lab = it.get("label_ents") or []
        pool = cfg.get("pool") or {}
        r = rng.random()
        add, n_d = [], 0
        if r < cfg.get("all_distract_prob", 0.0):               # V3: chỉ tên nhiễu
            n_d = rng.randint(3, cfg.get("n_distract_max", 10))
        elif r < cfg.get("all_distract_prob", 0.0) + cfg.get("label_ent_prob", 0.0) and lab:   # V2
            add = [e[2] for e in lab]
            n_d = rng.randint(3, cfg.get("n_distract_max", 10))
        fields = [e[1] for e in lab] or list(pool)
        for _ in range(n_d):
            vals = pool.get(rng.choice(fields)) if fields else None
            if vals:
                add.append(rng.choice(vals))
        ents = ents + add
        if add:
            rng.shuffle(ents)
        if rng.random() < sd:
            ents = []
    ents = _fit_list(tok, _dedup_keep_last(ents), BUDGET["ents"])
    ids = _ids(tok, conv)[-BUDGET["conv"]:]
    if prof:
        ids += _ids(tok, prof + ".")[:BUDGET["profile"]]
    if ents:
        ids += _ids(tok, ", ".join(ents) + ".")
    if not ids:
        return [], ents
    return [tok.convert_tokens_to_ids(SOP)] + ids, ents


# ---------- lexicon lúc test ----------
def _grams(s, n=3):
    s = f" {norm(s)} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def select_lexicon(tok, lex, apps, draft, budget=None, min_sim=0.5, exclude=()):
    """Chọn mục lexicon của các app đã cài giống nhất với một đoạn trong bản nháp ASR (Dice trên char-3gram,
    so với cửa sổ cùng số từ). Trả về list tên, đã sắp theo độ giống tăng dần (tên giống nhất ở cuối)."""
    words = norm(draft).split()
    if not words:
        return []
    ex = {norm(e) for e in exclude}
    scored = []
    for app in apps:
        for field, vals in (lex.get(app) or {}).items():
            for v in vals:
                nv = norm(v)
                if len(nv) < 3 or nv in ex:
                    continue
                k = len(nv.split())
                g = _grams(nv)
                best = 0.0
                for i in range(max(1, len(words) - k + 1)):
                    w = _grams(" ".join(words[i:i + k]))
                    best = max(best, 2 * len(g & w) / (len(g) + len(w)))
                if best >= min_sim:
                    scored.append((best, v))
    scored.sort(key=lambda x: x[0])
    vals = _dedup_keep_last([v for _, v in scored])
    return _fit_list(tok, vals, budget or BUDGET["ents"])


# ---------- chỉ số ----------
def name_tokens(ref):
    """Từ viết hoa không đứng đầu câu (proxy tên riêng). Bỏ từ đầu chuỗi và từ ngay sau . ? ! …"""
    out = []
    for i, m in enumerate(re.finditer(r"\w+", ref)):
        w = m.group()
        if i == 0 or not w[:1].isupper() or _SENT_END.search(ref[:m.start()]):
            continue
        out.append(norm(w))
    return out


def name_stats(refs, hyps, prompt_ents):
    """Lỗi tên riêng (tổng, tách theo tên có/không có trong danh sách tên của prompt) và chép nhầm
    (từ viết hoa của danh sách tên xuất hiện trong hyp mà không có trong ref)."""
    s = {"names": 0, "name_err": 0, "in_prompt": 0, "in_prompt_err": 0, "false_copy": 0, "utts": len(refs)}
    for ref, hyp, ents in zip(refs, hyps, prompt_ents):
        hw, rw = set(norm(hyp).split()), set(norm(ref).split())
        pw = {norm(w) for e in ents for w in re.findall(r"\w+", e) if w[:1].isupper()}   # phần tên riêng
        for n in name_tokens(ref):
            s["names"] += 1
            bad = n not in hw
            s["name_err"] += bad
            if n in pw:
                s["in_prompt"] += 1
                s["in_prompt_err"] += bad
        s["false_copy"] += len({w for w in hw & pw if w not in rw and len(w) > 1})
    s["name_err_rate"] = s["name_err"] / max(1, s["names"])
    return s
