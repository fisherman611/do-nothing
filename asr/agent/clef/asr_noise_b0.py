"""Nhiễu ASR cho FC v3 (PLAN_72h_v3 A1): lỗi nghe thật của PhoWhisper-medium GỐC (B0) trên audio train + TTS, "pha loãng"
như M2 (agent/make_m2.py): mỗi cụm lệch giữ bản nghe sai với xác suất q, còn lại trả về câu chuẩn.

So với AsrNoise (bảng 441 cụm): phủ mọi loại từ (động từ, ngày giờ, số, từ nối điều kiện), không chỉ tên riêng và từ đệm.
- Tra theo VĂN BẢN câu chuẩn đã chuẩn hoá (không theo sid/lượt) → dùng được cho cả bản alias/iface (cùng câu) lẫn câu TTS.
- Mỗi lần gọi pha loãng lại → các bản của cùng một câu có lỗi khác nhau.
- Ép có ít nhất một lỗi; mẫu nhiều call (≥ 3) ưu tiên lỗi rơi vào nửa sau câu (vế hành động thứ 2–3).
- Câu không có bản nghe thật (vd bản phiên âm đổi chữ) → rơi về bảng cụm cũ; ngoài ra `table_mix` (0.3) số lần dùng luôn bảng cụm cũ
  (lỗi của ASR ĐÃ fine-tune: tên riêng, từ đệm) để pha kiểu lỗi gần ASR dùng lúc thi với kiểu lỗi rộng của model gốc.
- Chốt chặn số: bản nhiễu làm đổi GIÁ TRỊ số (giờ, số tiền, số lượng; chữ ↔ số coi là như nhau) bị loại — nhãn giữ giá trị
  sạch nên mẫu như "bảy giờ" → "ba giờ" chỉ dạy model đoán bừa; thử lại tới `max_tries`, hết lượt thì giữ câu sạch.
- Chốt chặn theo NHÃN (kiểm toán 11/11, NOTES §1o): nhãn giữ theo câu sạch, nên bản nhiễu chỉ được nhận khi câu nhiễu KHÔNG
  đọc thành một lệnh hợp lệ khác và không nát hơn ASR dùng lúc thi (e46f, đo trên 405 câu held-out). `noise_reject` loại khi:
  số đổi giá trị · từ chỉ thời gian đổi (mai/nay/thứ/sáng/tối…) · từ phủ định / tự sửa đổi (không/khỏi/đừng/chưa/thôi/nhầm/khoan)
  · mất/thêm "nếu", "trừ" · WER của câu > 0.45 (mức tệ nhất của e46f là 0.47, p95 0.38) · một thực thể của nhãn (giá trị tham
  số chữ có nguyên văn trong câu sạch) biến thành cụm không còn gần âm (độ giống < 0.7). Phương ngữ quy về một dạng trước khi so
  (ni=nay, hông=không, bữa=hôm). Bị loại thì thử lại tới `max_tries`, hết lượt thì giữ câu sạch.
Nhãn (tool call, tham số) luôn giữ theo câu chuẩn.
"""
import difflib
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef.asr_noise import AsrNoise, _norm, _norm_tok       # noqa: E402
from agent.make_m2 import thin                                      # noqa: E402
from agent.geo_fix import canon as _canon, fold as _fold           # noqa: E402


def _late_error(text, noisy):
    a = [_norm_tok(t) for t in text.split() if _norm_tok(t)]
    b = [_norm_tok(t) for t in noisy.split() if _norm_tok(t)]
    half = len(a) // 2
    return any(op != "equal" and i1 >= half for op, i1, _, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes())


_UNITS = {"một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10}


def _nums(text):
    """Dãy giá trị số (đã sắp) trong câu; đọc cả chữ (0–99: 'ba mươi lăm', 'mười hai', 'hai chục') lẫn số."""
    t = re.findall(r"\w+", text.lower())
    out, i = [], 0
    while i < len(t):
        w = t[i]
        if w.isdigit():
            out.append(int(w)); i += 1; continue
        if w in _UNITS:
            v, j = _UNITS[w], i + 1
            if w == "mười" and j < len(t) and t[j] in _UNITS and t[j] != "mười":
                v, j = 10 + _UNITS[t[j]], j + 1
            elif j < len(t) and t[j] in ("mươi", "chục"):
                v, j = v * 10, j + 1
                if j < len(t) and t[j] in _UNITS and t[j] != "mười":
                    v, j = v + _UNITS[t[j]], j + 1
            out.append(v); i = j; continue
        i += 1
    return sorted(out)


_DIAL = {"ni": "nay", "hông": "không", "hổng": "không", "hong": "không", "hổm": "hôm", "bữa": "hôm", "mơi": "mai", "chừ": "giờ"}
_TIME = set("nay mai mốt kia qua hôm sáng trưa chiều tối đêm khuya thứ chủ nhật tuần tháng ngày giờ phút tiếng rưỡi kém sau trước tới".split())
_NEG = set("không khỏi đừng chưa chẳng chả thôi nhầm khoan lộn".split())
_COND = {"nếu", "trừ"}
MAX_UTT_WER, MIN_ENT_SIM = 0.45, 0.7


def _words(text):
    return [_DIAL.get(w, w) for w in re.findall(r"\w+", unicodedata.normalize("NFC", text.lower()))]


def _cls(words, vocab):
    return sorted(w for w in words if w in vocab)


def _utt_wer(a, b):
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal") / max(1, len(a))


def _canon_syls(text):
    return [_canon(w) for w in re.findall(r"[a-z0-9]+", _fold(text))]


def _ent_sim(ent, noisy):
    """Độ giống theo âm (0..1) giữa thực thể và cửa sổ gần nhất trong câu nhiễu. So trên chuỗi âm tiết đã quy đổi, BỎ khoảng
    trắng (từ nước ngoài hay bị phiên âm tách âm tiết: Nexora Labs → Nét xô ra Láp), cửa sổ dài 1..n+3 âm tiết."""
    e, h = _canon_syls(ent), _canon_syls(noisy)
    if not e or not h:
        return 0.0
    es, best = "".join(e), 0.0
    for n in range(1, len(e) + 4):
        for i in range(max(1, len(h) - n + 1)):
            w = "".join(h[i:i + n])
            if abs(len(w) - len(es)) <= max(2, len(es) // 2):
                best = max(best, difflib.SequenceMatcher(None, es, w, autojunk=False).ratio())
    return best


def label_entities(item):
    """Giá trị tham số chữ của nhãn (không phải id / ngày giờ) có nguyên văn (bỏ dấu) trong câu sạch."""
    fr = f" {' '.join(re.findall(r'[a-z0-9]+', _fold(item.get('audio_text') or '')))} "
    out = []
    for c in (item.get("label") or {}).get("tool_calls") or []:
        for k, v in (c.get("arguments") or {}).items():
            if not isinstance(v, str) or k.endswith("id") or re.match(r"\d{4}-\d\d", v):
                continue
            fv = " ".join(re.findall(r"[a-z0-9]+", _fold(v)))
            if len(fv) >= 3 and f" {fv} " in fr and fv not in out:
                out.append(fv)
    return out


def noise_reject(clean, noisy, ents=()):
    """None nếu bản nhiễu dùng được; ngược lại tên lý do loại."""
    if _nums(clean) != _nums(noisy):
        return "num"
    a, b = _words(clean), _words(noisy)
    if _cls(a, _TIME) != _cls(b, _TIME):
        return "time"
    if _cls(a, _NEG) != _cls(b, _NEG):
        return "neg"
    if _cls(a, _COND) != _cls(b, _COND):
        return "cond"
    if _utt_wer(a, b) > MAX_UTT_WER:
        return "wer"
    fh = f" {' '.join(re.findall(r'[a-z0-9]+', _fold(noisy)))} "
    for e in ents:
        if f" {e} " not in fh and (len(e.replace(" ", "")) <= 4 or _ent_sim(e, noisy) < MIN_ENT_SIM):
            return "ent"                     # thực thể ngắn (Lan, SJC, E5): phải còn nguyên văn (bỏ dấu)
    return None


class AsrNoiseB0(AsrNoise):
    def __init__(self, b0_path, ref_manifests, errors_path, subst_path, rng, q=0.45, p_item=0.6, p_hist=0.15,
                 n_user_turns=None, max_tries=10, table_mix=0.3):
        super().__init__(errors_path, subst_path, rng, p_item=p_item, p_hist=p_hist, n_user_turns=n_user_turns)
        hyps = json.load(open(b0_path, encoding="utf-8"))["hyps"]
        self.idx = {}
        for path in ref_manifests:
            for x in json.load(open(path, encoding="utf-8")):
                h = hyps.get(x["uid"])
                if x.get("split") == "train" and h:
                    self.idx.setdefault(_norm(x["text"]), []).append(h)
        self.q, self.max_tries, self.table_mix = q, max_tries, table_mix
        self.n_b0 = self.n_b0_late = self.n_fallback = self.n_mix_table = self.n_gave_up = 0
        self.rejects = {}

    def noise_text(self, text, key=None, late=False, ents=()):
        for _ in range(self.max_tries):
            nz = self._noise_text(text, key, late)
            if nz is None:
                return None
            why = noise_reject(text, nz, ents)
            if why is None:
                return nz
            self.rejects[why] = self.rejects.get(why, 0) + 1
        self.n_gave_up += 1
        return None

    def _noise_text(self, text, key=None, late=False):
        cands = self.idx.get(_norm(text))
        if cands and self.rng.random() < self.table_mix:     # kiểu lỗi của ASR đã fine-tune (tên, từ đệm) — gần ASR thật hơn
            self.n_mix_table += 1
            nz = super().noise_text(text, key)
            if nz is not None:
                return nz
        if cands:
            best = None
            for _ in range(self.max_tries):
                s, _, kept = thin(text, self.rng.choice(cands), self.q, self.rng)
                if kept == 0:
                    continue
                best = best or s
                if not late or _late_error(text, s):
                    best = s
                    self.n_b0 += 1
                    self.n_b0_late += bool(late)
                    return s
            if best is not None:
                self.n_b0 += 1
                return best
        self.n_fallback += 1
        return super().noise_text(text, key)

    def apply(self, item):
        if self.rng.random() >= self.p_item:
            self.n_clean += 1
            return item
        sid = item["id"].split("#")[0]
        hist, k = [], 0
        for t in item["history"]:
            if t["role"] == "user":
                k += 1
                if t.get("audio_text") and self.rng.random() < self.p_hist:
                    nz = self.noise_text(t["audio_text"], (sid, k))
                    if nz is not None:
                        t = dict(t, audio_text=nz)
            hist.append(t)
        late = len(item.get("label", {}).get("tool_calls") or []) >= 3
        nz = self.noise_text(item.get("audio_text") or "", (sid, k + 1), late=late, ents=label_entities(item))
        return dict(item, history=hist, audio_text=nz if nz is not None else item.get("audio_text", ""))

    def summary(self):
        return {**super().summary(), "source": "b0", "q": self.q, "utts_b0": self.n_b0, "utts_b0_late": self.n_b0_late,
                "utts_fallback_table": self.n_fallback, "utts_mix_table": self.n_mix_table, "table_mix": self.table_mix, "rejects": dict(self.rejects), "utts_gave_up_clean": self.n_gave_up, "b0_texts": len(self.idx)}
