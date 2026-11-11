"""Trộn lỗi ASR thật (A5b: asr_errors_<tag>.jsonl + asr_subst_<tag>.json) vào câu nói khi dựng train FC.

Hai mức: (1) câu có bản ASR sẵn (cùng sid, cùng lượt, câu chuẩn khớp) → thay NGUYÊN CÂU bằng một giả thuyết (chọn theo
trọng số); (2) còn lại → bảng thay thế theo cụm từ (1–3 từ) rút từ n-best, số cụm bị thay ~ số từ lỗi/câu lỗi của ASR
(≈2). Nhãn (tool call, tham số) giữ nguyên theo câu chuẩn. Lượt user trong history cũng bị nhiễu (xác suất riêng).
"""
import json
import re
import unicodedata

PUNCT = re.compile(r"[^\w\s]", re.U)


def _norm_tok(t):
    return PUNCT.sub("", unicodedata.normalize("NFC", t).lower())


def _norm(s):
    return " ".join(x for x in (_norm_tok(t) for t in (s or "").split()) if x)


class AsrNoise:
    def __init__(self, errors_path, subst_path, rng, p_item=0.4, p_hist=0.3, n_user_turns=None, max_src_len=3):
        """n_user_turns: {sid: số lượt user trong history của mẫu gốc} để đổi "final" → số thứ tự lượt."""
        self.rng, self.p_item, self.p_hist, self.max_src_len = rng, p_item, p_hist, max_src_len
        self.full = {}
        for l in open(errors_path, encoding="utf-8"):
            r = json.loads(l)
            if not r["hyps"]:
                continue
            turn = r["turn"]
            if turn == "final":
                if n_user_turns is None or r["sid"] not in n_user_turns:
                    continue
                turn = n_user_turns[r["sid"]] + 1
            self.full[(r["sid"], turn)] = (_norm(r["ref"]), r["hyps"])
        d = json.load(open(subst_path, encoding="utf-8"))
        self.stats = d["stats"]
        self.table = {k: v for k, v in d["table"].items() if k and len(k.split()) <= max_src_len}
        self.weight = {k: sum(w for _, w in v) for k, v in self.table.items()}
        self.n_full = self.n_table = self.n_clean = 0

    def _pick(self, pairs):
        tot = sum(w for _, w in pairs)
        x = self.rng.random() * tot
        for h, w in pairs:
            x -= w
            if x <= 0:
                return h
        return pairs[-1][0]

    def noise_text(self, text, key=None):
        """Trả về câu đã nhiễu, hoặc None nếu không có cách nào áp dụng."""
        if key in self.full and self.full[key][0] == _norm(text):
            self.n_full += 1
            return self._pick(self.full[key][1])
        toks = text.split()
        nt = [_norm_tok(t) for t in toks]
        cands = []                                   # (i, n, src)
        for i in range(len(toks)):
            for n in range(1, self.max_src_len + 1):
                if i + n > len(toks):
                    break
                src = " ".join(nt[i:i + n])
                if src in self.table:
                    cands.append((i, n, src))
        if not cands:
            return None
        k = self.rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        chosen, used = [], set()
        for _ in range(k * 4):
            if len(chosen) >= k or not cands:
                break
            c = self.rng.choices(cands, weights=[self.weight[s] for _, _, s in cands])[0]
            i, n, src = c
            if any(j in used for j in range(i, i + n)):
                cands.remove(c)
                continue
            used.update(range(i, i + n))
            chosen.append(c)
        if not chosen:
            return None
        out, i = [], 0
        for ci, n, src in sorted(chosen):
            out += toks[i:ci]
            dst = self._pick(self.table[src])
            if dst:
                first, last = toks[ci], toks[ci + n - 1]
                m = re.match(r"^(\W*)(.*?)(\W*)$", last, re.U)
                dst = m.group(1) + dst + m.group(3) if m else dst
                if first[:1].isupper():
                    dst = dst[:1].upper() + dst[1:]
                out.append(dst)
            i = ci + n
        out += toks[i:]
        self.n_table += 1
        return " ".join(out)

    def apply(self, item):
        """Bản sao item với audio_text (và lượt user trong history) đã nhiễu theo xác suất; item gốc không đổi."""
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
        nz = self.noise_text(item.get("audio_text") or "", (sid, k + 1))
        return dict(item, history=hist, audio_text=nz if nz is not None else item.get("audio_text", ""))

    def summary(self):
        return {"items_clean": self.n_clean, "utts_full_hyp": self.n_full, "utts_word_table": self.n_table,
                "p_item": self.p_item, "p_hist": self.p_hist, "asr": self.stats.get("asr")}
