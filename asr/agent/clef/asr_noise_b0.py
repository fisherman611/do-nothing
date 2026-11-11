"""Nhiễu ASR cho FC v3 (PLAN_72h_v3 A1): lỗi nghe thật của PhoWhisper-medium GỐC (B0) trên audio train + TTS, "pha loãng"
như M2 (agent/make_m2.py): mỗi cụm lệch giữ bản nghe sai với xác suất q, còn lại trả về câu chuẩn.

So với AsrNoise (bảng 441 cụm): phủ mọi loại từ (động từ, ngày giờ, số, từ nối điều kiện), không chỉ tên riêng và từ đệm.
- Tra theo VĂN BẢN câu chuẩn đã chuẩn hoá (không theo sid/lượt) → dùng được cho cả bản alias/iface (cùng câu) lẫn câu TTS.
- Mỗi lần gọi pha loãng lại → các bản của cùng một câu có lỗi khác nhau.
- Ép có ít nhất một lỗi; mẫu nhiều call (≥ 3) ưu tiên lỗi rơi vào nửa sau câu (vế hành động thứ 2–3).
- Câu không có bản nghe thật (vd bản phiên âm đổi chữ) → rơi về bảng cụm cũ; ngoài ra `table_mix` (0.3) số lần dùng luôn bảng cụm cũ
  (lỗi của ASR ĐÃ fine-tune: tên riêng, từ đệm) để pha kiểu lỗi gần ASR dùng lúc thi với kiểu lỗi rộng của model gốc.
Nhãn (tool call, tham số) luôn giữ theo câu chuẩn.
"""
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef.asr_noise import AsrNoise, _norm, _norm_tok       # noqa: E402
from agent.make_m2 import thin                                      # noqa: E402


def _late_error(text, noisy):
    a = [_norm_tok(t) for t in text.split() if _norm_tok(t)]
    b = [_norm_tok(t) for t in noisy.split() if _norm_tok(t)]
    half = len(a) // 2
    return any(op != "equal" and i1 >= half for op, i1, _, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes())


class AsrNoiseB0(AsrNoise):
    def __init__(self, b0_path, ref_manifests, errors_path, subst_path, rng, q=0.45, p_item=0.6, p_hist=0.15,
                 n_user_turns=None, max_tries=6, table_mix=0.3):
        super().__init__(errors_path, subst_path, rng, p_item=p_item, p_hist=p_hist, n_user_turns=n_user_turns)
        hyps = json.load(open(b0_path, encoding="utf-8"))["hyps"]
        self.idx = {}
        for path in ref_manifests:
            for x in json.load(open(path, encoding="utf-8")):
                h = hyps.get(x["uid"])
                if x.get("split") == "train" and h:
                    self.idx.setdefault(_norm(x["text"]), []).append(h)
        self.q, self.max_tries, self.table_mix = q, max_tries, table_mix
        self.n_b0 = self.n_b0_late = self.n_fallback = self.n_mix_table = 0

    def noise_text(self, text, key=None, late=False):
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
        nz = self.noise_text(item.get("audio_text") or "", (sid, k + 1), late=late)
        return dict(item, history=hist, audio_text=nz if nz is not None else item.get("audio_text", ""))

    def summary(self):
        return {**super().summary(), "source": "b0", "q": self.q, "utts_b0": self.n_b0, "utts_b0_late": self.n_b0_late,
                "utts_fallback_table": self.n_fallback, "utts_mix_table": self.n_mix_table, "table_mix": self.table_mix, "b0_texts": len(self.idx)}
