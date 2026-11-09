"""Token hoá sẵn dữ liệu Clef-token (build_data.py) thành dataset cho LLaMA-Factory (`tokenized_path`).

Dùng lại `train.encode` nên vị trí loss giống hệt train.py. Mỗi dòng: input_ids, attention_mask, length và dữ liệu
loss Clef (đọc trong lf_train.py):
  ch_pos / ch_allowed / ch_gold / ch_kind : vị trí dự đoán, id nhãn hợp lệ, chỉ số nhãn vàng, loại (0 q1, 1 yn, 2 opt)
  lm_pos / lm_tgt                         : vị trí dự đoán và token đích của giá trị tham số tự do

    python agent/clef/lf_prep.py --src agent/clef/data/train.jsonl --out /data/lf/train [--select first|longest --n 64]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef.build_data import TOKENIZER  # noqa: E402
from agent.clef.train import encode           # noqa: E402

KIND = {"q1": 0, "yn": 1, "opt": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--select", default="all", choices=["all", "first", "longest"])
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    if a.select == "first":
        rows = rows[: a.n]
    elif a.select == "longest":            # smoke bộ nhớ: lấy ví dụ dài nhất (theo ký tự) còn dưới max-len
        rows = sorted(rows, key=lambda r: -len(r["text"]))
    cols = {k: [] for k in ("input_ids", "attention_mask", "length", "ch_pos", "ch_allowed", "ch_gold", "ch_kind",
                            "lm_pos", "lm_tgt")}
    dropped = 0
    for r in rows:
        e = encode(r, tok, a.max_len)
        if not e or not (e[1] or e[2]):
            dropped += 1
            continue
        ids, choices, lms = e
        cols["input_ids"].append(ids)
        cols["attention_mask"].append([1] * len(ids))
        cols["length"].append(len(ids))
        cols["ch_pos"].append([c[0] for c in choices])
        cols["ch_allowed"].append([c[1] for c in choices])
        cols["ch_gold"].append([c[2] for c in choices])
        cols["ch_kind"].append([KIND[c[3]] for c in choices])
        cols["lm_pos"].append([p for p, _ in lms])
        cols["lm_tgt"].append([t for _, t in lms])
        if a.select == "longest" and len(cols["length"]) >= a.n:
            break
    n = len(cols["length"])
    DatasetDict({"train": Dataset.from_dict(cols)}).save_to_disk(a.out)
    L = sorted(cols["length"])
    print(f"{a.src} → {a.out}: {n} ví dụ (bỏ {dropped} quá dài/rỗng), token tb {sum(L) // max(1, n)}, "
          f"max {L[-1] if L else 0}, tổng {sum(L)}", flush=True)


if __name__ == "__main__":
    main()
