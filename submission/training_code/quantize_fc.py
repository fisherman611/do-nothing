"""SmoothQuant + INT8 (W8A8) cho FC Qwen3.5-4B (đã gộp LoRA), xuất định dạng compressed-tensors để vLLM chạy kernel INT8.

    PYTHONPATH=<llmc_pkg> python quantize_fc.py --model saved_models/fc --out saved_models/fc_w8a8 \
        --data asr/agent/clef/data/fmtA_v5/train.jsonl --n 128 --max-len 3072

- SmoothQuant (strength 0.5): dịch outlier của activation sang weight qua norm đứng trước (input_layernorm, post_attention_layernorm).
  Qwen3.5 dùng RMSNorm dạng (1 + weight): bản llmcompressor 0.15 chia thẳng `weight` cho hệ số làm mượt nên sai với dạng này;
  đã vá `_apply_smoothing` (xem README) để giữ (1 + w') = (1 + w) / s.
- Mọi lớp tiêu thụ đầu ra của một norm đều phải nằm trong `balance_layers` (kể cả in_proj_a / in_proj_b), nếu không hàm số bị đổi.
- Lượng tử hóa: weight INT8 đối xứng theo kênh, activation INT8 theo token (động). Giữ bf16: lm_head, embed, in_proj_a, in_proj_b.
- Calibration: các prompt thật của FC (train fmtA_v5, seed cố định), không dùng public_test / private_test.
"""
import argparse
import json
import random

import torch
from torch.utils.data import DataLoader
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    random.shuffle(rows)
    texts = [r["text"] for r in rows[: a.n]]
    enc = [tok(t, max_length=a.max_len, truncation=True, add_special_tokens=False) for t in texts]
    # DataLoader dựng sẵn (batch 1, không đệm): oneshot dùng thẳng, bỏ qua `datasets` (lỗi pickle với pyarrow mới)
    ds = DataLoader([{"input_ids": torch.tensor([e["input_ids"]]), "attention_mask": torch.tensor([e["attention_mask"]])}
                     for e in enc], batch_size=None, shuffle=False)

    # mapping theo TỪNG lớp (mô hình xen kẽ attention đầy đủ và Gated DeltaNet nên regex chung khớp nhầm giữa các lớp)
    lt = model.config.layer_types
    mappings = []
    for i, kind in enumerate(lt):
        L = rf"re:model\.layers\.{i}\."
        if kind == "full_attention":
            bal = [L + r"self_attn\.q_proj", L + r"self_attn\.k_proj", L + r"self_attn\.v_proj"]
        else:   # linear_attention: mọi lớp đọc đầu ra input_layernorm phải nằm trong balance
            bal = [L + r"linear_attn\.in_proj_qkv", L + r"linear_attn\.in_proj_z",
                   L + r"linear_attn\.in_proj_a", L + r"linear_attn\.in_proj_b"]
        mappings.append([bal, L + "input_layernorm"])
        mappings.append([[L + r"mlp\.gate_proj", L + r"mlp\.up_proj"], L + "post_attention_layernorm"])
    print("mappings:", len(mappings), "lớp:", {k: lt.count(k) for k in set(lt)}, flush=True)
    recipe = [
        SmoothQuantModifier(smoothing_strength=a.strength, mappings=mappings),
        QuantizationModifier(targets="Linear", scheme="W8A8",
                             ignore=["lm_head", "re:.*in_proj_a", "re:.*in_proj_b", "re:.*embed_tokens"]),
    ]
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.max_len, num_calibration_samples=len(texts),
            pipeline="basic")
    model.save_pretrained(a.out, save_compressed=True)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
