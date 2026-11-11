"""Gộp LoRA Clef vào Qwen3.5-4B text-only, chạy ngoài Modal (cùng logic agent/clef/modal_merge.py).
    python merge_local.py --base <qwen35_4b_text> --adapter <run>/final --out <dir>
"""
import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.base)
ids = tok("Xin chào, hãy đặt báo thức lúc 7 giờ sáng mai.", return_tensors="pt").input_ids.to(dev)
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
n0 = sum(p.numel() for p in base.parameters())
m = PeftModel.from_pretrained(base, a.adapter)
with torch.no_grad():
    la = m(input_ids=ids).logits[0, -1].float()
m = m.merge_and_unload()
with torch.no_grad():
    lb = m(input_ids=ids).logits[0, -1].float()
n1 = sum(p.numel() for p in m.parameters())
print(f"tham số: gốc {n0 / 1e9:.4f}B, sau gộp {n1 / 1e9:.4f}B | max |logit diff| adapter vs gộp: "
      f"{(la - lb).abs().max().item():.4f} | argmax bằng nhau: {la.argmax().item() == lb.argmax().item()}")
if os.path.isdir(a.out):
    shutil.rmtree(a.out)
m.save_pretrained(a.out, safe_serialization=True)
for f in os.listdir(a.base):
    if f.endswith((".json", ".jinja", ".txt", ".model")) and not f.startswith(("config", "model.safetensors")):
        shutil.copy(os.path.join(a.base, f), a.out)
cfg = json.load(open(os.path.join(a.out, "config.json")))
print("architectures:", cfg.get("architectures"), "| files:", sorted(os.listdir(a.out))[:12])
print("→", a.out)
