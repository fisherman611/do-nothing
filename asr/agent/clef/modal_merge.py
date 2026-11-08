"""Gộp LoRA Clef-token vào Qwen3.5-4B bản cắt và lưu thành model đầy đủ trên volume `hf-cache`, để vLLM serve như
một model thường (không phụ thuộc vào việc vLLM có hỗ trợ LoRA trên lớp Gated DeltaNet hay không).

    modal run agent/clef/modal_merge.py --adapter /data/runs/clef_p2_e1/final --name qwen35_4b_clef_p2
    → /hf/qwen35_4b_clef_p2 ; thêm vào MODELS của agent/modal_llm_server.py rồi deploy lại.

Kiểm tra sau khi gộp: số tham số bằng bản gốc (LoRA không thêm tham số), logit token cuối của model gộp ≈ model gốc +
adapter (max |diff| in ra).
"""
import json
import os
import shutil

import modal

app = modal.App("lifemate-clef-merge")
hf = modal.Volume.from_name("hf-cache")
vol = modal.Volume.from_name("clef-data")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.31.0", "peft", "flash-linear-attention")
         .env({"HF_HOME": "/hf"}))
BASE = "/hf/qwen35_4b_text"


@app.function(image=image, gpu="L40S", volumes={"/hf": hf, "/data": vol}, timeout=3600)
def merge(adapter: str, name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = f"/hf/{name}"
    tok = AutoTokenizer.from_pretrained(BASE)
    ids = tok("Xin chào, hãy đặt báo thức lúc 7 giờ sáng mai.", return_tensors="pt").input_ids.cuda()
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    n0 = sum(p.numel() for p in base.parameters())
    m = PeftModel.from_pretrained(base, adapter)
    with torch.no_grad():
        a = m(input_ids=ids).logits[0, -1].float()
    m = m.merge_and_unload()
    with torch.no_grad():
        b = m(input_ids=ids).logits[0, -1].float()
    n1 = sum(p.numel() for p in m.parameters())
    print(f"tham số: gốc {n0 / 1e9:.4f}B, sau gộp {n1 / 1e9:.4f}B | max |logit diff| adapter vs gộp: "
          f"{(a - b).abs().max().item():.4f} | argmax bằng nhau: {a.argmax().item() == b.argmax().item()}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    m.save_pretrained(out, safe_serialization=True)
    for f in os.listdir(BASE):     # tokenizer, chat template, generation config
        if f.endswith((".json", ".jinja", ".txt", ".model")) and not f.startswith(("config", "model.safetensors")):
            shutil.copy(os.path.join(BASE, f), out)
    cfg = json.load(open(os.path.join(out, "config.json")))
    print("architectures:", cfg.get("architectures"), "| files:", sorted(os.listdir(out))[:12])
    hf.commit()
    return out


@app.local_entrypoint()
def main(adapter: str, name: str):
    print(merge.remote(adapter, name))
