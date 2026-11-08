"""Strip Qwen3.5-4B to a text-only checkpoint (drop the vision encoder and the MTP head) so that
PhoWhisper-large (1.54B) + LLM stays under the 6B total of the competition.

Runs on Modal (CPU, the `hf-cache` volume that the vLLM servers read from):

    modal run asr/agent/strip_qwen35.py                 # writes /hf/qwen35_4b_text on volume hf-cache
    modal run asr/agent/strip_qwen35.py --check-only    # only print architectures/param counts

What is kept: every tensor under `model.language_model.*`, renamed to `model.*` (the layout of the text-only
`Qwen3_5ForCausalLM`). Dropped: `model.visual.*` (0.33B) and `mtp.*` (0.12B). lm_head is tied to the embeddings
in this checkpoint, so nothing else is needed.

Verification done here (CPU, bf16): (1) load the stripped model with transformers, no missing/unexpected keys;
(2) logits of the stripped model equal the logits of the original `Qwen3_5ForConditionalGeneration` on a text
prompt (max abs diff printed); (3) parameter count of the result.
"""
import json
import os
import shutil

import modal

REPO = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUT = "/hf/qwen35_4b_text"
KEEP = "model.language_model."

app = modal.App("lifemate-strip-qwen35")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.31.0", "huggingface_hub[hf_transfer]", "safetensors")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"}))


@app.function(image=image, volumes={"/hf": hf_cache}, cpu=8, memory=48 * 1024, timeout=3600)
def strip(check_only: bool = False):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    src = snapshot_download(REPO, revision=REV)
    print("source:", src)
    print("vLLM text-only arch supported:", _vllm_has("Qwen3_5ForCausalLM"))

    if not check_only:
        os.makedirs(OUT, exist_ok=True)
        index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
        n_param, new_map, dropped = 0, {}, {}
        for shard in sorted(set(index["weight_map"].values())):
            t = load_file(os.path.join(src, shard))
            keep = {}
            for k, v in t.items():
                if k.startswith(KEEP):
                    nk = "model." + k[len(KEEP):]
                    keep[nk] = v.contiguous()
                    n_param += v.numel()
                    new_map[nk] = shard
                else:
                    pfx = ".".join(k.split(".")[:2])
                    dropped[pfx] = dropped.get(pfx, 0) + v.numel()
            save_file(keep, os.path.join(OUT, shard), metadata={"format": "pt"})
            print(f"{shard}: kept {len(keep)} tensors")
        json.dump({"metadata": {"total_size": n_param * 2}, "weight_map": new_map},
                  open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

        cfg = json.load(open(os.path.join(src, "config.json")))
        text = cfg["text_config"]
        text["architectures"] = ["Qwen3_5ForCausalLM"]
        text["mtp_num_hidden_layers"] = 0
        text["transformers_version"] = cfg.get("transformers_version")
        json.dump(text, open(os.path.join(OUT, "config.json"), "w"), indent=2)
        for f in os.listdir(src):
            if f.endswith((".json", ".jinja", ".txt", ".model")) and f not in ("config.json", "model.safetensors.index.json") \
                    and not f.startswith("model.safetensors"):
                shutil.copy(os.path.join(src, f), OUT)       # tokenizer, chat template, generation config
        print(f"kept {n_param/1e9:.3f}B params; dropped: " + ", ".join(f"{k} {v/1e9:.3f}B" for k, v in dropped.items()))
        hf_cache.commit()

    # ---- verification ----
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(OUT)
    ids = tok("Xin chào, hãy đặt báo thức lúc 7 giờ sáng mai.", return_tensors="pt").input_ids
    m, info = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16, output_loading_info=True)
    print("stripped params:", sum(p.numel() for p in m.parameters()) / 1e9)
    print("loading info:", {k: v for k, v in info.items() if v})
    with torch.no_grad():
        a = m(ids).logits[0, -1].float()
    del m
    full = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16)
    print("original params:", sum(p.numel() for p in full.parameters()) / 1e9)
    with torch.no_grad():
        b = full(input_ids=ids).logits[0, -1].float()
    print("max |logit diff| stripped vs original:", (a - b).abs().max().item(),
          "| argmax equal:", a.argmax().item() == b.argmax().item())


def _vllm_has(arch):
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        return arch in ModelRegistry.get_supported_archs()
    except Exception as e:
        return f"? ({type(e).__name__}: {e})"


@app.local_entrypoint()
def main(check_only: bool = False):
    strip.remote(check_only)
