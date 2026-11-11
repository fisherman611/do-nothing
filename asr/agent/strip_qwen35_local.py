"""Strip Qwen3.5-4B → text-only (Qwen3_5ForCausalLM), chạy ngoài Modal (cùng logic agent/strip_qwen35.py).
    python strip_qwen35_local.py --out <dir>   (đọc snapshot HF cache local, rev 851bf6e...)
"""
import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

REPO, REV, KEEP = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "model.language_model."
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
src = snapshot_download(REPO, revision=REV, local_files_only=True)
os.makedirs(a.out, exist_ok=True)
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
    save_file(keep, os.path.join(a.out, shard), metadata={"format": "pt"})
    print(f"{shard}: kept {len(keep)} tensors", flush=True)
json.dump({"metadata": {"total_size": n_param * 2}, "weight_map": new_map},
          open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(src, "config.json")))
text = cfg["text_config"]
text["architectures"] = ["Qwen3_5ForCausalLM"]
text["mtp_num_hidden_layers"] = 0
text["transformers_version"] = cfg.get("transformers_version")
json.dump(text, open(os.path.join(a.out, "config.json"), "w"), indent=2)
for f in os.listdir(src):
    if f.endswith((".json", ".jinja", ".txt", ".model")) and f not in ("config.json", "model.safetensors.index.json") \
            and not f.startswith("model.safetensors"):
        shutil.copy(os.path.join(src, f), a.out)
print(f"kept {n_param/1e9:.3f}B params; dropped: " + ", ".join(f"{k} {v/1e9:.3f}B" for k, v in dropped.items()))
print("→", a.out)
