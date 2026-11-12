"""Dựng volume `clef-data` trên một workspace Modal mới cho A2 (train FC v3) — chỉ tải từ HF public, không cần token.

    modal run asr/agent/clef/modal_setup_ws.py

Kết quả trên volume `clef-data` (đúng chỗ modal_train.py đọc):
  /data/fmtA_v3/train.jsonl (+ .meta.json, report.json)   ← fisherman611/fc-8099
  /data/fmtA_v3/val_in.jsonl (+ .meta.json)               ← fisherman611/d-f48e fmtA_48e_20k (cùng val như fmtA_48e)
  /data/newapps_q1/{train,val_newapps,test_newapps}.jsonl ← fisherman611/fc-8099 (đo M3b)
  /data/runs/fmtA_48e/final/                              ← fisherman611/q-f48e-lora (adapter để --init-adapter)
Model gốc /hf/qwen35_4b_text trên volume `hf-cache` dựng riêng bằng `modal run asr/agent/strip_qwen35.py`.
"""
import modal

app = modal.App("lifemate-clef-setup")
vol = modal.Volume.from_name("clef-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"})

FILES = [  # (repo, repo_type, revision, path trong repo, đích trên volume)
    ("fisherman611/fc-8099", "dataset", "ffabd9a7c6987a26e8ee45cc115ff476b4a72514", "fmtA_v3/train.jsonl", "fmtA_v3/train.jsonl"),
    ("fisherman611/fc-8099", "dataset", "ffabd9a7c6987a26e8ee45cc115ff476b4a72514", "fmtA_v3/train.jsonl.meta.json", "fmtA_v3/train.jsonl.meta.json"),
    ("fisherman611/fc-8099", "dataset", "ffabd9a7c6987a26e8ee45cc115ff476b4a72514", "fmtA_v3/report.json", "fmtA_v3/report.json"),
    ("fisherman611/d-f48e", "dataset", "26be2d6840627b8d0e463ef17393c8275d50ba0f", "fmtA_48e_20k/val_in.jsonl", "fmtA_v3/val_in.jsonl"),
    ("fisherman611/d-f48e", "dataset", "26be2d6840627b8d0e463ef17393c8275d50ba0f", "fmtA_48e_20k/val_in.jsonl.meta.json", "fmtA_v3/val_in.jsonl.meta.json"),
    *[("fisherman611/fc-8099", "dataset", "ffabd9a7c6987a26e8ee45cc115ff476b4a72514", f"newapps_q1/{f}", f"newapps_q1/{f}")
      for f in ("train.jsonl", "val_newapps.jsonl", "test_newapps.jsonl")],
    *[("fisherman611/q-f48e-lora", "model", "a0a0b8834984d1cc6b7c010c831cdc4b63f408c3", f"final/{f}", f"runs/fmtA_48e/final/{f}")
      for f in ("adapter_config.json", "adapter_model.safetensors")],
]


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def setup():
    import os, shutil
    from huggingface_hub import hf_hub_download
    for repo, rtype, rev, path, dst in FILES:
        src = hf_hub_download(repo, path, repo_type=rtype, revision=rev, cache_dir="/tmp/hf")
        out = f"/data/{dst}"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy(src, out)
        print(f"{repo}:{path} → {out} ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    vol.commit()
    for root, _, fs in os.walk("/data"):
        for f in fs:
            p = os.path.join(root, f)
            n = sum(1 for _ in open(p, encoding="utf-8")) if p.endswith(".jsonl") else ""
            print(f"  {p}  {n}")


@app.local_entrypoint()
def main():
    setup.remote()
