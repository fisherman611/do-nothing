"""Chạy lf_prep.py / lf_train.py (Clef-token bằng LLaMA-Factory) trên Modal.

Image: LLaMA-Factory (main, kéo transformers 5.8.0 / peft 0.18.1 / torch 2.15) + flash-linear-attention 0.6.0;
model gốc /hf/qwen35_4b_text (volume `hf-cache`), dữ liệu + output trên volume `clef-data`.

    python agent/clef/modal_lf.py prep  --src /data/data/train.jsonl --out /data/lf/first64 --select first --n 64
    python agent/clef/modal_lf.py train --data /data/lf/first64 --out /data/runs/lf_smoke --gpu A100-80GB \
        [--set per_device_train_batch_size=2 disable_gradient_checkpointing=true ...]
"""
import argparse
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-clef-lf")
hf = modal.Volume.from_name("hf-cache")
vol = modal.Volume.from_name("clef-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("llamafactory @ git+https://github.com/hiyouga/LlamaFactory.git@28462f1",
                      "flash-linear-attention==0.6.0")
         .env({"HF_HOME": "/hf", "PYTHONPATH": "/root/zaic", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(os.path.dirname(HERE), "/root/zaic/agent",
                        ignore=["runs", "clef/data", "__pycache__", "*.jsonl"]))

BASE_CFG = {
    "stage": "sft", "do_train": True, "finetuning_type": "lora",
    "model_name_or_path": "/hf/qwen35_4b_text", "template": "qwen3_5_nothink", "trust_remote_code": False,
    "dataset": "clef_pretokenized",          # bắt buộc với parser, bị bỏ qua khi có tokenized_path
    "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "lora_target": "all",
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4, "num_train_epochs": 2.0, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
    "weight_decay": 0.0, "max_grad_norm": 1.0, "bf16": True, "seed": 0,
    "logging_steps": 1, "save_strategy": "no", "report_to": "none", "overwrite_output_dir": True,
    "dataloader_num_workers": 2,
}


def _parse(v):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


@app.function(image=image, volumes={"/hf": hf, "/data": vol}, cpu=8, memory=32768, timeout=3 * 3600)
def prep(args: list):
    import subprocess
    import sys
    p = subprocess.run([sys.executable, "-u", "/root/zaic/agent/clef/lf_prep.py", *args])
    vol.commit()
    return p.returncode


GPUS = ("A100-80GB", "H100", "A100-80GB:2", "H100:2", "H100:4")


@app.function(image=image, gpu="A100-80GB", volumes={"/hf": hf, "/data": vol}, timeout=12 * 3600)
def train(cfg: dict, env: dict):
    import json
    import subprocess
    import sys
    os.makedirs(cfg["output_dir"], exist_ok=True)
    path = os.path.join(cfg["output_dir"], "lf_cfg.json")
    json.dump(cfg, open(path, "w"), indent=1)
    print(json.dumps(cfg), flush=True)
    n = int(env.pop("NPROC", "1"))
    cmd = ([sys.executable, "-u", "/root/zaic/agent/clef/lf_train.py", path] if n == 1 else
           ["torchrun", "--nproc_per_node", str(n), "/root/zaic/agent/clef/lf_train.py", path])
    p = subprocess.run(cmd, env={**os.environ, **env})
    vol.commit()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", default="all")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--data")
    ap.add_argument("--gpu", default="A100-80GB", choices=list(GPUS))
    ap.add_argument("--set", nargs="*", default=[], help="khoá=giá trị ghi đè BASE_CFG")
    ap.add_argument("--env", nargs="*", default=[], help="CLEF_LS=0.1 ...")
    a = ap.parse_args()
    with modal.enable_output(), app.run():
        if a.cmd == "prep":
            rc = prep.remote(["--src", a.src, "--out", a.out, "--select", a.select, "--n", str(a.n)])
        else:
            cfg = {**BASE_CFG, "tokenized_path": a.data, "output_dir": a.out,
                   **{k: _parse(v) for k, v in (s.split("=", 1) for s in a.set)}}
            env = dict(s.split("=", 1) for s in a.env)
            if ":" in a.gpu:
                env.setdefault("NPROC", a.gpu.split(":")[1])
            rc = train.with_options(gpu=a.gpu).remote(cfg, env)
    print("RETURNCODE", rc)


if __name__ == "__main__":
    main()
