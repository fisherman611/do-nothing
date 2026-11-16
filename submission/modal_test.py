"""Chạy thử toàn luồng (ASR → bộ lọc → FC vLLM → ToolEnv) trên Modal L40S, mô phỏng điều kiện chấm:
1 GPU, 16 CPU, 64 GB RAM, KHÔNG có mạng lúc chạy (block_network). Model được kéo về lúc BUILD image, không phải lúc chạy.

    python -m modal run submission/modal_test.py --limit 40            # so FC bf16 và FC INT8 (SmoothQuant W8A8)
    python -m modal run submission/modal_test.py --limit 40 --fc fc    # chỉ một bản
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "train+public_test" / "public_test"

# repo @ commit (giống build.sh); FC INT8: bản SmoothQuant + W8A8 của cùng checkpoint FC
PINS = {
    "asr": ("fisherman611/pw-tmp-e46f", "85081b30df5a9de0b792aa82a7af7876da74a253"),
    "fc": ("fisherman611/q-v5-6c1e", "c2db4cff5f4cb7b3e2013c20ecd85a91224ca516"),
    "fc_w8a8": ("fisherman611/q-v5-w8-5d2f", "b38420baae253209a33001c0bb0eb5ec7947812a"),
    "filter/model": ("fisherman611/vibi-appfilter-v2-b", "386ba3baf57876a556d06bb5a0eb3a05fa7944ec"),
}
TOK = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
DL = "from huggingface_hub import snapshot_download as d\n" + "".join(
    f"d('{r}', revision='{v}', local_dir='/saved_models/{k}')\n" for k, (r, v) in PINS.items()
) + (f"d('{TOK[0]}', revision='{TOK[1]}', local_dir='/saved_models/tokenizer', "
     "allow_patterns=['tokenizer*','vocab*','merges*','special_tokens_map.json','chat_template*'])\n")

app = modal.App("zac2026-lifemate-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_XET": "1"})
    .apt_install("ffmpeg", "libsndfile1", "libgl1-mesa-glx", "libglib2.0-0", "curl")
    .pip_install("uv")
    .add_local_file(str(HERE / "requirements_fc.txt"), "/tmp/requirements_fc.txt", copy=True)
    .add_local_file(str(HERE / "requirements.txt"), "/tmp/requirements.txt", copy=True)
    .run_commands(
        "uv venv --python 3.12 /opt/fc && VIRTUAL_ENV=/opt/fc uv pip install --no-cache --no-deps -r /tmp/requirements_fc.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match "
        "&& /opt/fc/bin/python -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'",
        "uv venv --python 3.10 /opt/app && VIRTUAL_ENV=/opt/app uv pip install --no-cache -r /tmp/requirements.txt "
        "&& /opt/app/bin/python -c \"from pyvi import ViTokenizer; assert ViTokenizer.tokenize('bật đèn phòng khách')\"",
    )
    .add_local_dir(str(HERE / "config" / "filter"), "/saved_models/filter", copy=True)
    .run_commands(f"/opt/app/bin/python - <<'EOF'\n{DL}EOF", "rm -rf /saved_models/*/.cache /saved_models/filter/model/.cache")
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
    .add_local_dir(str(HERE), "/code", ignore=["saved_models", "*.tar.gz", "**/__pycache__", "training_code", "result", "modal_result", "*.log"])
    .add_local_dir(str(DATA), "/data/public_test")
)


@app.function(image=image, gpu="L40S", cpu=16, memory=65536, timeout=7200, block_network=True)
def run(fc: str, limit: int):
    env = {**os.environ, "MODELS_DIR": "/saved_models", "FC_DIRNAME": fc, "LIMIT": str(limit), "DATA_ROOT": "/data",
           "RESULT_DIR": f"/result_{fc}", "CODE_DIR": "/code", "VLLM_PORT": "8000"}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    p = subprocess.run(["/opt/app/bin/python", "/code/predict.py"], env=env, capture_output=True, text=True)
    out = p.stdout[-3000:] + ("\n--- stderr ---\n" + p.stderr[-2500:] if p.returncode else "")
    sub = Path(f"/result_{fc}/submission.jsonl")
    return {"fc": fc, "gpu": smi, "returncode": p.returncode, "wall_s": round(time.time() - t0, 1), "log": out,
            "submission": sub.read_text(encoding="utf-8") if sub.exists() else ""}


@app.local_entrypoint()
def main(limit: int = 40, fc: str = "fc,fc_w8a8"):
    out = HERE / "modal_result"
    out.mkdir(exist_ok=True)
    for name in fc.split(","):
        r = run.remote(name, limit)
        (out / f"submission_{name}.jsonl").write_text(r.pop("submission"), encoding="utf-8")
        (out / f"info_{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"== {name}: rc={r['returncode']} wall={r['wall_s']}s gpu={r['gpu']}\n{r['log'][-1200:]}", flush=True)
