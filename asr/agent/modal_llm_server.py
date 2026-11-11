"""vLLM OpenAI-compatible servers on Modal, one web endpoint per candidate LLM (zero-shot tool-calling eval).

The agent loop (`agent.py`) runs where the dataset is (local / myserver) and talks to these endpoints over HTTP,
so `*_environment/` never leaves the machine and is only ever read by `zaic_tools` (AGENT.md).

    modal secret create llm-api-key LLM_API_KEY=<random>      # once
    modal deploy asr/agent/modal_llm_server.py                  # prints one URL per model
    # containers start on the first request and stop after SCALEDOWN seconds idle (no cost while idle)

Pinned: vLLM 0.31.0 (2026-10-05) and an exact HF revision per model.
"""
import os
import subprocess

import modal

VLLM_VERSION = "0.31.0"
GPU = "L40S"          # 48 GB, $1.95/h; 4B bf16 + 16k context fits with room for KV cache
SCALEDOWN = 300       # seconds idle before the container stops
MAX_LEN = 16384       # competition limit for the LLM context

# key -> (HF repo, revision, extra vLLM args)
MODELS = {
    "qwen3_4b_2507": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", []),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                  ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']),
    "qwen3_1p7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", []),
    "sailor2_3b": ("sail/Sailor2-3B-Chat", "d60722644e700133576489719dcbc288036628d5", []),
    # Qwen3.5-4B with vision encoder + MTP head removed (strip_qwen35.py); a local path on the hf-cache volume
    "qwen35_4b_text": ("/hf/qwen35_4b_text", None, []),
    # Clef-token P2 (agent/clef): LoRA 1 epoch gộp vào trọng số bằng agent/clef/modal_merge.py
    "qwen35_4b_clef_p2": ("/hf/qwen35_4b_clef_p2", None, []),
    # Ablation format A (q2=all) / B (q2=discrete), smoke 1835 ví dụ, gộp bằng modal_merge.py (PLAN_prompt_format_v2.md)
    "qwen35_4b_smokeA": ("/hf/qwen35_4b_smokeA", None, []),
    "qwen35_4b_smokeB": ("/hf/qwen35_4b_smokeB", None, []),
    "qwen35_4b_fmtA_mid_e1": ("/hf/qwen35_4b_fmtA_mid_e1", None, []),
    # Cùng model, trả logprob của nhãn (PLAN_generalize_tools §3.0): tới 64 nhãn (Q1 có tới 52 phương án), logprob
    # tính sau logits processor (mask allowed_token_ids) → softmax chỉ trên nhãn
    "mid_lp": ("/hf/qwen35_4b_fmtA_mid_e1", None, ["--max-logprobs", "64", "--logprobs-mode", "processed_logprobs"]),
    # F2 (PLAN_48h): Clef trên fmtA_48e_20k (hai tầng + nhiễu ASR v2), gộp bằng merge_local.py trên myserver
    "qwen35_4b_fmtA_48e": ("/hf/qwen35_4b_fmtA_48e", None, ["--max-logprobs", "64", "--logprobs-mode", "processed_logprobs"]),
}

# LLM_APP=<tên> lúc deploy để tạo app riêng (không đụng endpoint của app đang chạy eval)
app = modal.App(os.environ.get("LLM_APP", "lifemate-llm-zeroshot"))
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf",
              # image slim không có nvcc: FlashInfer sampler cần JIT-compile nên bị lỗi; dùng sampler PyTorch
              "VLLM_USE_FLASHINFER_SAMPLER": "0"}))


def _serve(key: str):
    repo, rev, extra = MODELS[key]
    cmd = ["vllm", "serve", repo, *(["--revision", rev] if rev else []), "--served-model-name", "llm",
           "--host", "0.0.0.0", "--port", "8000", "--max-model-len", str(MAX_LEN),
           "--gpu-memory-utilization", "0.90", "--seed", "0", "--enable-prefix-caching",
           "--api-key", "$LLM_API_KEY", *extra]
    subprocess.Popen(" ".join(f"'{c}'" if c.startswith("{") else c for c in cmd), shell=True)


def _fn(**kw):
    return app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=6 * 3600,
                        scaledown_window=SCALEDOWN, secrets=[modal.Secret.from_name("llm-api-key")], **kw)


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen3_4b_2507():
    _serve("qwen3_4b_2507")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b():
    _serve("qwen35_4b")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen3_1p7b():
    _serve("qwen3_1p7b")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_sailor2_3b():
    _serve("sailor2_3b")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_text():
    _serve("qwen35_4b_text")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_clef_p2():
    _serve("qwen35_4b_clef_p2")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_smokeA():
    _serve("qwen35_4b_smokeA")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_smokeB():
    _serve("qwen35_4b_smokeB")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_fmtA_mid_e1():
    _serve("qwen35_4b_fmtA_mid_e1")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_mid_lp():
    _serve("mid_lp")


@_fn()
@modal.concurrent(max_inputs=256)
@modal.web_server(8000, startup_timeout=1200)
def serve_qwen35_4b_fmtA_48e():
    _serve("qwen35_4b_fmtA_48e")
