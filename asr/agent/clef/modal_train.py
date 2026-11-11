"""Chạy agent/clef/train.py trên Modal (A100-80GB). Model gốc: /hf/qwen35_4b_text trên volume `hf-cache`
(strip_qwen35.py). Dữ liệu + output: volume `clef-data` (upload bằng `modal volume put clef-data agent/clef/data /data`).

    modal deploy agent/clef/modal_train.py
    python -c "import modal; f=modal.Function.from_name('lifemate-clef-train','train'); \\
               print(f.spawn(args=['--epochs','2','--max-train','0']).object_id)"
    # smoke: args=['--max-train','64','--epochs','1']
    # ablation format (PLAN_prompt_format_v2.md): upload `modal volume put clef-data agent/clef/data/fmtA /fmtA` rồi
    #   f.spawn(args=[...], data='fmtA', run='fmtA_e1')  (tương tự fmtB)
    # train tiếp: thêm '--save-every','50' khi chạy; sau sự cố / để chạy thêm epoch:
    #   args=[..., '--resume','/data/runs/clef_lora/last', '--epochs','2']  (cùng --out, --accum, data)

Chưa chạy lần nào: bước 0 (kernel Gated DeltaNet, LoRA target, bộ nhớ ở 8–12k token) phải xong trước.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-clef-train")
hf = modal.Volume.from_name("hf-cache")
vol = modal.Volume.from_name("clef-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         # vllm 0.31.0 kéo theo bản torch/transformers đã chạy được Qwen3_5ForCausalLM (strip_qwen35.py)
         .pip_install("vllm==0.31.0", "peft", "flash-linear-attention", "huggingface_hub[hf_transfer]")
         .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/zaic"})
         # transformers tải kernel causal_conv1d dựng sẵn từ HF Hub qua gói `kernels` (không có thì lùi về PyTorch)
         .pip_install("kernels==0.16.0")      # transformers 5.17 đòi 0.16.0 <= kernels < 0.17.0
         .pip_install("wandb")
         .add_local_dir(os.path.dirname(HERE), "/root/zaic/agent",
                        ignore=["runs", "clef/data", "__pycache__", "*.jsonl"]))


# GPU: env CLEF_GPU lúc deploy/chạy (A100-80GB, H100, ...). W&B: cần secret `wandb` (WANDB_API_KEY=...) — gắn luôn,
# không đặt điều kiện theo env (container không thấy env lúc deploy → lệch số dependency, fail startup).
# Tắt log một lần chạy: args có '--wandb', ''
# W&B: key lấy từ env WANDB_API_KEY lúc deploy (luôn đúng 1 secret → không lệch số dependency; rỗng = không log W&B)
@app.function(image=image, gpu=os.environ.get("CLEF_GPU", "A100-80GB"), volumes={"/hf": hf, "/data": vol},
              timeout=12 * 3600, secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def train(args: list, data: str = "data", run: str = "clef_lora"):
    """data: thư mục con của /data chứa train.jsonl + val_in.jsonl (+ *.meta.json), vd 'data', 'fmtA', 'fmtB'.
    run: tên thư mục output trong /data/runs."""
    import subprocess
    import sys
    if "--wandb" not in args and os.environ.get("WANDB_API_KEY"):
        args = [*args, "--wandb", "lifemate-clef", "--wandb-name", run]
    cmd = [sys.executable, "-u", "/root/zaic/agent/clef/train.py", "--model", "/hf/qwen35_4b_text",
           "--train", f"/data/{data}/train.jsonl", "--val", f"/data/{data}/val_in.jsonl", "--out", f"/data/runs/{run}",
           *args]
    print(" ".join(cmd), flush=True)
    import threading
    stop = threading.Event()

    def committer():                    # commit volume định kỳ: job chết giữa chừng vẫn còn checkpoint để --resume
        while not stop.wait(600):
            try:
                vol.commit()
            except Exception as e:      # noqa: BLE001
                print("vol.commit lỗi:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    os.makedirs(f"/data/runs/{run}", exist_ok=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(f"/data/runs/{run}/stdout.log", "a") as f:   # tee: stdout+stderr vào volume để xem lại khi job lỗi
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    rc = p.wait()
    stop.set()
    vol.commit()
    return rc
