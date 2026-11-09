"""Chạy agent/clef/train.py trên Modal (A100-80GB). Model gốc: /hf/qwen35_4b_text trên volume `hf-cache`
(strip_qwen35.py). Dữ liệu + output: volume `clef-data` (upload bằng `modal volume put clef-data agent/clef/data /data`).

    modal deploy agent/clef/modal_train.py
    python -c "import modal; f=modal.Function.from_name('lifemate-clef-train','train'); \\
               print(f.spawn(args=['--epochs','2','--max-train','0']).object_id)"
    # smoke: args=['--max-train','64','--epochs','1']
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
         .add_local_dir(os.path.dirname(HERE), "/root/zaic/agent",
                        ignore=["runs", "clef/data", "__pycache__", "*.jsonl"]))


@app.function(image=image, gpu="A100-80GB", volumes={"/hf": hf, "/data": vol}, timeout=12 * 3600)
def train(args: list):
    import subprocess
    import sys
    cmd = [sys.executable, "-u", "/root/zaic/agent/clef/train.py", "--model", "/hf/qwen35_4b_text",
           "--train", "/data/data/train.jsonl", "--val", "/data/data/val_in.jsonl", "--out", "/data/runs/clef_lora",
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
    p = subprocess.run(cmd)
    stop.set()
    vol.commit()
    return p.returncode
