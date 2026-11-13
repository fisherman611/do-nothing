"""ASR v4m trên Modal H100: GĐ1 (thật + TTS P1, lr 1.5e-5 cosine, 6 epoch) → GĐ2 (chỉ thật, lr 5e-6 cosine, 3 epoch).
Volume `zaic-asr`: /data (manifest.json, train/audio, noise_bank, tts_pub, tts_manifest_pub.json), /out (kết quả).
    set -a; source ~/.zaic_wandb.env; set +a; modal deploy asr_v4_code/modal_asr_train.py
    python -c "import modal;print(modal.Function.from_name('lifemate-asr-train','run').spawn().object_id)"
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-asr-train")
vol = modal.Volume.from_name("zaic-asr", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
         .pip_install("torch==2.8.0", "torchaudio==2.8.0", "transformers==4.57.1", "peft==0.20.0", "accelerate==1.10.1",
                      "jiwer==4.0.0", "soundfile==0.13.1", "scipy==1.15.3", "numpy==2.2.6", "wandb==0.31.0", "librosa")
         .env({"ZAIC_DATA": "/data", "TOKENIZERS_PARALLELISM": "false", "WANDB_PROJECT": "zaic-lifemate-asr",
               "WANDB_LOG_MODEL": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/hfc"})
         .add_local_file(os.path.join(HERE, "asr_train.py"), "/root/asr_train.py")
         .add_local_file(os.path.join(HERE, "bias_prompt.py"), "/root/bias_prompt.py"))

M = ["--model-id", "fisherman611/pw-tmp-e46f", "--model-revision", "85081b30df5a9de0b792aa82a7af7876da74a253"]
COMMON = [*M, "--use-prompt", "--label-smoothing", "0.1", "--ctc-weight", "0", "--bs", "8", "--accum", "4", "--strip-punct",
          "--aug-v3", "0.3", "--lr-sched", "cosine", "--report-to", "wandb"]   # bs·accum = 32 như bản myserver


@app.function(image=image, gpu=os.environ.get("ASR_GPU", "H100"), volumes={"/data": vol}, timeout=8 * 3600,
              secrets=[modal.Secret.from_dict({"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})])
def run(tag: str = "v4m"):
    import glob, subprocess, sys, threading
    out = f"/data/out/{tag}"; os.makedirs(out, exist_ok=True)
    env = {**os.environ, "ZAIC_OUT": out}
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try: vol.commit()
            except Exception as e: print("commit lỗi", e, flush=True)
    threading.Thread(target=committer, daemon=True).start()

    def stage(name, extra):
        cmd = [sys.executable, "-u", "/root/asr_train.py", *COMMON, *extra]
        print("==", name, " ".join(cmd), flush=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**env, "WANDB_NAME": f"{tag}_{name}"})
        with open(f"{out}/{name}.log", "a") as f:
            for line in p.stdout:
                print(line, end="", flush=True); f.write(line)
        rc = p.wait(); vol.commit(); return rc

    rc = stage("p1_full", ["--lr", "1.5e-5", "--epochs", "6", "--warmup-steps", "60", "--patience", "6",
                           "--tts-manifest", "tts_manifest_pub.json", "--tag-suffix", f"{tag}p1"])
    if rc: stop.set(); return {"p1": rc}
    p1 = sorted(glob.glob(f"{out}/*{tag}p1_adapter"))[0]
    rc2 = stage("p2_real", ["--lr", "5e-6", "--epochs", "3", "--warmup-steps", "30", "--patience", "2",
                            "--init-adapter", p1, "--no-resume", "--tag-suffix", f"{tag}p2_real"])
    stop.set(); vol.commit()
    return {"p1": rc, "p2": rc2}
