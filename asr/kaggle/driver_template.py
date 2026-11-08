# Kernel Kaggle train PhoWhisper v2 (sinh bởi zaic/kaggle/launch.py, đừng sửa tay).
# CFG và FILES được launch.py chèn vào. Chạy: cài thư viện → tìm data trong /kaggle/input → copy checkpoint của
# kernel trước (nếu resume) → asr_train.py có giới hạn thời gian → ghi status.json.
CFG = __CFG__
FILES = __FILES__

import base64, glob, json, os, re, shutil, subprocess, sys, time, zipfile

T0 = time.time()
W = "/kaggle/working"
CODE, OUT, DATA = f"{W}/code", f"{W}/out", "/kaggle/temp/data"
os.makedirs(CODE, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
for name, b64 in FILES.items():
    open(f"{CODE}/{name}", "wb").write(base64.b64decode(b64))
print("CFG", json.dumps(CFG, ensure_ascii=False))
subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "peft==0.17.1",
                "accelerate==1.10.1", "jiwer", "soundfile"], check=True)
import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "| arch", torch.cuda.get_arch_list())

# ---- data: manifest_v2.json + lexicon_v2.json + train/audio (Kaggle có thể giữ train.zip hoặc đã giải nén) ----
man = sorted(glob.glob("/kaggle/input/**/manifest_v2.json", recursive=True))
assert man, "không thấy manifest_v2.json trong /kaggle/input"
root = os.path.dirname(man[0])
for f in ("manifest_v2.json", "lexicon_v2.json"):
    shutil.copy(f"{root}/{f}", f"{DATA}/{f}")
def find_audio(base):
    return [d for d in glob.glob(f"{base}/**/train/audio", recursive=True) if os.path.isdir(d)]


aud = find_audio("/kaggle/input")
if not aud:   # Kaggle giữ nguyên train.zip → giải nén vào /kaggle/temp
    zs = sorted(glob.glob("/kaggle/input/**/train*.zip", recursive=True))
    assert zs, "không thấy train/audio hoặc train.zip"
    zipfile.ZipFile(zs[0]).extractall("/kaggle/temp/unz")
    aud = find_audio("/kaggle/temp/unz") or [d for d in glob.glob("/kaggle/temp/unz/**/audio", recursive=True)]
os.symlink(os.path.dirname(aud[0]), f"{DATA}/train")
n_wav = len(glob.glob(f"{DATA}/train/audio/*/*.wav"))
print("data:", root, "| wav:", n_wav)
assert n_wav > 5000, n_wav

# ---- resume: checkpoint từ output kernel trước ----
for prev in sorted(glob.glob("/kaggle/input/**/out", recursive=True)):
    if os.path.isdir(prev) and prev != OUT:
        print("copy checkpoint từ", prev)
        shutil.copytree(prev, OUT, dirs_exist_ok=True)

# ---- train ----
# Kaggle cấp T4 x2: chỉ dùng 1 GPU (Trainer sẽ tự bật DataParallel, autocast không theo sang thread replica)
env = dict(os.environ, ZAIC_DATA=DATA, ZAIC_OUT=OUT, PYTHONPATH=CODE, TOKENIZERS_PARALLELISM="false",
           CUDA_VISIBLE_DEVICES="0")


def run(args):
    cmd = [sys.executable, "-u", f"{CODE}/asr_train.py"] + args
    print("RUN", " ".join(cmd), flush=True)
    log = open(f"{W}/train.log", "a")
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail, result = [], None
    for line in p.stdout:
        sys.stdout.write(line)
        log.write(line)
        tail = (tail + [line])[-200:]
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
    p.wait()
    return p.returncode, "".join(tail), result


budget = CFG["time_budget_h"] - (time.time() - T0) / 3600      # tính cả thời gian cài đặt
args = CFG["args"] + ["--time-budget-h", f"{budget:.2f}"]
code, tail, result = run(args + ["--bs", str(CFG["bs"]), "--accum", str(CFG["accum"])])
if code != 0 and "out of memory" in tail.lower() and glob.glob(f"{OUT}/*_adapter"):
    # OOM lúc eval cuối (train đã xong, adapter đã lưu): resume từ checkpoint cuối → vào thẳng eval, giải mã batch 1
    print("OOM lúc eval cuối → chạy lại eval với --eval-bs 1")
    budget = CFG["time_budget_h"] - (time.time() - T0) / 3600
    args = CFG["args"] + ["--time-budget-h", f"{max(budget, 0.5):.2f}", "--eval-bs", "1"]
    code, tail, result = run(args + ["--bs", str(CFG["bs"]), "--accum", str(CFG["accum"])])
elif code != 0 and "out of memory" in tail.lower() and CFG["bs"] > 2:
    bs2 = CFG["bs"] // 2
    print(f"OOM với bs {CFG['bs']} → thử bs {bs2}")
    budget = CFG["time_budget_h"] - (time.time() - T0) / 3600
    args = CFG["args"] + ["--time-budget-h", f"{budget:.2f}"]
    code, tail, result = run(args + ["--bs", str(bs2), "--accum", str(CFG["accum"] * CFG["bs"] // bs2)])

status = {"returncode": code, "result": result, "hours": round((time.time() - T0) / 3600, 2), "cfg": CFG}
json.dump(status, open(f"{W}/status.json", "w"), ensure_ascii=False, indent=1)
print("STATUS", json.dumps(status, ensure_ascii=False)[:2000])
shutil.rmtree(CODE, ignore_errors=True)
if code == 0 and not (result or {}).get("status") == "incomplete":   # xong hẳn: bỏ checkpoint (optimizer) cho output nhẹ
    for d in glob.glob(f"{OUT}/*_ckpt"):
        shutil.rmtree(d, ignore_errors=True)
if code != 0:
    raise SystemExit(f"train lỗi (code {code}), xem train.log")
