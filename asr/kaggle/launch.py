"""Chạy 3 biến thể ASR v2 (PLAN_ASR_bias_v2.md) trên Kaggle, mỗi biến thể một tài khoản.

Token Kaggle (KGAT_...) KHÔNG để trong repo: một file, mỗi dòng một token theo thứ tự V1, V2, V3:
    export KAGGLE_TOKENS=/đường/dẫn/tokens

    python kaggle/launch.py data                 # dựng + upload dataset riêng tư `zaic-asr-v2-data` cho 3 tài khoản
    python kaggle/launch.py push V1 --smoke      # chạy thử nhanh (24 hội thoại, 1 epoch) trước
    python kaggle/launch.py push V2              # đẩy + chạy kernel train (T4, có giới hạn thời gian)
    python kaggle/launch.py push V2 --resume 1   # phiên tiếp theo: resume từ output của phiên trước
    python kaggle/launch.py status               # trạng thái mọi kernel
    python kaggle/launch.py output V2 [--resume 1]   # tải output (status.json, eval.json, train.log)

Cần: `python modal/make_manifest_v2.py data/train+public_test` trước (sinh manifest_v2.json, lexicon_v2.json).
P100 đã bị Kaggle ngừng (CLI chuyển sang T4), nên dùng NvidiaTeslaT4. T4 không có bf16: asr_train tự dùng fp16.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZAIC = os.path.dirname(HERE)
WORK = os.environ.get("KAGGLE_WORK", os.path.join(ZAIC, ".kaggle_work"))   # thư mục tạm (không commit)
KAGGLE = os.environ.get("KAGGLE_BIN", shutil.which("kaggle") or "kaggle")
DATASET = os.environ.get("KAGGLE_DATASET", "zaic-asr-v2-data")   # L1/L2 (v2.1 + TTS): zaic-asr-v21-data
ACCEL = "NvidiaTeslaT4"

COMMON = ["--label-smoothing", "0.1", "--ctc-weight", "0.3", "--use-prompt", "--prompt-v2",
          "--epochs", "4", "--base-fp16", "--eval-items", "150", "--eval-bs", "4"]
VARIANTS = {   # account = dòng trong file token
    "V1": dict(account=0, args=COMMON),
    "V2": dict(account=1, args=COMMON + ["--label-ent-prob", "0.5"]),
    "V3": dict(account=2, args=COMMON + ["--label-ent-prob", "0.5", "--all-distract-prob", "0.3"]),
}
# A4 large (11/11): công thức v2.1 của ASR v2 medium (TTS + tăng cường + prompt V2.1 + CTC/inter-CTC/KWS) trên PhoWhisper-large.
# L1 = từ PhoWhisper-large gốc; L2 = từ v1 large đã fine-tune (fisherman611/pw-tmp-e46f). Dataset: KAGGLE_DATASET=zaic-asr-v21-data
V21 = ["--label-smoothing", "0.1", "--ctc-weight", "0.3", "--inter-ctc-layer", "16", "--kws-weight", "0.1",
       "--use-prompt", "--prompt-v2", "--label-ent-prob", "0.5", "--prompt-drop", "0.4", "--prev-noise", "0.7",
       "--tts-manifest", "tts_manifest.json", "--audio-aug", "0.5", "--no-eval-lexicon", "--base-fp16",
       "--eval-items", "150", "--eval-bs", "4", "--patience", "2", "--save-limit", "1", "--workers", "4"]
VARIANTS["L1"] = dict(account=0, args=V21 + ["--epochs", "4", "--tag-suffix", "v21L"])
VARIANTS["L2"] = dict(account=1, args=V21 + ["--model-id", "fisherman611/pw-tmp-e46f", "--model-revision", "",
                                             "--epochs", "2", "--tag-suffix", "v21L_e46f"])
TIME_BUDGET_H = 8.0   # giờ train trong phiên 12 h, chừa ~3.5 h cho eval cuối (3 điều kiện, beam 5) và lưu
BS, ACCUM = 4, 8      # batch hiệu dụng 32 như v1; OOM thì driver tự giảm bs 2 × accum 16


def tokens():
    p = os.environ.get("KAGGLE_TOKENS")
    assert p and os.path.exists(p), "đặt KAGGLE_TOKENS=<file token, mỗi dòng một KGAT_..., thứ tự V1 V2 V3>"
    return [l.strip() for l in open(p) if l.strip()]


def kg(account, *args, capture=False):
    env = dict(os.environ, KAGGLE_API_TOKEN=tokens()[account])
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    r = subprocess.run([KAGGLE, *args], env=env, text=True, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"kaggle {' '.join(args[:3])} lỗi")
    return r.stdout if capture else print(r.stdout.strip()[-1500:])


def username(account):
    out = kg(account, "config", "view", capture=True)
    return next(l.split(":", 1)[1].strip() for l in out.splitlines() if "username" in l)


def cmd_data(a):
    src = os.path.join(ZAIC, "modal")
    ds = os.path.join(WORK, "dataset")
    if not os.path.isdir(f"{ds}/train/audio"):
        os.makedirs(f"{ds}/train", exist_ok=True)
        shutil.copytree(os.path.join(ZAIC, a.data_root, "train", "audio"), f"{ds}/train/audio")
    if a.v21:   # bộ v2.1 đã dùng train ASR v2 medium (data/tts_v2): manifest/lexicon mới + TTS + bảng nhiễu lượt trước
        t = os.path.join(ZAIC, "data", "tts_v2")
        for f in ("manifest_v2.json", "lexicon_v2.json", "tts_manifest.json", "prev_noise_table.json"):
            shutil.copy(f"{t}/{f}", f"{ds}/{f}")
        if not os.path.isdir(f"{ds}/tts/wav"):
            shutil.copytree(f"{t}/wav", f"{ds}/tts/wav")
    else:
        for f in ("manifest_v2.json", "lexicon_v2.json"):
            shutil.copy(f"{src}/{f}", f"{ds}/{f}")
    accounts = sorted({v["account"] for v in VARIANTS.values()}) if a.account is None else [a.account]
    for acc in accounts:
        user = username(acc)
        json.dump({"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]},
                  open(f"{ds}/dataset-metadata.json", "w"))
        exists = DATASET in kg(acc, "datasets", "list", "--mine", "-s", DATASET, capture=True)
        print(f"[{user}] {'cập nhật' if exists else 'tạo'} dataset riêng tư {DATASET}")
        if exists:
            kg(acc, "datasets", "version", "-p", ds, "-m", "update", "-r", "zip")
        else:
            kg(acc, "datasets", "create", "-p", ds, "-r", "zip")   # mặc định private


def slug(variant, resume, smoke=False):
    return f"zaic-asr-{variant.lower()}" + ("-smoke" if smoke else "") + (f"-r{resume}" if resume else "")


def cmd_push(a):
    v = VARIANTS[a.variant]
    user = username(v["account"])
    files = {f: base64.b64encode(open(os.path.join(ZAIC, "modal", f), "rb").read()).decode()
             for f in ("asr_train.py", "bias_prompt.py")}
    args = v["args"] + (["--max-items", "100", "--epochs", "1", "--eval-items", "12", "--no-final-eval"] if a.smoke else [])
    cfg = {"variant": a.variant, "args": args, "time_budget_h": TIME_BUDGET_H, "bs": BS, "accum": ACCUM,
           "resume": a.resume}
    tpl = open(os.path.join(HERE, "driver_template.py")).read()
    code = tpl.replace("__CFG__", json.dumps(cfg, ensure_ascii=False)).replace("__FILES__", json.dumps(files))
    s = slug(a.variant, a.resume, a.smoke)
    kd = os.path.join(WORK, "kernels", s)
    os.makedirs(kd, exist_ok=True)
    open(f"{kd}/kernel.py", "w").write(code)
    meta = {"id": f"{user}/{s}", "title": s, "code_file": "kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": ACCEL, "dataset_sources": [f"{user}/{DATASET}"],
            "kernel_sources": [f"{user}/{slug(a.variant, a.resume - 1, a.smoke)}"] if a.resume else []}
    json.dump(meta, open(f"{kd}/kernel-metadata.json", "w"), indent=1)
    print(f"[{user}] push {s} ({a.variant}{', resume từ ' + meta['kernel_sources'][0] if a.resume else ''})")
    kg(v["account"], "kernels", "push", "-p", kd, "--accelerator", ACCEL)


def cmd_status(a):
    for name, v in VARIANTS.items():
        user = username(v["account"])
        for r in range(a.max_resume + 1):
            print(name, kg(v["account"], "kernels", "status", f"{user}/{slug(name, r)}", capture=True).strip())


def cmd_output(a):
    v = VARIANTS[a.variant]
    user = username(v["account"])
    s = slug(a.variant, a.resume, a.smoke)
    d = os.path.join(WORK, "outputs", s)
    os.makedirs(d, exist_ok=True)
    kg(v["account"], "kernels", "output", f"{user}/{s}", "-p", d, "--file-pattern",
       r".*(status\.json|eval\.json|train\.log|\.log)$")
    print("→", d)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    p.add_argument("--data-root", default="data/train+public_test")
    p.add_argument("--v21", action="store_true", help="bộ v2.1 + TTS (cho L1/L2)")
    p.add_argument("--account", type=int)
    p = sub.add_parser("push")
    p.add_argument("variant", choices=VARIANTS)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="chạy thử ~100 hội thoại, 1 epoch (đo tốc độ)")
    p = sub.add_parser("status")
    p.add_argument("--max-resume", type=int, default=0)
    p = sub.add_parser("output")
    p.add_argument("variant", choices=VARIANTS)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"data": cmd_data, "push": cmd_push, "status": cmd_status, "output": cmd_output}[a.cmd](a)


if __name__ == "__main__":
    main()
