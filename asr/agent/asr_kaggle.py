"""Chạy PhoWhisper (fine-tune trên HF) trên manifest ASR (asr_manifest.py) bằng Kaggle, 2 GPU T4 song song (mỗi GPU 1 shard theo hội thoại).
Cùng logic với asr_modal.py (prompt = transcript ASR các lượt user trước + text assistant; beam 5; fp16).

Trong kernel:  python asr_kaggle.py --root /kaggle/input/<dataset> --split val --model fisherman611/pw-tmp-c3d9 --out /kaggle/working/val.asr.json
Dataset: <root>/manifest.json, <root>/audio/<đường dẫn audio trong manifest> (cùng cấu trúc `audio/<id>/<id>.wav`).
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

SOP = "<|startofprev|>"
MAX_PROMPT_TOKENS = 180


def prompt_ids(tok, prev, hyps):
    parts = []
    for role, text, uid in prev:
        if role == "user" and uid is not None:
            text = hyps.get(uid, "")
        if text:
            parts.append(text.strip())
    if not parts:
        return []
    ids = tok(" " + " ".join(parts), add_special_tokens=False).input_ids[-MAX_PROMPT_TOKENS:]
    return [tok.convert_tokens_to_ids(SOP)] + ids


def worker(a):
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    items = json.load(open(a.shard_file, encoding="utf-8"))
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda").eval()
    sot, tok = model.config.decoder_start_token_id, proc.tokenizer

    def run(wav, pid):
        feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=a.beams, max_new_tokens=200, **kw)[0].tolist()
        if sot in ids:
            ids = ids[ids.index(sot):]
        return proc.batch_decode([ids], skip_special_tokens=True)[0].strip()

    by_sid = collections.defaultdict(list)
    for it in items:
        by_sid[it["sid"]].append(it)
    hyps, secs, errs = {}, {}, {}
    t_all = time.time()
    for n, (sid, utts) in enumerate(by_sid.items()):
        utts.sort(key=lambda x: (x["kind"] == "final", x["uid"]))
        for it in utts:
            t0 = time.time()
            try:
                wav, sr = sf.read(os.path.join(a.root, "audio", it["audio"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                assert sr == 16000, sr
                hyps[it["uid"]] = run(wav, prompt_ids(tok, it["prev"], hyps) if not a.no_prompt else [])
            except Exception as e:  # noqa: BLE001
                hyps[it["uid"]] = ""
                errs[it["uid"]] = f"{type(e).__name__}: {e}"
            secs[it["uid"]] = round(time.time() - t0, 3)
        if n % 25 == 0:
            print(f"  [gpu{a.worker}] {n}/{len(by_sid)} hội thoại, {time.time() - t_all:.0f}s", flush=True)
    json.dump({"hyps": hyps, "secs": secs, "errors": errs, "total_s": round(time.time() - t_all, 1)},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="fisherman611/pw-tmp-c3d9")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--n-gpu", type=int, default=2)
    ap.add_argument("--worker", type=int)
    ap.add_argument("--shard-file")
    a = ap.parse_args()
    if a.worker is not None:
        return worker(a)
    m = json.load(open(os.path.join(a.root, "manifest.json"), encoding="utf-8"))["items"]
    by = collections.defaultdict(list)
    for it in m:
        by[it["sid"]].append(it)
    shards = [[] for _ in range(a.n_gpu)]
    for sid in sorted(by, key=lambda s: -len(by[s])):            # cân bằng số đoạn giữa các shard
        min(shards, key=len).extend(by[sid])
    print("shard:", [len(s) for s in shards], flush=True)
    procs, outs = [], []
    for i, sh in enumerate(shards):
        sf_, of_ = f"/tmp/shard{i}.json", f"/tmp/out{i}.json"
        json.dump(sh, open(sf_, "w", encoding="utf-8"), ensure_ascii=False)
        outs.append(of_)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--root", a.root, "--model", a.model, "--out", of_, "--beams", str(a.beams),
               "--worker", str(i), "--shard-file", sf_] + (["--no-prompt"] if a.no_prompt else [])
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}))
    t0 = time.time()
    rc = [p.wait() for p in procs]
    res = {"model": a.model, "use_prompt": not a.no_prompt, "num_beams": a.beams, "hyps": {}, "secs": {}, "errors": {}}
    for of_ in outs:
        if os.path.exists(of_):
            r = json.load(open(of_, encoding="utf-8"))
            for k in ("hyps", "secs", "errors"):
                res[k].update(r[k])
    res["total_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"xong rc={rc}: {len(res['hyps'])}/{len(m)} đoạn, lỗi {len(res['errors'])}, {res['total_s']}s → {a.out}", flush=True)


if __name__ == "__main__":
    main()
