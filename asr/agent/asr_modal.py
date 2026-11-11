"""Chạy PhoWhisper (bản fine-tune trên HF) trên một manifest (agent/asr_manifest.py), audio nằm trên volume `zaic-audio`.
Điều kiện thi: mỗi hội thoại chạy theo thứ tự u1, u2, ..., final; prompt = transcript ASR các lượt user trước + text
assistant (cách v1 của asr_train.py). Trả về {uid: text} + thời gian từng đoạn.

    modal deploy agent/asr_modal.py
    python -c "import modal,json; f=modal.Function.from_name('lifemate-asr','transcribe_manifest'); \\
        m=json.load(open('agent/runs/asr/public_test.manifest.json')); \\
        out=f.remote(m['items'], 'public_test', 'fisherman611/pw-tmp-c3d9', True); json.dump(out, open('agent/runs/asr/public_test.asr.json','w'), ensure_ascii=False)"
Upload audio: modal volume put zaic-audio data/train+public_test/public_test/audio /public_test/audio
"""
import os

import modal

app = modal.App("lifemate-asr")
audio = modal.Volume.from_name("zaic-audio", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libsndfile1")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "soundfile", "numpy", "huggingface_hub[hf_transfer]")
         .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))

SOP = "<|startofprev|>"
MAX_PROMPT_TOKENS = 180


def _prompt_ids(tok, prev, hyps):
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


@app.function(image=image, gpu="L40S", volumes={"/audio": audio, "/hf": hf}, timeout=4 * 3600)
def transcribe_manifest(items: list, split: str, model_id: str, use_prompt: bool = True, num_beams: int = 5):
    import collections
    import time
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    sot = model.config.decoder_start_token_id
    tok = proc.tokenizer

    def run(wav, pid):
        feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=num_beams,
                                 max_new_tokens=200, **kw)[0].tolist()
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
                wav, sr = sf.read(f"/audio/{split}/{it['audio']}", dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                assert sr == 16000, sr
                pid = _prompt_ids(tok, it["prev"], hyps) if use_prompt else []
                hyps[it["uid"]] = run(wav, pid)
            except Exception as e:  # noqa: BLE001
                hyps[it["uid"]] = ""
                errs[it["uid"]] = f"{type(e).__name__}: {e}"
            secs[it["uid"]] = round(time.time() - t0, 3)
        if n % 50 == 0:
            print(f"  {n}/{len(by_sid)} hội thoại, {time.time() - t_all:.0f}s", flush=True)
    return {"model": model_id, "use_prompt": use_prompt, "num_beams": num_beams, "hyps": hyps, "secs": secs, "errors": errs,
            "total_s": round(time.time() - t_all, 1)}


@app.local_entrypoint()
def main(manifest: str, split: str, model_id: str = "fisherman611/pw-tmp-c3d9", out: str = "", no_prompt: bool = False,
         beams: int = 5):
    import json
    m = json.load(open(manifest, encoding="utf-8"))
    res = transcribe_manifest.remote(m["items"], split, model_id, not no_prompt, beams)
    out = out or manifest.replace(".manifest.json", ".asr.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    s = sorted(res["secs"].values())
    print(f"{len(res['hyps'])} đoạn, lỗi {len(res['errors'])}, {res['total_s']}s tổng; p50 {s[len(s) // 2]:.2f}s p95 {s[int(.95 * len(s))]:.2f}s → {out}")
