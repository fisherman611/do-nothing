"""Chép lời bằng ASR v3 (model gộp trên volume `zaic-asr-v3-out/<tag>`) hoặc model HF, ĐÚNG điều kiện thi như
agent/asr_modal.py (bản đã tạo transcript public 62.76 của e46f): mỗi hội thoại chạy u1, u2, ..., final; prompt = transcript
của chính model cho các lượt user trước + text assistant; beam 5; fp16. App riêng, không đụng app train.

Audio trên volume `zaic-asr-v3-data`: train/audio (held-out), public_test/audio, tts_v2p/, tts_v3/, tts_v3el/.
    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy modal/modal_asr_infer_v3.py
    python modal/modal_asr_infer_v3.py upload_public
    python modal/modal_asr_infer_v3.py run /out/<tag> <ra.json>        # held-out + public + TTS val
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ZAIC = os.path.dirname(HERE)
APP = "lifemate-asr-infer-v3"
app = modal.App(APP)
data_vol = modal.Volume.from_name("zaic-asr-v3-data", create_if_missing=True)
out_vol = modal.Volume.from_name("zaic-asr-v3-out", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("libsndfile1")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "soundfile", "numpy<2.3", "huggingface_hub")
         .env({"HF_HOME": "/out/hf_cache"}))
SOP, MAX_PROMPT_TOKENS = "<|startofprev|>", 180


@app.function(image=image, gpu="L40S", cpu=4, volumes={"/data": data_vol, "/out": out_vol}, timeout=4 * 3600)
def transcribe(items: list, model_id: str, num_beams: int = 5, use_prompt: bool = True):
    """items: [{uid, sid, kind, path (tương đối /data), prev}] → {uid: text}."""
    import collections, time
    import soundfile as sf, torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    sot, tok = model.config.decoder_start_token_id, proc.tokenizer

    def prompt_ids(prev, hyps):
        parts = []
        for role, text, uid in prev:
            if role == "user" and uid is not None:
                text = hyps.get(uid, "")
            if text: parts.append(text.strip())
        if not parts: return []
        return [tok.convert_tokens_to_ids(SOP)] + tok(" " + " ".join(parts), add_special_tokens=False).input_ids[-MAX_PROMPT_TOKENS:]

    def run(wav, pid):
        feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=num_beams, max_new_tokens=200, **kw)[0].tolist()
        if sot in ids: ids = ids[ids.index(sot):]
        return proc.batch_decode([ids], skip_special_tokens=True)[0].strip()

    by_sid = collections.defaultdict(list)
    for it in items: by_sid[it["sid"]].append(it)
    hyps, errs, t0 = {}, {}, time.time()
    for n, (sid, utts) in enumerate(by_sid.items()):
        utts.sort(key=lambda x: (x["kind"] == "final", x["uid"]))
        for it in utts:
            try:
                wav, sr = sf.read(f"/data/{it['path']}", dtype="float32")
                wav = wav.mean(1) if wav.ndim > 1 else wav
                hyps[it["uid"]] = run(wav, prompt_ids(it.get("prev") or [], hyps) if use_prompt else [])
            except Exception as e:  # noqa: BLE001
                hyps[it["uid"]] = ""; errs[it["uid"]] = f"{type(e).__name__}: {e}"
        if n % 200 == 0: print(f"{n}/{len(by_sid)} hội thoại, {time.time() - t0:.0f}s", flush=True)
    return {"model": model_id, "use_prompt": use_prompt, "num_beams": num_beams, "hyps": hyps, "errors": errs, "total_s": round(time.time() - t0, 1)}


def build_items():
    """held-out 405 + public (1 933 đoạn) + TTS val (giọng chưa nghe)."""
    held = json.load(open(f"{ZAIC}/agent/runs/asr/asr_heldout.manifest.json"))["items"]
    pub = json.load(open(f"{ZAIC}/agent/runs/asr/public_test.manifest.json"))["items"]
    tts = json.load(open(f"{ZAIC}/data/tts_v3/tts_manifest_v3.json"))
    it = [{"uid": h["uid"], "sid": h["sid"], "kind": h["kind"], "path": f"train/{h['audio']}", "prev": h.get("prev") or [], "set": "heldout"} for h in held]
    it += [{"uid": p["uid"], "sid": p["sid"], "kind": p["kind"], "path": f"public_test/{p['audio']}", "prev": p.get("prev") or [], "set": "public"} for p in pub]
    it += [{"uid": f"{x['root']}/{x['audio'][:-4]}", "sid": f"{x['root']}/{x['audio'][:-4]}", "kind": "final", "path": f"{x['root']}/{x['audio']}",
            "prev": [], "set": "tts"} for x in tts if x["split"] in ("val", "val_voice")]
    return it


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "upload_public":
        with data_vol.batch_upload(force=True) as b:
            b.put_directory(f"{ZAIC}/data/train+public_test/public_test/audio", "/public_test/audio")
        print("upload public xong")
    elif cmd == "run":
        model_id, out = sys.argv[2], sys.argv[3]
        items = build_items(); print(len(items), "đoạn:", {s: sum(i["set"] == s for i in items) for s in ("heldout", "public", "tts")})
        res = modal.Function.from_name(APP, "transcribe").remote(items, model_id)
        sets = {i["uid"]: i["set"] for i in items}
        res["by_set"] = {s: {u: h for u, h in res["hyps"].items() if sets.get(u) == s} for s in ("heldout", "public", "tts")}
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        print("xong", len(res["hyps"]), "đoạn, lỗi", len(res["errors"]), "→", out)
