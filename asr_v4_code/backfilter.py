"""Lọc vòng ngược TTS P1 (ASR v4): e46f nghe 1 499 file đã augment (greedy, không prompt), so với nhãn (cả hai đổi số → chữ,
bỏ dấu câu, chữ thường). CER > NGƯỠNG → split="drop". Ghi DATA/tts_manifest_pub_f.json (+ bf_cer, bf_hyp) và in thống kê.
    ZAIC_DATA=<data/asr_v3> CUDA_VISIBLE_DEVICES=0 venv/bin/python backfilter.py [--thr 0.15]
"""
import collections, json, os, re, sys, unicodedata

DATA = os.environ.get("ZAIC_DATA", "/data")
MODEL, REV = "fisherman611/pw-tmp-e46f", "85081b30df5a9de0b792aa82a7af7876da74a253"
THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.15
D = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def two(n):
    if n < 10: return D[n]
    c, u = divmod(n, 10)
    if c == 1: return "mười" + ("" if u == 0 else " lăm" if u == 5 else " " + D[u])
    return D[c] + " mươi" + ("" if u == 0 else " mốt" if u == 1 else " tư" if u == 4 else " lăm" if u == 5 else " " + D[u])


def three(n):
    t, r = divmod(n, 100)
    if t == 0: return two(r)
    return D[t] + " trăm" + ("" if r == 0 else f" linh {D[r]}" if r < 10 else " " + two(r))


def n2w(n):
    if n < 1000: return three(n)
    out = []; m, r = divmod(n, 1_000_000)
    if m: out.append(three(m) + " triệu")
    k, r2 = divmod(r, 1000)
    if k: out.append(three(k) + " nghìn")
    if r2: out.append(three(r2))
    return " ".join(out)


def norm(s):
    s = re.sub(r"(\d)[.,](\d{3})\b", r"\1\2", s or "")
    s = re.sub(r"\d+", lambda m: " ".join(D[int(c)] for c in m.group(0)) if m.group(0).startswith("0") or len(m.group(0)) >= 9 else n2w(int(m.group(0))), s)
    s = unicodedata.normalize("NFC", s.lower()).replace("ngàn", "nghìn")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    import jiwer, soundfile as sf, torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    man = json.load(open(f"{DATA}/tts_manifest_pub.json", encoding="utf-8"))
    proc = WhisperProcessor.from_pretrained(MODEL, revision=REV)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16).cuda().eval()
    B = 16
    for i in range(0, len(man), B):
        batch = man[i:i + B]
        wavs = [sf.read(f"{DATA}/{x['root']}/{x['audio']}")[0].astype("float32") for x in batch]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=1, max_new_tokens=200)
        hyps = proc.batch_decode(out, skip_special_tokens=True)
        for x, h in zip(batch, hyps):
            r, hh = norm(x["text"]), norm(h)
            x["bf_hyp"] = h.strip(); x["bf_cer"] = round(jiwer.cer(r, hh) if r else 1.0, 4)
            if x["bf_cer"] > THR: x["split"] = "drop"
        if (i // B) % 10 == 0: print(f"{i + len(batch)}/{len(man)}", flush=True)
    json.dump(man, open(f"{DATA}/tts_manifest_pub_f.json", "w", encoding="utf-8"), ensure_ascii=False)
    keep = [x for x in man if x["split"] == "train"]
    print(f"giữ {len(keep)}/{len(man)} (ngưỡng CER {THR}) | CER trung vị {sorted(x['bf_cer'] for x in man)[len(man) // 2]:.3f}")
    for k in ("region", "voice"):
        c = collections.Counter(x[k] for x in man); d = collections.Counter(x[k] for x in man if x["split"] == "drop")
        print(k, {v: f"bỏ {d[v]}/{c[v]}" for v in sorted(c)})
    print("miền sau lọc:", dict(collections.Counter(x["region"] for x in keep)))
    for x in sorted(man, key=lambda z: -z["bf_cer"])[:5]: print("  ví dụ bỏ:", x["bf_cer"], "|", x["text"][:70], "→", x["bf_hyp"][:70])


if __name__ == "__main__":
    main()
