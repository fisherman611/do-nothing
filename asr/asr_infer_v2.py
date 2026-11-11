"""A5 (PLAN_48h): infer ASR v2 (chỉ infer) để lấy lỗi của chính v2 cho dữ liệu FC. Với mỗi đoạn: nbest[0] = beam search (đầu ra thật), các mục sau = phương án thay thế lấy mẫu T=0.7 kèm tần suất.
which: 'val' (405 đoạn thật giữ riêng, đúng điều kiện thi: prompt dùng transcript của chính model cho lượt trước),
       'tts' (đoạn TTS giữ riêng, split tts_holdout), 'train_aug' (n_train đoạn train thật đã làm nhiễu — audio khác thứ model đã học).
Chia shard theo sid (giữ nguyên hội thoại)."""
import collections
import hashlib
import json
import os
import random


def run(tag, which, shard=0, n_shards=1, n_train=1500, beams=5, nbest=6, limit=0, sids=()):
    """which: val (405 giữ riêng) | tts (TTS giữ riêng) | train_aug (audio train làm nhiễu) | fcval (các sid trong `sids`,
    vd val_in+val_ood của FC — phần lớn in-sample với ASR, chỉ để chạy pipeline với transcript thật)."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import asr_train as T
    import bias_prompt as bp
    # tag: thư mục trong OUT, hoặc repo HF "user/name" (vd fisherman611/w-m4q9 = ASR v2 medium) — kéo thẳng từ HF
    path = tag if ("/" in tag and not os.path.isdir(f"{T.OUT}/{tag}")) else f"{T.OUT}/{tag}"
    proc = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    tok, sot = proc.tokenizer, model.config.decoder_start_token_id
    man = json.load(open(f"{T.DATA}/manifest_v2.json")) if which in ("val", "fcval", "train_aug") else None   # manifest train/val
    if which == "val":
        items = [x for x in man if x["split"] == "val"]
    elif which == "public":
        # public test: manifest dựng bằng asr_srv/make_public_manifest_v2.py (items_from_row, không nhãn, root=public_test)
        items = json.load(open(f"{T.DATA}/public_manifest_v2.json"))
    elif which == "fcval":
        want = set(sids)
        items = [x for x in man if x["sid"] in want]
    elif which == "tts":
        items = [x for x in json.load(open(f"{T.DATA}/tts_manifest.json")) if x["split"] == "tts_holdout"]
    else:
        items = random.Random(123).sample([x for x in man if x["split"] == "train"], n_train)
    items = [x for x in items if int(hashlib.md5(x["sid"].encode()).hexdigest(), 16) % n_shards == shard]
    if limit:
        items = items[:limit]
    by_sid = collections.defaultdict(list)
    for it in items:
        by_sid[it["sid"]].append(it)
    hyps, res = {}, {}
    for sid, utts in by_sid.items():
        utts.sort(key=lambda x: (x["kind"] == "final", x["uid"]))
        for it in utts:
            wav = T.load_wavs([it])[0]
            if which == "train_aug":
                wav = T.augment_wav(wav, random.Random(int(hashlib.md5(it["uid"].encode()).hexdigest(), 16) % (1 << 30) + 999))
            pid, ents = bp.build_prompt(tok, it, hyps=hyps if which in ("val", "fcval", "public") else None)
            feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=model.dtype)
            kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
            with torch.no_grad():
                top = model.generate(input_features=feats, language="vi", task="transcribe", num_beams=beams, max_new_tokens=200, **kw)
                # num_return_sequences của Whisper.generate chỉ nhân bản đầu vào (không trả n-best của beam) → lấy phương án thay thế
                # bằng lấy mẫu từ phân phối của model (T=0.7): tần suất xuất hiện ≈ xác suất model "suýt chọn".
                alt = model.generate(input_features=feats.repeat(nbest, 1, 1), language="vi", task="transcribe", do_sample=True, temperature=0.7,
                                     num_beams=1, max_new_tokens=200, **kw) if nbest > 0 else []
            dec = lambda rows: [t.strip() for t in proc.batch_decode([T.strip_after_sot(r, sot) for r in rows], skip_special_tokens=True)]   # noqa: E731
            t1 = dec(top.tolist())[0]
            cnt = collections.Counter(dec(alt.tolist())) if nbest > 0 else collections.Counter()
            nb = [[t1, None]] + ([[t, c / nbest] for t, c in cnt.most_common() if t != t1] if nbest > 0 else [])
            hyps[it["uid"]] = nb[0][0]
            res[it["uid"]] = {"sid": sid, "kind": it["kind"], "ref": it["text"], "nbest": nb, "prompt_ents": ents}
    return res


if __name__ == "__main__":
    # chạy ngoài Modal (myserver): ZAIC_DATA=<dir manifest + audio> ZAIC_OUT=<dir chứa model ở <tag>/>
    #   python asr_infer_v2.py <tag> <which> <out.json> [--nbest 0] [--shard i/n]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("which")
    ap.add_argument("out")
    ap.add_argument("--nbest", type=int, default=0)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    res = run(a.tag, a.which, shard=si, n_shards=sn, beams=a.beams, nbest=a.nbest, limit=a.limit)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)} utt → {a.out}")
