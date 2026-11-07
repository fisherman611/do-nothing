"""Fine-tune PhoWhisper-large (LoRA) on Modal.

Loss: CE (+ label smoothing) | weighted CE theo token (--entity-weight) | joint CTC+attention (--ctc-weight).
Contextual biasing (--use-prompt): đưa transcript các lượt trước (user + assistant) vào decoder sau
<|startofprev|>; loss chỉ tính trên phần transcript. Khi eval tính 2 số: prompt = nhãn thật ("oracle")
và prompt = transcript do chính model sinh cho các lượt user trước đó ("real", đúng điều kiện thi).

Dữ liệu: Modal volume `zaic-data` (manifest.json + train/audio), kết quả: volume `zaic-out/<tag>`.
Chia train/val theo sample id (xem manifest). Mỗi lượt ghi vào thư mục mới theo tag, không ghi đè.

Chạy tách khỏi máy local (tắt máy vẫn chạy):
    modal deploy asr/modal_finetune_phowhisper.py
    python -c "import modal; f=modal.Function.from_name('lifemate-phowhisper-finetune','train'); \
               print(f.spawn(label_smoothing=0.1, ctc_weight=0.3, use_prompt=True).object_id)"
Chạy thường:  modal run asr/modal_finetune_phowhisper.py --label-smoothing 0.1 --ctc-weight 0.3 --use-prompt
"""

import modal

MODEL_ID = "vinai/PhoWhisper-large"
MODEL_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"

app = modal.App("lifemate-phowhisper-finetune")
data_vol = modal.Volume.from_name("zaic-data")
out_vol = modal.Volume.from_name("zaic-out")
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1",
                 "jiwer", "librosa", "soundfile", "numpy<2.3")
)


def norm(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_wavs(items):
    import soundfile as sf
    return [sf.read(f"/data/train/{i['audio']}")[0].astype("float32") for i in items]


SOP = "<|startofprev|>"


def prompt_ids_for(tokenizer, prev_text, max_tokens):
    """[<|startofprev|>] + token của ngữ cảnh (giữ `max_tokens` token cuối). Cùng dạng với tokenizer.get_prompt_ids."""
    if not prev_text:
        return []
    ids = tokenizer(" " + prev_text.strip(), add_special_tokens=False).input_ids[-max_tokens:]
    return [tokenizer.convert_tokens_to_ids(SOP)] + ids


def prev_text_of(item, hyps=None):
    """Ghép các lượt trước thành một đoạn. `hyps`: uid -> transcript ASR, thay cho nhãn thật của lượt user."""
    parts = []
    for role, text, uid in item.get("prev", []):
        if role == "user" and hyps is not None:
            text = hyps.get(uid, "")
        if text:
            parts.append(text.strip())
    return " ".join(parts)


def strip_after_sot(ids, sot):
    """Bỏ phần prompt trong output của generate: giữ từ token <|startoftranscript|> trở đi."""
    ids = list(ids)
    return ids[ids.index(sot):] if sot in ids else ids


def transcribe(model, proc, items, bs=16, prompts=None, max_tokens=180):
    """prompts: None (không prompt, batch) hoặc list text ngữ cảnh cùng độ dài items (batch 1, prompt_ids 1-D)."""
    import torch
    model.eval()
    sot = model.config.decoder_start_token_id
    hyp = []
    if prompts is None:
        for i in range(0, len(items), bs):
            feats = proc.feature_extractor(load_wavs(items[i:i + bs]), sampling_rate=16000,
                                           return_tensors="pt").input_features.to("cuda", dtype=model.dtype)
            with torch.no_grad():
                ids = model.generate(input_features=feats, language="vi", task="transcribe",
                                     num_beams=5, max_new_tokens=200)
            hyp += proc.batch_decode([strip_after_sot(r, sot) for r in ids.tolist()], skip_special_tokens=True)
        return hyp
    for it, p in zip(items, prompts):
        feats = proc.feature_extractor(load_wavs([it]), sampling_rate=16000,
                                       return_tensors="pt").input_features.to("cuda", dtype=model.dtype)
        pid = prompt_ids_for(proc.tokenizer, p, max_tokens)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe",
                                 num_beams=5, max_new_tokens=200, **kw)
        hyp += proc.batch_decode([strip_after_sot(ids[0].tolist(), sot)], skip_special_tokens=True)
    return hyp


def transcribe_conv(model, proc, items, max_tokens=180):
    """Điều kiện thi: prompt lấy từ transcript ASR của các lượt user trước (assistant là text có sẵn).
    Chạy theo thứ tự lượt trong từng hội thoại."""
    import collections
    by_sid = collections.defaultdict(list)
    for it in items:
        by_sid[it["sid"]].append(it)
    hyps = {}
    for sid, utts in by_sid.items():
        utts.sort(key=lambda x: (x["kind"] == "final", x["uid"]))   # u1, u2, ..., final
        for it in utts:
            hyps[it["uid"]] = transcribe(model, proc, [it], prompts=[prev_text_of(it, hyps)], max_tokens=max_tokens)[0]
    return [hyps[it["uid"]] for it in items]


def rare_words(texts, coverage=0.9):
    """Từ nằm ngoài nhóm phổ biến nhất chiếm `coverage` số lần xuất hiện (cách của B-Whisper)."""
    import collections
    import re
    cnt = collections.Counter(w for t in texts for w in re.findall(r"\w+", norm(t)))
    total, acc, rare = sum(cnt.values()), 0, set()
    for w, c in cnt.most_common():
        if acc / total >= coverage:
            rare.add(w)
        acc += c
    return rare


def token_weights(text, tokenizer, rare, beta):
    """Trọng số từng token: beta cho từ hiếm / có chữ số / tên riêng (chữ hoa không đầu câu), còn lại 1."""
    import re
    spans = []
    for k, m in enumerate(re.finditer(r"\w+", text)):
        w = m.group()
        heavy = bool(re.search(r"\d", w)) or norm(w) in rare or (k > 0 and w[0].isupper())
        spans.append((m.start(), m.end(), heavy))
    enc = tokenizer(text, return_offsets_mapping=True)
    out = []
    for a, b in enc["offset_mapping"]:
        hit = a < b and any(s < b and a < e and h for s, e, h in spans)
        out.append(beta if hit else 1.0)
    return enc["input_ids"], out


def score(refs, hyp):
    import jiwer
    r, h = [norm(x) for x in refs], [norm(x) for x in hyp]
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h)}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": data_vol, "/out": out_vol}, timeout=4 * 3600)
def train(epochs: int = 6, lr: float = 3e-4, label_smoothing: float = 0.0, full_data: bool = False,
          seed: int = 42, entity_weight: float = 1.0, ctc_weight: float = 0.0, resume: bool = True,
          use_prompt: bool = False, prompt_drop: float = 0.4, prompt_max_tokens: int = 180):
    import json
    import random
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    manifest = json.load(open("/data/manifest.json"))
    tr = [x for x in manifest if full_data or x["split"] == "train"]
    va = [x for x in manifest if x["split"] == "val"]
    proc = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    proc.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                            torch_dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    sot = model.config.decoder_start_token_id
    rare = rare_words([x["text"] for x in tr]) if entity_weight != 1.0 else set()
    print("rare words:", len(rare), "entity_weight:", entity_weight)
    eos = proc.tokenizer.eos_token_id
    tok_map = {}
    if ctc_weight > 0:
        # CTC head phụ trên encoder, chỉ dùng lúc train (bỏ đi khi suy luận). Rút gọn vocab về các token xuất hiện
        # trong nhãn train (blank = 0) để tiết kiệm bộ nhớ so với vocab 51k.
        used = sorted({t for x in tr for t in proc.tokenizer(x["text"]).input_ids if t < eos})
        tok_map = {t: k + 1 for k, t in enumerate(used)}
        model.ctc_head = torch.nn.Linear(model.config.d_model, len(used) + 1)
        print("ctc vocab:", len(used) + 1, "alpha:", ctc_weight)
    pad = proc.tokenizer.pad_token_id
    if use_prompt:
        n_prev = sum(bool(x.get("prev")) for x in tr)
        print(f"prompt: {n_prev}/{len(tr)} utt train có ngữ cảnh, drop {prompt_drop}, max {prompt_max_tokens} token")

    class DS(torch.utils.data.Dataset):
        def __init__(self, items, training):
            self.items, self.training = items, training

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            x = self.items[i]
            ids, w = token_weights(x["text"], proc.tokenizer, rare, entity_weight)
            wav = load_wavs([x])[0]
            p = []
            # Train: bỏ prompt ngẫu nhiên để model không phụ thuộc hoàn toàn vào ngữ cảnh (lượt đầu không có).
            # Eval: luôn có prompt (= nhãn thật của lượt trước, "oracle").
            if use_prompt and (not self.training or random.random() >= prompt_drop):
                p = prompt_ids_for(proc.tokenizer, prev_text_of(x), prompt_max_tokens)
            return {"f": proc.feature_extractor(wav, sampling_rate=16000).input_features[0],
                    "l": ids, "w": w, "n": len(wav), "p": p}

    def collate(batch):
        feats = torch.tensor(np.stack([x["f"] for x in batch]))
        # Chuỗi decoder đầy đủ: [prompt...] [SOT] [vi] [transcribe] [notimestamps] text... [EOT]
        #   decoder_input_ids = full[:-1], labels = full[1:], phần prompt (và vị trí sinh SOT) gán -100.
        # Không prompt thì trùng với cách cũ: dec = [SOT] + labels[:-1]. Trainer nhận decoder_input_ids
        # nên không tự dịch labels nữa (cần cho label smoothing / weighted CE).
        rows = []
        for x in batch:
            l = x["l"] if x["l"][0] == sot else [sot] + x["l"]
            w = x["w"] if x["l"][0] == sot else [0.0] + x["w"]
            full = x["p"] + l
            n_p = len(x["p"])
            rows.append((full[:-1], [-100] * n_p + l[1:], [0.0] * n_p + w[1:]))
        width = max(len(d) for d, _, _ in rows)
        dec = torch.full((len(batch), width), pad)
        labels = torch.full((len(batch), width), -100)
        weights = torch.zeros(len(batch), width)
        for i, (d, lab, w) in enumerate(rows):
            dec[i, :len(d)] = torch.tensor(d)
            labels[i, :len(lab)] = torch.tensor(lab)
            weights[i, :len(w)] = torch.tensor(w)
        batch_out = {"input_features": feats, "labels": labels, "decoder_input_ids": dec}
        if use_prompt:
            batch_out["prompt_ids"] = [x["p"] for x in batch]   # chỉ dùng ở prediction_step
        if entity_weight != 1.0:
            batch_out["loss_weights"] = weights
        if ctc_weight > 0:
            tg = [[tok_map[t] for t in x["l"] if t < eos and t in tok_map] for x in batch]
            batch_out["ctc_targets"] = torch.tensor([t for row in tg for t in row])
            batch_out["ctc_tlens"] = torch.tensor([len(row) for row in tg])
            # encoder Whisper: 1 frame / 320 mẫu (50 frame/s), tối đa 1500
            batch_out["ctc_ilens"] = torch.tensor([min(1500, -(-x["n"] // 320)) for x in batch])
        return batch_out

    def metrics(p):
        lab = p.label_ids
        lab[lab == -100] = pad
        pred = p.predictions
        pred[pred == -100] = pad
        return score(proc.batch_decode(lab, skip_special_tokens=True),
                     proc.batch_decode(pred, skip_special_tokens=True))

    class WeightedTrainer(Seq2SeqTrainer):
        """CE có trọng số theo token (+ label smoothing) [+ CTC]. Lúc eval bỏ các key phụ khỏi batch;
        khi có prompt thì tự gọi generate từng mẫu với prompt_ids (HF chỉ nhận prompt 1-D cho cả batch)."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            weights = inputs.pop("loss_weights", None)
            inputs.pop("prompt_ids", None)
            ctc = [inputs.pop(k, None) for k in ("ctc_targets", "ctc_tlens", "ctc_ilens")]
            out = model(**inputs)
            logits = out.logits.float()
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                reduction="none", label_smoothing=label_smoothing).view_as(labels)
            valid = (labels != -100).float()
            w = valid if weights is None else weights.to(ce.device) * valid
            loss = (ce * w).sum() / w.sum()
            if ctc_weight > 0 and ctc[0] is not None:
                # L = (1 - alpha) * L_attention + alpha * L_ctc (Watanabe et al., 2017)
                lp = torch.log_softmax(model.ctc_head(out.encoder_last_hidden_state).float(), dim=-1)
                l_ctc = torch.nn.functional.ctc_loss(lp.transpose(0, 1), ctc[0], ctc[2], ctc[1],
                                                     blank=0, reduction="mean", zero_infinity=True)
                loss = (1 - ctc_weight) * loss + ctc_weight * l_ctc
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **k):
            drop = ("loss_weights", "ctc_targets", "ctc_tlens", "ctc_ilens")
            prompts = inputs.pop("prompt_ids", None)
            inputs = {key: v for key, v in inputs.items() if key not in drop}
            if not use_prompt:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **k)
            labels = inputs["labels"]
            outs = []
            with torch.no_grad():
                for i in range(labels.size(0)):
                    kw = {"prompt_ids": torch.tensor(prompts[i], device=labels.device)} if prompts[i] else {}
                    ids = model.generate(input_features=inputs["input_features"][i:i + 1], language="vi",
                                         task="transcribe", max_new_tokens=200, **kw)[0].tolist()
                    outs.append(strip_after_sot(ids, sot))
            width = max(len(o) for o in outs)
            pred = torch.full((len(outs), width), pad, device=labels.device)
            for i, o in enumerate(outs):
                pred[i, :len(o)] = torch.tensor(o, device=labels.device)
            return None, pred, labels

    from transformers import TrainerCallback

    class CommitCkpt(TrainerCallback):
        """Đẩy checkpoint mỗi epoch lên volume để job bị dừng giữa chừng không mất hết."""

        def on_save(self, args, state, control, **kw):
            out_vol.commit()

    weighted = entity_weight != 1.0 or ctc_weight > 0 or use_prompt   # dùng trainer tự viết
    tag = (f"lora_ls{label_smoothing}{'_w' + str(entity_weight) if entity_weight != 1.0 else ''}"
           f"{'_ctc' + str(ctc_weight) if ctc_weight > 0 else ''}{'_prompt' if use_prompt else ''}"
           f"{'_full' if full_data else ''}")
    # Có prompt thì chuỗi decoder dài ~3x → batch 16 bị OOM trên A100-80GB; giữ 32 mẫu/bước bằng accumulation.
    bs, accum = (8, 4) if use_prompt else (16, 2)
    args = Seq2SeqTrainingArguments(
        output_dir=f"/out/{tag}_ckpt", per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type="cosine", num_train_epochs=epochs,
        bf16=True, label_smoothing_factor=0.0 if weighted else label_smoothing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=False,   # giữ optimizer/scheduler để resume được
        
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=200, per_device_eval_batch_size=16,
        logging_steps=20, report_to=[], remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=4, seed=seed)
    trainer = (WeightedTrainer if weighted else Seq2SeqTrainer)(model=model, args=args, train_dataset=DS(tr, True), eval_dataset=DS(va, False),
                             data_collator=collate, compute_metrics=metrics,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CommitCkpt()])
    from transformers.trainer_utils import get_last_checkpoint
    import os
    last = None
    if resume and ctc_weight == 0 and os.path.isdir(args.output_dir):
        last = get_last_checkpoint(args.output_dir)   # CTC head không nằm trong adapter nên không resume được
    print("resume from:", last)
    trainer.train(resume_from_checkpoint=last)
    model = trainer.model.merge_and_unload().to(torch.bfloat16).cuda()
    refs = [x["text"] for x in va]
    extra = {}
    if use_prompt:
        hyp_oracle = transcribe(model, proc, va, prompts=[prev_text_of(x) for x in va], max_tokens=prompt_max_tokens)
        hyp_none = transcribe(model, proc, va)
        hyp = transcribe_conv(model, proc, va, max_tokens=prompt_max_tokens)   # điều kiện thi → số chính
        res = score(refs, hyp)
        extra = {"res_prompt_oracle": score(refs, hyp_oracle), "res_no_prompt": score(refs, hyp_none),
                 "pairs_prompt_oracle": [(x["uid"], x["text"], h) for x, h in zip(va, hyp_oracle)],
                 "pairs_no_prompt": [(x["uid"], x["text"], h) for x, h in zip(va, hyp_none)]}
        print("oracle prompt:", extra["res_prompt_oracle"], "| no prompt:", extra["res_no_prompt"])
    else:
        hyp = transcribe(model, proc, va)
        res = score(refs, hyp)
    out = f"/out/{tag}"
    model.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    json.dump({"res": res, "config": {"epochs": epochs, "lr": lr, "label_smoothing": label_smoothing,
                                      "full_data": full_data, "seed": seed, "entity_weight": entity_weight, "ctc_weight": ctc_weight, "resumed_from": last,
                                      "use_prompt": use_prompt, "prompt_drop": prompt_drop, "prompt_max_tokens": prompt_max_tokens,
                                      "revision": MODEL_REVISION},
               "log": trainer.state.log_history,
               "pairs": [(x["uid"], x["text"], h) for x, h in zip(va, hyp)], **extra},
              open(f"{out}/eval.json", "w"), ensure_ascii=False, indent=1)
    out_vol.commit()
    print("FINAL", tag, res)
    return res


@app.local_entrypoint()
def main(epochs: int = 6, lr: float = 3e-4, label_smoothing: float = 0.0, full_data: bool = False,
         entity_weight: float = 1.0, ctc_weight: float = 0.0, resume: bool = True,
         use_prompt: bool = False, prompt_drop: float = 0.4, prompt_max_tokens: int = 180):
    print(train.remote(epochs=epochs, lr=lr, label_smoothing=label_smoothing, full_data=full_data,
                       entity_weight=entity_weight, ctc_weight=ctc_weight, resume=resume,
                       use_prompt=use_prompt, prompt_drop=prompt_drop, prompt_max_tokens=prompt_max_tokens))
