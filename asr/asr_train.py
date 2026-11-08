"""Fine-tune PhoWhisper-large (LoRA), không phụ thuộc Modal: dùng chung cho Modal (modal_finetune_phowhisper.py),
myserver và Kaggle (kaggle/). Xem docstring của modal_finetune_phowhisper.py cho các cờ.

    ZAIC_DATA=<dir có manifest*.json, lexicon_v2.json, train/audio> ZAIC_OUT=<dir ra> \
        python asr_train.py --label-smoothing 0.1 --ctc-weight 0.3 --use-prompt --prompt-v2 [--bs 4 --accum 8]
"""
import os

MODEL_ID = "vinai/PhoWhisper-large"
MODEL_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"
DATA = os.environ.get("ZAIC_DATA", "/data")
OUT = os.environ.get("ZAIC_OUT", "/out")


def norm(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_wavs(items):
    import soundfile as sf
    return [sf.read(f"{DATA}/train/{i['audio']}")[0].astype("float32") for i in items]


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
        pid = p if isinstance(p, list) else prompt_ids_for(proc.tokenizer, p, max_tokens)   # v2: ids dựng sẵn
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe",
                                 num_beams=5, max_new_tokens=200, **kw)
        hyp += proc.batch_decode([strip_after_sot(ids[0].tolist(), sot)], skip_special_tokens=True)
    return hyp


def transcribe_conv(model, proc, items, max_tokens=180, builder=None):
    """Điều kiện thi: prompt lấy từ transcript ASR của các lượt user trước (assistant là text có sẵn).
    Chạy theo thứ tự lượt trong từng hội thoại. builder(it, hyps) -> prompt ids (v2); None thì như v1."""
    import collections
    by_sid = collections.defaultdict(list)
    for it in items:
        by_sid[it["sid"]].append(it)
    hyps = {}
    for sid, utts in by_sid.items():
        utts.sort(key=lambda x: (x["kind"] == "final", x["uid"]))   # u1, u2, ..., final
        for it in utts:
            p = builder(it, hyps) if builder else prev_text_of(it, hyps)
            hyps[it["uid"]] = transcribe(model, proc, [it], prompts=[p], max_tokens=max_tokens)[0]
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


def train(epochs: int = 6, lr: float = 3e-4, label_smoothing: float = 0.0, full_data: bool = False,
          seed: int = 42, entity_weight: float = 1.0, ctc_weight: float = 0.0, resume: bool = True,
          use_prompt: bool = False, prompt_drop: float = 0.4, prompt_max_tokens: int = 180,
          prompt_v2: bool = False, label_ent_prob: float = 0.0, all_distract_prob: float = 0.0,
          n_distract_max: int = 10, section_drop: float = 0.3, bs: int = 0, accum: int = 0,
          max_items: int = 0, eval_items: int = 0, base_fp16: bool = False, time_budget_h: float = 0.0,
          eval_bs: int = 16, final_eval: bool = True, commit=None):
    """commit: hàm gọi sau mỗi lần lưu (Modal: volume.commit). time_budget_h > 0: dừng train, lưu checkpoint khi
    hết giờ (Kaggle 12 h/phiên), trả về {"status": "incomplete"} để phiên sau resume. eval_items: chỉ eval giữa các
    epoch trên N utt đầu của val (eval cuối vẫn trên toàn bộ val). base_fp16: nạp model gốc fp16, LoRA fp32 (GPU 16 GB)."""
    import time
    t_start = time.time()
    commit_out = commit or (lambda: None)
    import json
    import random
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    import bias_prompt as bp
    assert not prompt_v2 or use_prompt, "--prompt-v2 cần --use-prompt"
    manifest = json.load(open(f"{DATA}/manifest_v2.json" if prompt_v2 else f"{DATA}/manifest.json"))
    tr = [x for x in manifest if full_data or x["split"] == "train"]
    va = [x for x in manifest if x["split"] == "val"]
    va_epoch = va
    if eval_items:
        sids = sorted({x["sid"] for x in va})[: max(1, eval_items // 3)]
        va_epoch = [x for x in va if x["sid"] in set(sids)][:eval_items]
    if max_items:   # smoke test: vài hội thoại đầu, giữ nguyên thứ tự lượt
        keep = {x["sid"] for x in manifest[:max_items]}
        tr, va = [x for x in tr if x["sid"] in keep] or tr[:max_items], [x for x in va][:max_items]
    proc = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    proc.tokenizer.set_prefix_tokens(language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                            torch_dtype=torch.float16 if base_fp16 else torch.float32)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    if base_fp16:   # GradScaler cần tham số train ở fp32
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
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
    lex, pcfg = {}, {}
    if prompt_v2:
        lex = json.load(open(f"{DATA}/lexicon_v2.json"))       # chỉ từ split train (make_manifest_v2.py)
        pool = {}
        for fields in lex.values():
            for f, vals in fields.items():
                pool.setdefault(f, []).extend(vals)
        pcfg = dict(label_ent_prob=label_ent_prob, all_distract_prob=all_distract_prob,
                    n_distract_max=n_distract_max, section_drop=section_drop, pool=pool)
        print("prompt v2:", {k: v for k, v in pcfg.items() if k != "pool"}, "| lexicon", len(lex), "app")
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
                if prompt_v2:   # eval: tất định, chỉ nguồn có lúc test (hội thoại dùng nhãn lượt trước)
                    p = bp.build_prompt(proc.tokenizer, x, train=self.training, rng=random, cfg=pcfg)[0]
                else:
                    p = prompt_ids_for(proc.tokenizer, prev_text_of(x), prompt_max_tokens)
            return {"f": proc.feature_extractor(wav, sampling_rate=16000).input_features[0],
                    "l": ids, "w": w, "n": len(wav), "p": p}

    def collate(batch):
        feats = torch.tensor(np.stack([x["f"] for x in batch]))
        if base_fp16:
            feats = feats.half()
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
                lp = torch.log_softmax(self.accelerator.unwrap_model(model).ctc_head(out.encoder_last_hidden_state).float(), dim=-1)
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
                    ids = self.accelerator.unwrap_model(model).generate(input_features=inputs["input_features"][i:i + 1], language="vi",
                                         task="transcribe", max_new_tokens=200, **kw)[0].tolist()
                    outs.append(strip_after_sot(ids, sot))
            width = max(len(o) for o in outs)
            pred = torch.full((len(outs), width), pad, device=labels.device)
            for i, o in enumerate(outs):
                pred[i, :len(o)] = torch.tensor(o, device=labels.device)
            return None, pred, labels

    from transformers import TrainerCallback

    class CommitCkpt(TrainerCallback):
        """Lưu CTC head cạnh checkpoint (adapter không chứa nó) và đẩy lên volume để resume được."""

        def on_save(self, args, state, control, **kw):
            if ctc_weight > 0:
                d = f"{args.output_dir}/checkpoint-{state.global_step}"
                if os.path.isdir(d):
                    torch.save(model.ctc_head.state_dict(), f"{d}/ctc_head.pt")
            commit_out()

    class TimeBudget(TrainerCallback):
        stopped = False

        def on_step_end(self, args, state, control, **kw):
            if time_budget_h and time.time() - t_start > time_budget_h * 3600:
                TimeBudget.stopped = True
                control.should_save = True
                control.should_training_stop = True

    weighted = entity_weight != 1.0 or ctc_weight > 0 or use_prompt   # dùng trainer tự viết
    tag = (f"lora_ls{label_smoothing}{'_w' + str(entity_weight) if entity_weight != 1.0 else ''}"
           f"{'_ctc' + str(ctc_weight) if ctc_weight > 0 else ''}{'_prompt' if use_prompt else ''}"
           f"{'_pv2' if prompt_v2 else ''}{'_le' + str(label_ent_prob) if label_ent_prob else ''}"
           f"{'_ad' + str(all_distract_prob) if all_distract_prob else ''}"
           f"{'_full' if full_data else ''}{f'_smoke{max_items}e{epochs}' if max_items else ''}")
    # Có prompt thì chuỗi decoder dài ~3x → batch 16 bị OOM trên A100-80GB; giữ 32 mẫu/bước bằng accumulation.
    if not bs:
        bs, accum = (8, 4) if use_prompt else (16, 2)
    on_bf16 = torch.cuda.get_device_capability()[0] >= 8   # T4 (sm_75) báo is_bf16_supported()=True nhưng chỉ giả lập, rất chậm → fp16
    args = Seq2SeqTrainingArguments(
        output_dir=f"{OUT}/{tag}_ckpt", per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type="cosine", num_train_epochs=epochs,
        bf16=on_bf16, fp16=not on_bf16, gradient_checkpointing=bs < 8,
        gradient_checkpointing_kwargs={"use_reentrant": False}, label_smoothing_factor=0.0 if weighted else label_smoothing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=False,   # giữ optimizer/scheduler để resume được
        
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=200, per_device_eval_batch_size=16,
        logging_steps=20, report_to=[], remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=4, seed=seed)
    trainer = (WeightedTrainer if weighted else Seq2SeqTrainer)(model=model, args=args, train_dataset=DS(tr, True), eval_dataset=DS(va_epoch, False),
                             data_collator=collate, compute_metrics=metrics,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=2), CommitCkpt(), TimeBudget()])
    from transformers.trainer_utils import get_last_checkpoint
    last = None
    if resume and os.path.isdir(args.output_dir):
        last = get_last_checkpoint(args.output_dir)
        if last and ctc_weight > 0:
            if os.path.exists(f"{last}/ctc_head.pt"):
                model.ctc_head.load_state_dict(torch.load(f"{last}/ctc_head.pt"))
            else:
                print("checkpoint không có ctc_head.pt (bản cũ) → train lại từ đầu")
                last = None
    print("resume from:", last)
    trainer.train(resume_from_checkpoint=last)
    if TimeBudget.stopped:
        print(f"hết thời gian ({time_budget_h} h) ở bước {trainer.state.global_step}: checkpoint đã lưu, chạy lại để resume")
        commit_out()
        return {"status": "incomplete", "step": trainer.state.global_step, "tag": tag}
    trainer.model.save_pretrained(f"{OUT}/{tag}_adapter")
    import gc
    best, log_history = trainer.model, trainer.state.log_history
    del trainer
    gc.collect()
    torch.cuda.empty_cache()   # giải phóng optimizer/activation trước khi giải mã (T4 16 GB)
    model = best.merge_and_unload().to(torch.bfloat16 if on_bf16 else torch.float16).cuda()
    if not final_eval:   # chỉ đo tốc độ train (smoke test)
        return {"status": "train_only", "tag": tag, "log": log_history[-3:]}
    refs = [x["text"] for x in va]
    extra = {}
    if prompt_v2:
        # Điều kiện thi. (a) hội thoại (transcript model) + hồ sơ + tên trong history.
        # (b) (a) + mục lexicon train chọn theo app đã cài và bản nháp: giải mã 2 lượt cho mỗi utt.
        tk = proc.tokenizer
        ents_a, ents_b = {}, {}

        def build_a(it, hyps):
            ids, e = bp.build_prompt(tk, it, hyps=hyps)
            ents_a[it["uid"]] = e
            return ids

        draft_of = {}

        def build_b(it, hyps):
            draft = draft_of[it["uid"]]   # bản nháp = kết quả điều kiện (a) của cùng utt (đỡ một lượt giải mã)
            extra = bp.select_lexicon(tk, lex, it["apps"], draft, exclude=[e[2] for e in it["hist_ents"]])
            ids, e = bp.build_prompt(tk, it, hyps=hyps, extra_ents=extra)
            ents_b[it["uid"]] = e
            return ids

        hyp = transcribe_conv(model, proc, va, builder=build_a)
        draft_of.update({x["uid"]: h for x, h in zip(va, hyp)})
        json.dump({"(a) wer": score(refs, hyp), "pairs": [(x["uid"], x["text"], h) for x, h in zip(va, hyp)]},
                  open(f"{OUT}/{tag}_partial_a.json", "w"), ensure_ascii=False)
        commit_out()
        hyp_b = transcribe_conv(model, proc, va, builder=build_b)
        hyp_none = transcribe(model, proc, va, bs=eval_bs)
        res = score(refs, hyp)
        res["names"] = bp.name_stats(refs, hyp, [ents_a[x["uid"]] for x in va])
        extra = {"res_lexicon": dict(score(refs, hyp_b), names=bp.name_stats(refs, hyp_b, [ents_b[x["uid"]] for x in va])),
                 "res_no_prompt": dict(score(refs, hyp_none), names=bp.name_stats(refs, hyp_none, [ents_a[x["uid"]] for x in va])),
                 "pairs_lexicon": [(x["uid"], x["text"], h, ents_b[x["uid"]]) for x, h in zip(va, hyp_b)],
                 "pairs_no_prompt": [(x["uid"], x["text"], h) for x, h in zip(va, hyp_none)],
                 "prompt_ents": {x["uid"]: ents_a[x["uid"]] for x in va}}
        print("(a):", res, "\n(b) lexicon:", extra["res_lexicon"], "\nno prompt:", extra["res_no_prompt"])
    elif use_prompt:
        hyp_oracle = transcribe(model, proc, va, prompts=[prev_text_of(x) for x in va], max_tokens=prompt_max_tokens)
        hyp_none = transcribe(model, proc, va, bs=eval_bs)
        hyp = transcribe_conv(model, proc, va, max_tokens=prompt_max_tokens)   # điều kiện thi → số chính
        res = score(refs, hyp)
        extra = {"res_prompt_oracle": score(refs, hyp_oracle), "res_no_prompt": score(refs, hyp_none),
                 "pairs_prompt_oracle": [(x["uid"], x["text"], h) for x, h in zip(va, hyp_oracle)],
                 "pairs_no_prompt": [(x["uid"], x["text"], h) for x, h in zip(va, hyp_none)]}
        print("oracle prompt:", extra["res_prompt_oracle"], "| no prompt:", extra["res_no_prompt"])
    else:
        hyp = transcribe(model, proc, va, bs=eval_bs)
        res = score(refs, hyp)
    out = f"{OUT}/{tag}"
    model.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    json.dump({"res": res, "config": {"epochs": epochs, "lr": lr, "label_smoothing": label_smoothing,
                                      "full_data": full_data, "seed": seed, "entity_weight": entity_weight, "ctc_weight": ctc_weight, "resumed_from": last,
                                      "use_prompt": use_prompt, "prompt_drop": prompt_drop, "prompt_max_tokens": prompt_max_tokens,
                                      "prompt_v2": prompt_v2, "label_ent_prob": label_ent_prob, "all_distract_prob": all_distract_prob,
                                      "n_distract_max": n_distract_max, "section_drop": section_drop, "budget": bp.BUDGET,
                                      "bs": bs, "accum": accum, "bf16": on_bf16, "base_fp16": base_fp16,
                                      "eval_items": eval_items, "eval_bs": eval_bs, "gpu": torch.cuda.get_device_name(),
                                      "revision": MODEL_REVISION},
               "log": log_history,
               "pairs": [(x["uid"], x["text"], h) for x, h in zip(va, hyp)], **extra},
              open(f"{out}/eval.json", "w"), ensure_ascii=False, indent=1)
    commit_out()
    print("FINAL", tag, res)
    return res


if __name__ == "__main__":   # cờ dòng lệnh = tham số của train()
    import argparse
    import inspect
    import json
    ap = argparse.ArgumentParser()
    for name, prm in inspect.signature(train).parameters.items():
        if name == "commit":
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(prm.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=prm.default)
        else:
            ap.add_argument(flag, type=type(prm.default), default=prm.default)
    res = train(**vars(ap.parse_args()))
    print("RESULT_JSON", json.dumps(res, ensure_ascii=False))
