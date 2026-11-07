"""Fine-tune PhoWhisper-large (LoRA) on Modal, with optional label smoothing.

Dữ liệu: Modal volume `zaic-data` (manifest.json + train/audio), kết quả: volume `zaic-out/<tag>`.
Chia train/val theo sample id (xem manifest). Mỗi lượt ghi vào thư mục mới theo tag, không ghi đè.

modal run --profile vyvy372004 asr/modal_finetune_phowhisper.py --label-smoothing 0.1 --epochs 6
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


def transcribe(model, proc, items, bs=16):
    import torch
    model.eval()
    hyp = []
    for i in range(0, len(items), bs):
        feats = proc.feature_extractor(load_wavs(items[i:i + bs]), sampling_rate=16000,
                                       return_tensors="pt").input_features.to("cuda", dtype=model.dtype)
        with torch.no_grad():
            ids = model.generate(input_features=feats, language="vi", task="transcribe",
                                 num_beams=5, max_new_tokens=200)
        hyp += proc.batch_decode(ids, skip_special_tokens=True)
    return hyp


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
          seed: int = 42, entity_weight: float = 1.0):
    import json
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
    pad = proc.tokenizer.pad_token_id

    class DS(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            x = self.items[i]
            ids, w = token_weights(x["text"], proc.tokenizer, rare, entity_weight)
            return {"f": proc.feature_extractor(load_wavs([x])[0], sampling_rate=16000).input_features[0],
                    "l": ids, "w": w}

    def collate(batch):
        feats = torch.tensor(np.stack([x["f"] for x in batch]))
        width = max(len(x["l"]) for x in batch)
        labels = torch.full((len(batch), width), -100)
        weights = torch.zeros(len(batch), width)
        for i, x in enumerate(batch):
            labels[i, :len(x["l"])] = torch.tensor(x["l"])
            weights[i, :len(x["w"])] = torch.tensor(x["w"])
        if (labels[:, 0] == sot).all():
            labels, weights = labels[:, 1:], weights[:, 1:]
        # Khi bật label smoothing, Trainer lấy `labels` ra khỏi batch để tự tính loss,
        # nên phải đưa decoder_input_ids vào (labels dịch phải, thêm token bắt đầu).
        shifted = labels.clone()
        shifted[shifted == -100] = pad
        dec = torch.cat([torch.full((len(batch), 1), sot), shifted[:, :-1]], dim=1)
        batch_out = {"input_features": feats, "labels": labels, "decoder_input_ids": dec}
        if entity_weight != 1.0:
            batch_out["loss_weights"] = weights
        return batch_out

    def metrics(p):
        lab = p.label_ids
        lab[lab == -100] = pad
        return score(proc.batch_decode(lab, skip_special_tokens=True),
                     proc.batch_decode(p.predictions, skip_special_tokens=True))

    class WeightedTrainer(Seq2SeqTrainer):
        """CE có trọng số theo token (+ label smoothing). Lúc eval bỏ `loss_weights` khỏi batch."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            weights = inputs.pop("loss_weights")
            out = model(**inputs)
            logits = out.logits.float()
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                reduction="none", label_smoothing=label_smoothing).view_as(labels)
            w = weights.to(ce.device) * (labels != -100)
            loss = (ce * w).sum() / w.sum()
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, *a, **k):
            inputs = {key: v for key, v in inputs.items() if key != "loss_weights"}
            return super().prediction_step(model, inputs, *a, **k)

    weighted = entity_weight != 1.0
    tag = f"lora_ls{label_smoothing}{'_w' + str(entity_weight) if weighted else ''}{'_full' if full_data else ''}"
    args = Seq2SeqTrainingArguments(
        output_dir=f"/tmp/{tag}", per_device_train_batch_size=16, gradient_accumulation_steps=2,
        learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type="cosine", num_train_epochs=epochs,
        bf16=True, label_smoothing_factor=0.0 if weighted else label_smoothing,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, save_only_model=True,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=200, per_device_eval_batch_size=16,
        logging_steps=20, report_to=[], remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=4, seed=seed)
    trainer = (WeightedTrainer if weighted else Seq2SeqTrainer)(model=model, args=args, train_dataset=DS(tr), eval_dataset=DS(va),
                             data_collator=collate, compute_metrics=metrics,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    model = trainer.model.merge_and_unload().to(torch.bfloat16).cuda()
    hyp = transcribe(model, proc, va)
    res = score([x["text"] for x in va], hyp)
    out = f"/out/{tag}"
    model.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    json.dump({"res": res, "config": {"epochs": epochs, "lr": lr, "label_smoothing": label_smoothing,
                                      "full_data": full_data, "seed": seed, "entity_weight": entity_weight,
                                      "revision": MODEL_REVISION},
               "log": trainer.state.log_history,
               "pairs": [(x["uid"], x["text"], h) for x, h in zip(va, hyp)]},
              open(f"{out}/eval.json", "w"), ensure_ascii=False, indent=1)
    out_vol.commit()
    print("FINAL", tag, res)
    return res


@app.local_entrypoint()
def main(epochs: int = 6, lr: float = 3e-4, label_smoothing: float = 0.0, full_data: bool = False,
         entity_weight: float = 1.0):
    print(train.remote(epochs=epochs, lr=lr, label_smoothing=label_smoothing, full_data=full_data,
                       entity_weight=entity_weight))
