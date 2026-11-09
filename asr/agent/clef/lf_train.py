"""Train Clef-token bằng LLaMA-Factory, giữ loss của train.py.

LLaMA-Factory lo phần tải model, LoRA, optimizer, scheduler, gradient checkpointing, bf16, nhiều GPU. Ở đây chỉ thay
hai thứ trong workflow SFT:
- data collator: pad phải input_ids, giữ nguyên dữ liệu loss Clef của từng ví dụ (dataset từ lf_prep.py);
- compute_loss: như train.losses — softmax GIỚI HẠN vào token nhãn của từng câu hỏi (label-smoothed CE + Brier),
  CE thường cho giá trị tham số tự do; chỉ tính logit ở các vị trí này. Loss của một ví dụ = w_choice * choice +
  w_lm * lm (thiếu phần nào thì phần đó bằng 0), loss của batch = trung bình các ví dụ.
Pad phải không ảnh hưởng các token thật (causal, kể cả lớp Gated DeltaNet). Không packing.

    python agent/clef/lf_train.py cfg.yaml            # 1 GPU
    torchrun --nproc_per_node 4 agent/clef/lf_train.py cfg.yaml

Tham số loss qua biến môi trường: CLEF_LS (0.1), CLEF_BRIER (1.0), CLEF_W_CHOICE (1.0), CLEF_W_LM (1.0).
"""
import os
import sys
import time

import torch
import torch.nn.functional as Fn

LS = float(os.environ.get("CLEF_LS", "0.1"))
BRIER = float(os.environ.get("CLEF_BRIER", "1.0"))
W_CHOICE = float(os.environ.get("CLEF_W_CHOICE", "1.0"))
W_LM = float(os.environ.get("CLEF_W_LM", "1.0"))
SIDE = ("ch_pos", "ch_allowed", "ch_gold", "ch_kind", "lm_pos", "lm_tgt")


class ClefCollator:
    def __init__(self, pad_id, multiple=8):
        self.pad_id, self.multiple = pad_id, multiple

    def __call__(self, feats):
        T = max(len(f["input_ids"]) for f in feats)
        T = -(-T // self.multiple) * self.multiple
        ids = torch.full((len(feats), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(feats), T), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            mask[i, :n] = 1
        out = {"input_ids": ids, "attention_mask": mask}
        for k in SIDE:
            out[k] = [f[k] for f in feats]
        return out


def clef_loss(model, inputs):
    """→ (loss, {thống kê}) ; giống train.losses cho batch 1."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    dev = h.device
    per_ex, st = [], {"choice": [], "lm": [], "acc_q1": [], "acc_yn": [], "acc_opt": []}
    for b in range(h.shape[0]):
        loss = torch.zeros((), device=dev, dtype=torch.float32)
        pos = inputs["ch_pos"][b]
        if pos:
            logits = base.lm_head(h[b, pos]).float()
            lc, lb = [], []
            for j, (allowed, g, kind) in enumerate(zip(inputs["ch_allowed"][b], inputs["ch_gold"][b],
                                                       inputs["ch_kind"][b])):
                z = logits[j, allowed]
                tgt = torch.tensor([g], device=dev)
                lc.append(Fn.cross_entropy(z[None], tgt, label_smoothing=LS))
                p = z.softmax(-1)
                onehot = torch.zeros_like(p)
                onehot[g] = 1
                lb.append(((p - onehot) ** 2).sum())
                st[("acc_q1", "acc_yn", "acc_opt")[kind]].append(float(z.argmax().item() == g))
            c = torch.stack(lc).mean() + BRIER * torch.stack(lb).mean()
            loss = loss + W_CHOICE * c
            st["choice"].append(c.item())
        lpos = inputs["lm_pos"][b]
        if lpos:
            lm = Fn.cross_entropy(base.lm_head(h[b, lpos]).float(), torch.tensor(inputs["lm_tgt"][b], device=dev))
            loss = loss + W_LM * lm
            st["lm"].append(lm.item())
        per_ex.append(loss)
    return torch.stack(per_ex).mean(), st


def patch_llamafactory():
    import llamafactory.train.sft.workflow as W
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    W.SFTDataCollatorWith4DAttentionMask = lambda **kw: ClefCollator(kw["tokenizer"].pad_token_id)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # loss đã là trung bình theo ví dụ: để Trainer tự chia cho số bước tích luỹ như với loss thường
        self.model_accepts_loss_kwargs = False
        loss, st = clef_loss(model, inputs)
        acc = getattr(self, "_clef_stats", None)
        if acc is None:
            acc = self._clef_stats = {k: [] for k in st}
        for k, v in st.items():
            acc[k] += v
        return (loss, None) if return_outputs else loss

    CustomSeq2SeqTrainer.compute_loss = compute_loss

    orig_log = CustomSeq2SeqTrainer.log

    def log(self, logs, *args, **kwargs):     # thêm loss thành phần + độ chính xác vào log của Trainer
        acc = getattr(self, "_clef_stats", None)
        if acc and "loss" in logs:
            for k, v in acc.items():
                if v:
                    logs[f"clef_{k}"] = round(sum(v) / len(v), 4)
            self._clef_stats = {k: [] for k in acc}
        return orig_log(self, logs, *args, **kwargs)

    CustomSeq2SeqTrainer.log = log


def timing_callback():
    from transformers import TrainerCallback

    class Timing(TrainerCallback):
        """Thời gian mỗi bước tối ưu + bộ nhớ GPU tối đa (bước đầu gồm cả khởi động, xem các bước sau)."""

        def on_step_begin(self, args, state, control, **kw):
            if not hasattr(self, "t"):
                self.t = time.time()

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            if state.is_world_process_zero:
                print({"step": state.global_step, "of": state.max_steps, "step_sec": round(now - self.t, 2),
                       "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}, flush=True)
            self.t = now

    return Timing()


def main(cfg=None):
    """cfg: dict tham số LLaMA-Factory; None → đọc file yaml/json ở sys.argv[1] (như llamafactory-cli train)."""
    patch_llamafactory()
    from llamafactory.train.tuner import run_exp
    if cfg is None and len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        import json                       # read_args của LLaMA-Factory gọi json.load(Path) → lỗi, tự đọc ở đây
        cfg = json.load(open(sys.argv[1]))
    run_exp(args=cfg, callbacks=[timing_callback()])


if __name__ == "__main__":
    main()
