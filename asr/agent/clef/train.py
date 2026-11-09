"""Train LoRA cho Qwen3.5-4B (bản cắt) theo định dạng Clef-token (PLAN_FC_clef_qwen35.md mục 4).

Loss chỉ ở các vị trí đã đánh dấu trong dữ liệu (build_data.py):
- choice (nhãn Q1/Q2): softmax GIỚI HẠN vào token nhãn của câu hỏi; label-smoothed CE + Brier (như Clef).
- lm (giá trị tham số tự do + "\\n</parameter>"): CE thường trên toàn vocab.
Chỉ tính logit ở các vị trí này (không tính 248k logit cho cả chuỗi 5–10k token).

    python agent/clef/train.py --model /hf/qwen35_4b_text --train agent/clef/data/train.jsonl \\
        --val agent/clef/data/val_in.jsonl --out runs/clef_lora [--epochs 2 --lr 1e-4 --accum 16]

Cần: transformers hỗ trợ Qwen3_5ForCausalLM, peft; nên có flash-linear-attention + causal-conv1d (không có thì lớp
Gated DeltaNet chạy đường torch rất chậm). Không packing (trạng thái linear-attention sẽ lẫn giữa các mẫu).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent.clef import fmt as F            # noqa: E402
from agent.clef.build_data import TOKENIZER  # noqa: E402


def encode(ex, tok, max_len):
    """→ input_ids, [(pos dự đoán, [id nhãn hợp lệ], id vàng, loại câu hỏi)], [(pos dự đoán, id đích)] hoặc None."""
    enc = tok(ex["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None
    start_tok = {a: i for i, (a, b) in enumerate(offs)}
    choices, lms = [], []
    for j, sp in enumerate(ex["spans"]):
        if sp["type"] == "choice":
            t = start_tok.get(sp["pos"])
            if t is None or t == 0:
                continue
            allowed = [tok(" " + l, add_special_tokens=False).input_ids[0] for l in sp["labels"]]
            gold = tok(" " + sp["gold"], add_special_tokens=False).input_ids[0]
            if ids[t] != gold:
                continue
            kind = "q1" if j == 0 else ("yn" if sp["labels"][0] == F.YES else "opt")   # span đầu luôn là Q1
            choices.append((t - 1, allowed, allowed.index(gold), kind))
        else:
            for i, (a, b) in enumerate(offs):
                if sp["start"] <= a < sp["end"] and i > 0:
                    lms.append((i - 1, ids[i]))
    return ids, choices, lms


def lora_targets(model):
    """Tên lá của mọi nn.Linear trong các lớp decoder (gồm cả lớp Gated DeltaNet), trừ lm_head."""
    import torch.nn as nn
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and ".layers." in n:
            names.add(n.split(".")[-1])
    return sorted(names)


def losses(model, batch, ls, brier_w, device):
    import torch
    import torch.nn.functional as Fn
    ids, choices, lms = batch
    x = torch.tensor([ids], device=device)
    base = model.get_base_model()
    h = base.model(input_ids=x).last_hidden_state[0]            # [T, d]
    out, stats = {}, {}
    if choices:
        pos = torch.tensor([c[0] for c in choices], device=device)
        logits = base.lm_head(h[pos]).float()                    # [n, V]
        lc, lb, acc = [], [], {}
        for j, (_, allowed, g, kind) in enumerate(choices):
            z = logits[j, allowed]
            lc.append(Fn.cross_entropy(z[None], torch.tensor([g], device=device), label_smoothing=ls))
            p = z.softmax(-1)
            onehot = torch.zeros_like(p)
            onehot[g] = 1
            lb.append(((p - onehot) ** 2).sum())
            acc.setdefault(kind, []).append(float(z.argmax().item() == g))
        out["choice"] = torch.stack(lc).mean() + brier_w * torch.stack(lb).mean()
        stats = {k: (sum(v), len(v)) for k, v in acc.items()}
    if lms:
        pos = torch.tensor([p for p, _ in lms], device=device)
        tgt = torch.tensor([t for _, t in lms], device=device)
        out["lm"] = Fn.cross_entropy(base.lm_head(h[pos]).float(), tgt)
    return out, stats


def evaluate(model, data, args, device):
    import torch
    model.eval()
    tot, n, acc = {"choice": 0.0, "lm": 0.0}, {"choice": 0, "lm": 0}, {}
    with torch.no_grad():
        for b in data:
            l, st = losses(model, b, 0.0, args.brier, device)
            for k, v in l.items():
                tot[k] += v.item()
                n[k] += 1
            for k, (c, m) in st.items():
                a = acc.setdefault(k, [0, 0])
                a[0] += c
                a[1] += m
    model.train()
    return {**{f"val_{k}": tot[k] / max(1, n[k]) for k in tot},
            **{f"acc_{k}": round(c / m, 4) for k, (c, m) in acc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="thư mục Qwen3.5-4B bản cắt (strip_qwen35.py)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing cho câu hỏi chọn")
    ap.add_argument("--brier", type=float, default=1.0)
    ap.add_argument("--w-choice", type=float, default=1.0)
    ap.add_argument("--w-lm", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test: chỉ lấy N ví dụ")
    ap.add_argument("--eval-every", type=int, default=0, help="số bước tối ưu giữa hai lần eval (0: mỗi epoch)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--hub-kernels", action="store_true", help="from_pretrained(use_kernels=True), cần gói `kernels`")
    ap.add_argument("--save-every", type=int, default=0,
                    help="lưu checkpoint đầy đủ (adapter + trạng thái train) mỗi N bước tối ưu; 0: chỉ lúc eval / "
                         "cuối mỗi epoch / cuối job. Luôn ghi đè thêm <out>/last để resume sau sự cố")
    ap.add_argument("--resume", help="thư mục checkpoint (vd <out>/last, <out>/step220) để train tiếp đúng chỗ dừng")
    ap.add_argument("--sched-epochs", type=float, default=0,
                    help="số epoch dùng để tính lịch LR (0: = --epochs). Vd chạy --epochs 1 --sched-epochs 2 rồi "
                         "--resume ... --epochs 2: LR giống hệt một lần train 2 epoch liền")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])

    def load(path, limit=0):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            rows = rows[:limit]
        enc = [encode(r, tok, a.max_len) for r in rows]
        kept = [e for e in enc if e and (e[1] or e[2])]
        print(f"{path}: {len(rows)} ví dụ, giữ {len(kept)} (bỏ {len(rows) - len(kept)} quá dài/rỗng)", flush=True)
        return kept

    train = load(a.train, a.max_train)
    val = load(a.val, max(1, a.max_train // 4) if a.max_train else 0) if a.val else []

    try:
        import fla  # noqa: F401
        print("flash-linear-attention: có")
    except ImportError:
        print("CẢNH BÁO: không có flash-linear-attention, Gated DeltaNet sẽ chạy đường torch (chậm)")
    # --hub-kernels: kernel dựng sẵn từ HF Hub qua gói `kernels` (causal_conv1d_fn ← kernels-community/mamba-ssm)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 **({"use_kernels": True} if a.hub_kernels else {})).to(device)
    targets = lora_targets(model)
    print("LoRA target:", targets)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(train) * a.epochs / a.accum)                       # số bước job này chạy tới
    horizon = math.ceil(len(train) * (a.sched_epochs or a.epochs) / a.accum)  # số bước của lịch LR
    if horizon < total:
        raise SystemExit(f"--sched-epochs ({a.sched_epochs}) < --epochs ({a.epochs})")
    warm = max(1, int(horizon * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, horizon - warm)))))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    data_id = {"train": os.path.abspath(a.train), "train_bytes": os.path.getsize(a.train), "n_train": len(train)}
    json.dump(vars(a) | data_id | {"lora_targets": targets, "n_val": len(val), "steps": total, "sched_steps": horizon},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)

    step, micro, t0, run = 0, 0, time.time(), {"choice": [], "lm": []}
    ntok = 0
    order = []
    ep = 0

    def save_ckpt(name):
        """Adapter (dùng được ngay cho modal_merge.py) + train_state.pt (đủ để --resume tiếp đúng chỗ)."""
        d = os.path.join(a.out, name)
        model.save_pretrained(d)
        torch.save({"step": step, "ep": ep, "order": list(order), "ntok": ntok, "elapsed": time.time() - t0,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "data": data_id,
                    "args": {k: v for k, v in vars(a).items() if k not in ("resume",)},
                    "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all()}},
                   os.path.join(d, "train_state.pt"))
        print("CKPT", d, flush=True)

    if a.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(os.path.join(a.resume, "train_state.pt"), map_location="cpu", weights_only=False)
        if st["data"]["n_train"] != len(train) or st["data"]["train_bytes"] != data_id["train_bytes"]:
            raise SystemExit(f"dữ liệu train khác checkpoint: {st['data']} vs {data_id}")
        for k in ("accum", "r", "alpha", "max_len", "max_train"):
            if st["args"].get(k) != getattr(a, k):
                raise SystemExit(f"--{k} khác checkpoint: {st['args'].get(k)} vs {getattr(a, k)}")
        res = set_peft_model_state_dict(model, load_file(os.path.join(a.resume, "adapter_model.safetensors")))
        if getattr(res, "unexpected_keys", None):
            raise SystemExit(f"adapter không khớp: {res.unexpected_keys[:5]}")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])      # last_epoch = số bước đã chạy; lịch LR lấy theo horizon ở trên
        step, ep, order, ntok = st["step"], st["ep"], list(st["order"]), st["ntok"]
        micro = step * a.accum                  # checkpoint luôn ở ranh giới bước tối ưu (grad đã zero)
        t0 = time.time() - st["elapsed"]
        random.setstate(st["rng"]["py"])
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        print(f"RESUME {a.resume}: bước {step}/{total} (lịch LR {horizon}), epoch {ep}, còn {len(order)} ví dụ "
              f"trong epoch, lr {sched.get_last_lr()[0]:.3g}", flush=True)
        if step >= total:
            print("checkpoint đã ở/qua --epochs, không còn gì để chạy (tăng --epochs để train tiếp)")

    while step < total:
        if not order:
            order = list(range(len(train)))
            random.Random(a.seed + ep).shuffle(order)
            ep += 1
        b = train[order.pop()]
        ntok += len(b[0])
        l, _ = losses(model, b, a.ls, a.brier, device)
        loss = a.w_choice * l.get("choice", 0) + a.w_lm * l.get("lm", 0)
        (loss / a.accum).backward()
        for k, v in l.items():
            run[k].append(v.item())
        micro += 1
        if micro % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 or step == total:
                el = time.time() - t0
                rec = {"step": step, "of": total, "lr": sched.get_last_lr()[0], "sec": round(el),
                       "tok_per_s": round(ntok / el), "tokens": ntok,
                       "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                       "eta_min": round(el / step * (total - step) / 60, 1),
                       **{f"train_{k}": round(sum(v) / max(1, len(v)), 4) for k, v in run.items()}}
                run = {"choice": [], "lm": []}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            per_epoch = max(1, len(train) // a.accum)
            is_eval = (a.eval_every and step % a.eval_every == 0) or (not a.eval_every and step % per_epoch == 0) \
                or step == total
            if val and is_eval:
                rec = {"step": step, "epoch": round(step / per_epoch, 2), **evaluate(model, val, a, device)}
                print("EVAL", rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if is_eval or not order:            # sau eval / hết một epoch: checkpoint đầy đủ giữ lại
                save_ckpt(f"step{step}")
            if a.save_every and step % a.save_every == 0 and step < total:
                save_ckpt("last")               # ghi đè: điểm resume mới nhất (cuối job đã có final)
    save_ckpt("final")
    print("xong", round((time.time() - t0) / 60, 1), "phút")


if __name__ == "__main__":
    main()
