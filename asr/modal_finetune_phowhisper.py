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

Prompt v2 (PLAN_ASR_bias_v2.md, --prompt-v2, manifest_v2.json + lexicon_v2.json, logic trong bias_prompt.py):
    V1: --prompt-v2                          (chỉ nguồn có lúc test: hội thoại + hồ sơ + tên trong history)
    V2: --prompt-v2 --label-ent-prob 0.5     (+ tên trong label kèm tên gây nhiễu, chỉ lúc train)
    V3: --prompt-v2 --label-ent-prob 0.5 --all-distract-prob 0.3
    f.spawn(label_smoothing=0.1, ctc_weight=0.3, use_prompt=True, prompt_v2=True, label_ent_prob=0.5)
Logic train nằm ở asr_train.py (không phụ thuộc Modal); file này chỉ bọc nó thành Modal function.
Không có Modal (myserver / Kaggle): ZAIC_DATA=<dir có manifest + train/audio> ZAIC_OUT=<dir ra> \
    python asr_train.py --label-smoothing 0.1 --ctc-weight 0.3 --use-prompt --prompt-v2 [--bs 4 --accum 8]
Kaggle: xem kaggle/README.md.
"""

import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-phowhisper-finetune")
data_vol = modal.Volume.from_name("zaic-data")
out_vol = modal.Volume.from_name("zaic-out")
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1",
                 "jiwer", "librosa", "soundfile", "numpy<2.3")
    .env({"ZAIC_DATA": "/data", "ZAIC_OUT": "/out"})
    .add_local_file(os.path.join(HERE, "bias_prompt.py"), "/root/bias_prompt.py")
    .add_local_file(os.path.join(HERE, "asr_train.py"), "/root/asr_train.py")
)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": data_vol, "/out": out_vol}, timeout=4 * 3600)
def train(**kw):
    """Các tham số: xem asr_train.train (epochs, lr, label_smoothing, ctc_weight, use_prompt, prompt_v2, ...)."""
    import asr_train
    return asr_train.train(**kw, commit=out_vol.commit)


@app.local_entrypoint()
def main(epochs: int = 6, lr: float = 3e-4, label_smoothing: float = 0.0, full_data: bool = False,
         entity_weight: float = 1.0, ctc_weight: float = 0.0, resume: bool = True,
         use_prompt: bool = False, prompt_drop: float = 0.4, prompt_max_tokens: int = 180,
         prompt_v2: bool = False, label_ent_prob: float = 0.0, all_distract_prob: float = 0.0):
    print(train.remote(epochs=epochs, lr=lr, label_smoothing=label_smoothing, full_data=full_data,
                       entity_weight=entity_weight, ctc_weight=ctc_weight, resume=resume,
                       use_prompt=use_prompt, prompt_drop=prompt_drop, prompt_max_tokens=prompt_max_tokens,
                       prompt_v2=prompt_v2, label_ent_prob=label_ent_prob, all_distract_prob=all_distract_prob))
