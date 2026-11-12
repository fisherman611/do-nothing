"""B4 (PLAN_72h_v3): train ASR v3 TỪ PhoWhisper-large GỐC + LoRA, prompt v1, data v3 (thật ×2 + TTS v2 xử lý lại + TTS v3
giọng chuẩn + phương ngữ ElevenLabs), augment_v3 (tốc độ + nhiễu thật), Modal H100.

Volume `zaic-asr-v3-data` (dựng bằng `python modal/modal_asr_v3.py upload`):
  manifest.json (audio thật, split train/val = 405 held-out), tts_manifest_v3.json,
  train/audio/... (audio thật), tts_v2p/, tts_v3/, tts_v3el/ (TTS đã xử lý), noise_bank/ (+ eq_dashscope.json)
Kết quả: volume `zaic-asr-v3-out/<tag>` (adapter, model gộp, eval.json). Resume tự động nếu chạy lại cùng tham số.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy modal/modal_asr_v3.py
    python modal/modal_asr_v3.py smoke     # 24 hội thoại, 1 epoch: kiểm bộ nhớ/tốc độ
    python modal/modal_asr_v3.py run       # chạy thật (spawn, trả về call id)
"""
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ZAIC = os.path.dirname(HERE)
APP = "lifemate-asr-train-v3"
app = modal.App(APP)
data_vol = modal.Volume.from_name("zaic-asr-v3-data", create_if_missing=True)
out_vol = modal.Volume.from_name("zaic-asr-v3-out", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1",
                 "jiwer", "librosa", "soundfile", "scipy", "numpy<2.3")
    .env({"ZAIC_DATA": "/data", "ZAIC_OUT": "/out", "HF_HOME": "/out/hf_cache"})
    .add_local_file(os.path.join(HERE, "bias_prompt.py"), "/root/bias_prompt.py")
    .add_local_file(os.path.join(HERE, "asr_train.py"), "/root/asr_train.py")
)

V3_KW = dict(
    model_id="vinai/PhoWhisper-large", model_revision="b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb",
    epochs=6, patience=1, save_limit=2, lr=3e-4, label_smoothing=0.1, ctc_weight=0.3,
    use_prompt=True, prompt_drop=0.4, tts_manifest="tts_manifest_v3.json", real_repeat=2, aug_v3=0.3,
    eval_items=0, eval_lexicon=False, workers=8, time_budget_h=8.0, tag_suffix="v3base",   # eval đủ 405 câu/epoch: patience 1 cần số đo ít nhiễu
)


@app.function(image=image, gpu="H100", cpu=8, memory=49152, volumes={"/data": data_vol, "/out": out_vol},
              timeout=10 * 3600)
def train(**kw):
    import asr_train
    return asr_train.train(**kw, commit=out_vol.commit)


def upload():
    """Đẩy data lên volume (bỏ qua file đã có cùng tên)."""
    D = os.path.join(ZAIC, "data")
    with data_vol.batch_upload(force=True) as b:
        b.put_file(os.path.join(HERE, "manifest.json"), "/manifest.json")
        b.put_file(f"{D}/tts_v3/tts_manifest_v3.json", "/tts_manifest_v3.json")
        b.put_directory(f"{D}/train+public_test/train/audio", "/train/audio")
        b.put_directory(f"{D}/tts_v2/wav_v3proc", "/tts_v2p")
        b.put_directory(f"{D}/tts_v3/wav", "/tts_v3")
        b.put_directory(f"{D}/tts_v3/wav_el", "/tts_v3el")
        b.put_directory(f"{D}/noise_bank", "/noise_bank")
    print("upload xong")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "print"
    if cmd == "upload":
        upload()
    elif cmd in ("smoke", "run", "run_e46f", "run_b4c"):
        f = modal.Function.from_name(APP, "train")
        kw = dict(V3_KW)
        if cmd == "run_e46f":   # dự phòng B4b: train tiếp từ e46f (giữ phần nghe giọng thật), LR thấp, 1 epoch, thật ×3
            kw.update(model_id="fisherman611/pw-tmp-e46f", model_revision="85081b30df5a9de0b792aa82a7af7876da74a253",
                      epochs=1, lr=1e-4, real_repeat=3, time_budget_h=2.0, tag_suffix="v3e46f")
        if cmd == "run_b4c":    # B4c: khởi tạo từ epoch 4 (tốt nhất) của B4, lịch LR mới 1e-4 cosine 3 epoch, patience 2
            kw.update(init_adapter="/out/lora_ls0.1_ctc0.3_prompt_tts_v3base_ckpt/checkpoint-2044", epochs=3, lr=1e-4,
                      patience=2, save_limit=3, time_budget_h=2.5, tag_suffix="v3b4c")
        if cmd == "smoke":
            kw.update(max_items=24, epochs=1, eval_items=12, final_eval=False, time_budget_h=0.5, tag_suffix="v3smoke")
        print(cmd, f.spawn(**kw).object_id)
    else:
        import json
        print(json.dumps(V3_KW, indent=1))
