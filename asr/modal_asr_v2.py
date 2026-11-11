"""A4 (PLAN_48h): train ASR v2 = PhoWhisper-medium + LoRA, prompt V2.1, CTC + inter-CTC + KWS, TTS + tăng cường audio, trên Modal H100.

Dữ liệu trên volume `zaic-data`: manifest_v2.json, lexicon_v2.json, tts_manifest.json, prev_noise_table.json
(synth/tts_text/make_asr_data.py), train/audio/... (audio thật), tts/wav/... (audio TTS). Kết quả: volume `zaic-out/<tag>`.

    modal deploy modal/modal_asr_v2.py
    python -c "import modal; f=modal.Function.from_name('lifemate-asr-train-v2','train'); print(f.spawn(**KW).object_id)"
KW chuẩn: xem V2_KW bên dưới (in ra bằng `python modal/modal_asr_v2.py`).
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-asr-train-v2")
data_vol = modal.Volume.from_name("zaic-data", create_if_missing=True)
out_vol = modal.Volume.from_name("zaic-out", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1",
                 "jiwer", "librosa", "soundfile", "scipy", "numpy<2.3")
    .env({"ZAIC_DATA": "/data", "ZAIC_OUT": "/out", "HF_HOME": "/out/hf_cache"})
    .add_local_file(os.path.join(HERE, "bias_prompt.py"), "/root/bias_prompt.py")
    .add_local_file(os.path.join(HERE, "asr_train.py"), "/root/asr_train.py")
)

V2_KW = dict(
    model_id="vinai/PhoWhisper-medium", model_revision="55a7e3eb6c906de891f8f06a107754427dd3be79",
    epochs=10, patience=2, save_limit=3, lr=3e-4, label_smoothing=0.1, ctc_weight=0.3, inter_ctc_layer=12, kws_weight=0.1,
    use_prompt=True, prompt_v2=True, label_ent_prob=0.5, prompt_drop=0.4, prev_noise=0.7,
    tts_manifest="tts_manifest.json", audio_aug=0.5, eval_lexicon=False, bs=16, accum=2, workers=8, tag_suffix="v2",
)


@app.function(image=image, gpu=os.environ.get("ASR_GPU", "H100"), cpu=8, memory=32768,
              volumes={"/data": data_vol, "/out": out_vol}, timeout=6 * 3600)
def train(**kw):
    import asr_train
    return asr_train.train(**kw, commit=out_vol.commit)


if __name__ == "__main__":
    import json
    print(json.dumps(V2_KW, indent=1))
