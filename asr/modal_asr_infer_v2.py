"""A5: infer ASR v2 trên Modal (app riêng, không đụng app train đang chạy). Xem asr_infer_v2.py.
    modal deploy modal/modal_asr_infer_v2.py
    f = modal.Function.from_name('lifemate-asr-infer-v2', 'infer'); f.remote(tag=..., which='val')"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-asr-infer-v2")
data_vol = modal.Volume.from_name("zaic-data")
out_vol = modal.Volume.from_name("zaic-out")
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1",
                 "jiwer", "librosa", "soundfile", "scipy", "numpy<2.3")
    .env({"ZAIC_DATA": "/data", "ZAIC_OUT": "/out"})
    .add_local_file(os.path.join(HERE, "bias_prompt.py"), "/root/bias_prompt.py")
    .add_local_file(os.path.join(HERE, "asr_train.py"), "/root/asr_train.py")
    .add_local_file(os.path.join(HERE, "asr_infer_v2.py"), "/root/asr_infer_v2.py")
)


@app.function(image=image, gpu=os.environ.get("ASR_INFER_GPU", "L40S"), cpu=4, volumes={"/data": data_vol, "/out": out_vol}, timeout=2 * 3600)
def infer(**kw):
    import asr_infer_v2
    return asr_infer_v2.run(**kw)
