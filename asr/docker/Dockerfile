# Image nộp bài theo "Hướng dẫn dùng docker để nộp bài ZAC 2026 - LifeMate" của BTC:
#   audio → ASR (PhoWhisper-large) → bộ lọc app → FC (Qwen3.5-4B + Clef-token, vLLM) ⇄ zaic_tools → /result/submission.jsonl
# Dựng bằng docker/build.sh (cần mạng lúc BUILD để kéo model; lúc CHẠY không cần mạng). BTC chạy:
#   docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/predict.sh
#   docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/start_jupyter.sh
# Cấu trúc /code: predict.sh predict.py start_jupyter.sh predict_notebook.ipynb requirements.txt README.md
#                 saved_models/{asr,fc,pack,fmt.json} training_code/ agent/ docker/asr_stage.py selfcheck.py
# Base image: CUDA 13.0.2 (BTC yêu cầu CUDA không cao hơn 13.2, driver 595.58.03).
FROM vllm/vllm-openai:v0.31.0

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_USE_FLASHINFER_SAMPLER=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# FC + bộ lọc + Jupyter chạy bằng Python của image (vLLM, transformers mới).
# pyvi: bộ lọc app (vietnamese-bi-encoder, gốc PhoBERT) được train với câu đã tách từ bằng pyvi. Thiếu pyvi thì
# filter.Embedder lặng lẽ bỏ tách từ VÀ bỏ luôn giới hạn 256 token → điểm embedding lệch lúc đo, tài liệu app dài có thể IndexError.
# jupyterlab: BTC bắt buộc, dùng predict_notebook.ipynb để đo thời gian từng mẫu.
RUN pip install --no-cache-dir sentence-transformers soundfile huggingface_hub pyvi==0.1.1 jupyterlab \
    && python3 -c "from pyvi import ViTokenizer as V; s = V.tokenize('Trường đại học bách khoa hà nội'); assert '_' in s, s; print('pyvi ok:', s)" \
    && jupyter lab --version

# ASR chạy trong venv riêng: PhoWhisper + prompt_ids cần transformers 4.57.1 (như lúc train/đo)
RUN python3 -m venv /opt/asr \
    && /opt/asr/bin/pip install --no-cache-dir torch==2.8.0 transformers==4.57.1 accelerate==1.10.1 soundfile "numpy<2.3"

# ---- model đã fine-tune (kéo lúc build, ghim commit). Base model KHÔNG đưa vào image; Qwen/Qwen3.5-4B chỉ lấy file tokenizer ----
ARG ASR_REPO=fisherman611/pw-tmp-e46f
ARG ASR_REV=85081b30df5a9de0b792aa82a7af7876da74a253
ARG FC_REPO=fisherman611/q-f48e
ARG FC_REV=0cd1ed0dd6aaaf3c31cc3dba1b2468ab29a6b071
ARG TOK_REPO=Qwen/Qwen3.5-4B
ARG TOK_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RUN python3 -c "import os; from huggingface_hub import snapshot_download as d; \
d('${ASR_REPO}', revision='${ASR_REV}', local_dir='/code/saved_models/asr'); \
d('${FC_REPO}', revision='${FC_REV}', local_dir='/code/saved_models/fc'); \
d('${TOK_REPO}', revision='${TOK_REV}', allow_patterns=['*.json','*.txt','tokenizer*','vocab*','merges*'])"

# ---- code + bộ lọc app + định dạng prompt (docker/ctx.sh dựng thư mục ctx/) ----
COPY ctx/code /code
COPY ctx/pack /code/saved_models/pack
COPY ctx/fmt.json /code/saved_models/fmt.json

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /code
# Tự kiểm ngay lúc build (không cần GPU): đủ file, pyvi, tokenizer nạp khi tắt mạng, bộ lọc app chạy được, jupyter có mặt.
RUN python3 /code/selfcheck.py

# Base image đặt ENTRYPOINT ["vllm","serve"]; BTC truyền lệnh đầy đủ (/bin/bash /code/predict.sh) nên phải bỏ entrypoint.
ENTRYPOINT []
CMD ["/bin/bash", "/code/predict.sh"]
