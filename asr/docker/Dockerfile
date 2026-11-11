# D1 (PLAN_72h_v3): image chạy OFFLINE toàn bộ luồng  audio → ASR (PhoWhisper) → bộ lọc app → FC (Qwen3.5-4B + Clef) → submission.jsonl
# Dựng bằng docker/build.sh (cần mạng lúc BUILD để kéo model; lúc CHẠY không cần mạng).
#   docker run --rm --gpus all -v <data>:/data:ro -v <out>:/result -e SPLIT=private_test lifemate:dev
# Giả định chưa được BTC xác nhận (tài liệu đóng gói Docker "sẽ có riêng"): /data có cấu trúc như data/train+public_test
# (<split>/<split>.jsonl, <split>/audio, <split>/zaic_tools, <split>_environment/...), kết quả ghi ở /result/submission.jsonl.
FROM vllm/vllm-openai:v0.31.0

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf HF_HUB_DISABLE_TELEMETRY=1 VLLM_USE_FLASHINFER_SAMPLER=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# FC + bộ lọc chạy bằng Python của image (vLLM, transformers mới)
RUN pip install --no-cache-dir sentence-transformers soundfile huggingface_hub

# ASR chạy trong venv riêng: PhoWhisper + prompt_ids cần transformers 4.57.1 (như lúc train/đo)
RUN python3 -m venv /opt/asr \
    && /opt/asr/bin/pip install --no-cache-dir torch==2.8.0 transformers==4.57.1 accelerate==1.10.1 soundfile "numpy<2.3"

# ---- model (kéo lúc build, ghim commit) ----
ARG ASR_REPO=fisherman611/pw-tmp-e46f
ARG ASR_REV=85081b30df5a9de0b792aa82a7af7876da74a253
ARG FC_REPO=fisherman611/q-f48e
ARG FC_REV=0cd1ed0dd6aaaf3c31cc3dba1b2468ab29a6b071
ARG TOK_REPO=Qwen/Qwen3.5-4B
ARG TOK_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RUN python3 -c "import os; from huggingface_hub import snapshot_download as d; \
d('${ASR_REPO}', revision='${ASR_REV}', local_dir='/models/asr'); \
d('${FC_REPO}', revision='${FC_REV}', local_dir='/models/fc'); \
d('${TOK_REPO}', revision='${TOK_REV}', allow_patterns=['*.json','*.txt','tokenizer*','vocab*','merges*'])"

# ---- code + bộ lọc app + định dạng prompt ----
COPY ctx/agent /app/agent
COPY ctx/docker /app/docker
COPY ctx/pack /models/pack
COPY ctx/fmt.json /models/fmt.json

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["bash", "/app/docker/run.sh"]
