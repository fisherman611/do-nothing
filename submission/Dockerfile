# ZAC 2026 LifeMate — image nộp bài. Chạy offline: mọi model nằm trong /code/saved_models.
#   bash build.sh                          # kéo model (ghim commit) vào saved_models/, rồi docker build
#   docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result zac2026-<team>:v1 /bin/bash /code/predict.sh
FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Ho_Chi_Minh PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0 vim python3 python3-pip tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir uv

# FC server: vLLM 0.31.0 (cu129) — đúng bộ thư viện đã dùng để đo bản FC này
COPY requirements_fc.txt /tmp/requirements_fc.txt
RUN uv venv --python 3.12 /opt/fc \
    && VIRTUAL_ENV=/opt/fc uv pip install --no-cache --no-deps -r /tmp/requirements_fc.txt \
         --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
    && /opt/fc/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"

# App: predict.py / notebook / ASR / bộ lọc / agent
COPY requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.10 /opt/app \
    && VIRTUAL_ENV=/opt/app uv pip install --no-cache -r /tmp/requirements.txt \
    && /opt/app/bin/python -c "from pyvi import ViTokenizer; assert ViTokenizer.tokenize('bật đèn phòng khách')" \
    && /opt/app/bin/python -m ipykernel install --name python3 --display-name "Python 3 (ipykernel)" --prefix /usr/local \
    && ln -sf /opt/app/bin/jupyter /usr/local/bin/jupyter

COPY . /code
RUN chmod +x /code/predict.sh /code/start_jupyter.sh \
    && test -f /code/saved_models/fc/model.safetensors \
    && test -f /code/saved_models/asr/config.json \
    && test -f /code/saved_models/filter/model/model.safetensors \
    && test -f /code/saved_models/tokenizer/tokenizer.json
WORKDIR /code
CMD ["/bin/bash", "/code/predict.sh"]
