# Docker nộp bài (theo hướng dẫn của BTC)

Căn cứ: "Hướng dẫn dùng docker để nộp bài ZAC 2026 - LifeMate" (PDF của BTC). Hạn nộp: 16/11 23:59, tối đa 2 lần.

## Lối chạy BTC chấm

```bash
docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/predict.sh          # → /result/submission.jsonl
docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/start_jupyter.sh
#   http://localhost:9777 , mật khẩu zac2026 → predict_notebook.ipynb → Run All → /result/jupyter_submission.jsonl
```

`predict.py` chạy **lần lượt từng mẫu**, model nạp một lần:
`asr_stage.py --serve` (PhoWhisper, venv `/opt/asr`, giữ trên GPU) + `vllm serve` (FC) → với mỗi mẫu: đồng hồ → một `ToolEnv` →
ASR mọi lượt user → `clef_agent.run_one` (bộ lọc app + FC + ToolEnv) → dòng `{id, tool_calls, state_hash, time}`.
Notebook gọi đúng hàm `predict.infer` đó nên hai file kết quả phải khớp (BTC chỉ dùng `time` của notebook khi điểm hai file bằng nhau).

| Yêu cầu của BTC | Ở đâu |
|---|---|
| `/code/predict.sh`, đọc `/data`, ghi `/result/submission.jsonl`, tự tạo `/result` | `predict.sh`, `predict.py` |
| Trường `time` (ms, từ trước khi đọc audio) | `predict.infer` |
| In thời gian nạp model và thời gian predict ra stdout | `load model: … ms`, `predict: … ms/mẫu` |
| `predict_notebook.ipynb`: seed → nạp model → đọc test → predict từng mẫu → `jupyter_submission.jsonl` | `predict_notebook.ipynb` |
| `start_jupyter.sh` cổng 9777, mật khẩu `zac2026`; có `jupyterlab` | `start_jupyter.sh`, Dockerfile |
| Tham số mặc định đặt trong code | `predict._DEF` |
| `zaic_tools` lấy từ `/data/<split>`; split tự nhận | `predict.detect_split`, `load_models` |
| Không có mạng lúc chạy | model + tokenizer nằm trong image, `HF_HUB_OFFLINE=1` |
| `training_code/` + README, `README.md` mô tả ý tưởng | `training_README.md`, `code_README.md` (ctx.sh chép vào `/code`) |
| Base image CUDA ≤ 13.2 | `vllm/vllm-openai:v0.31.0` = CUDA 13.0.2 |
| Không có entrypoint chặn lệnh của BTC | `ENTRYPOINT []` |
| Nộp `zac2026_<TeamName>.tar.gz` + MD5 | `save.sh` |

## Dựng

```bash
cd asr
PACK=<pack_b> bash docker/build.sh zac2026-lifemate:v1     # cần mạng; kéo model ghim commit; cuối cùng chạy selfcheck.py
bash docker/save.sh zac2026-lifemate:v1 <TeamName>         # → zac2026_<TeamName>.tar.gz + .md5
```

`selfcheck.py` chạy ngay lúc build (không cần GPU): đủ file, `pyvi`, tokenizer nạp khi tắt mạng, bộ lọc app chạy được, có Jupyter.
Chạy thử trên Modal bằng đúng lối của BTC: `modal run docker/modal_d1.py --limit 20`.

## Đã kiểm / chưa kiểm

**Đã kiểm trên máy không GPU, với ASR giả + vLLM giả** (vòng `--serve` thật của `asr_stage.py`, WAV thật, `ToolEnv` thật):
- `predict.sh` chạy đủ 634 mẫu public; `python -m zaic_tools --validate` của BTC: 634 dòng, 0 lỗi schema (có trường `time`).
- Notebook chạy Run All (nbconvert): 5 ô không lỗi; `jupyter_submission.jsonl` khớp `submission.jsonl` 634/634 (id, tool_calls, state_hash).
- ASR treo quá `ASR_TIMEOUT` hoặc tiến trình ASR chết: mẫu vẫn có dòng kết quả (FC chạy với transcript rỗng), mẫu sau ASR tự bật lại.
- vLLM không trả lời: mỗi mẫu vẫn có dòng kết quả sau ~5 s (`REQ_RETRIES=2`), không treo.

**Chưa kiểm (cần GPU + `pack_b` + model thật):**
- Chưa build image thật và chưa chạy với model thật. Lệnh `pip install` trong base image vLLM (Python do `uv` quản lý) chưa được xác nhận.
- **Bộ nhớ GPU khi ASR và vLLM cùng nạp** (trước đây ASR thoát rồi mới bật vLLM). Mặc định `GPU_UTIL=0.70` cho vLLM, phần còn lại cho PhoWhisper; phải đo trên GPU 24 GB và chỉnh trong `predict._DEF`.
- Thời gian mỗi mẫu khi chạy một luồng trên GPU chấm (p50/p95), và tổng thời gian 634 mẫu (ước 2–3 giờ nếu ~15 s/mẫu).
- Điểm của lối chạy tuần tự so với bản đã nộp 62.76 (chạy nhiều luồng): cần một lượt nộp public để xác nhận không lệch.
- Tài liệu BTC ghi GPU RTX 5090 32 GB ở phần chữ nhưng ảnh `nvidia-smi` và luật thi ghi A30 24 GB → thiết kế cho 24 GB, nên hỏi BTC.
- `training_code/README.md` còn các mục **[ĐỘI ĐIỀN]** (lệnh train, URL + commit của `pack_b` và dữ liệu).

## Biến môi trường (chỉ để thử; BTC chạy không truyền gì)

`DATA` `SPLIT` `OUT` `WORK` `LIMIT` `GPU_UTIL` `TIME_BUDGET` `ASR_TIMEOUT` `REQ_RETRIES` `REQ_TIMEOUT` `GPU_LOG` `LLM_URL` — xem `predict._DEF`.

## Lối chạy theo lô (chỉ để đo nhanh lúc phát triển)

`run.sh`: ASR hết mọi mẫu → thoát → vLLM → `clef_agent.py` nhiều luồng → `submit.py`. Nhanh hơn nhưng không có trường `time`
đúng nghĩa và kết quả có thể lệch vài mẫu giữa các lần chạy; **không phải lối BTC chấm**.
`docker run … <image> /bin/bash /code/docker/run.sh` với `-e SPLIT=public_test`.

- ASR v3 (prompt thiên vị từ khoá `bias_prompt`): `asr_stage.py` mới có chế độ v1; chế độ v21 cần nối `asr_srv/asr_infer_v2.run`.
- Cờ N8/I5: đã đo, không tăng điểm, mặc định tắt.
