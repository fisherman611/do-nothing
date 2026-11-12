# LifeMate — giải pháp ZAC 2026 (thư mục /code)

Trợ lý giọng nói tiếng Việt: nghe yêu cầu, gọi tool của các app mô phỏng qua `zaic_tools`, trả `tool_calls` + `state_hash`.

## Chạy

```bash
# kết quả: /result/submission.jsonl  (mỗi dòng: id, tool_calls, state_hash, time[ms])
docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/predict.sh
# đo thời gian bằng notebook: mở http://localhost:9777 (mật khẩu zac2026) → predict_notebook.ipynb → Run All
# kết quả: /result/jupyter_submission.jsonl
docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result <image> /bin/bash /code/start_jupyter.sh
```

Không cần mạng. `/data/<split>/` có cấu trúc như gói public test; split được tự nhận theo `<split>_tools_catalog.json`.
Tham số mặc định nằm trong `predict.py` (`_DEF`). `python3 /code/selfcheck.py` kiểm image mà không cần GPU.

## Mô hình

| Thành phần | Mô hình | Tham số | Ghi chú |
|---|---|---|---|
| ASR | PhoWhisper-large fine-tune (`saved_models/asr`) | 1.54B | beam 5, fp16; prompt = transcript các lượt trước + lời assistant |
| LLM gọi tool | Qwen3.5-4B bản chỉ-text + LoRA đã gộp (`saved_models/fc`) | 4.206B (< 5B) | context tối đa 16 384 token; vLLM, temperature 0, seed 0, không thinking |
| Bộ lọc app | `vietnamese-bi-encoder` fine-tune + BM25 (`saved_models/pack`) | 0.135B | chạy CPU |
| **Tổng** | | **5.88B (< 6B)** | |

## Suy luận (lần lượt từng mẫu)

1. Tạo đúng một `ToolEnv` cho mẫu (dùng `zaic_tools` của split trong `/data`, không dùng bản đóng trong image).
2. **ASR** mọi lượt user của mẫu theo thứ tự (history rồi câu cuối); transcript lượt trước làm prompt cho lượt sau.
3. **Bộ lọc app**: BM25 top-k ∪ embedding top-k ∪ luật (app đã gọi trong history, app liên quan qua x-ref) → tập app "tầng 1".
   Catalog trong prompt có hai tầng: app tầng 1 ghi đủ schema, các app đã cài còn lại chỉ ghi tên tool (vẫn gọi được).
4. **Vòng agent Clef-token** (tối đa 8 bước): mỗi bước LLM trả lời bằng **một token nhãn** trên tập nhãn hợp lệ
   - Q1: bước tiếp theo — một tool, `sys.ask_missing_info`, hoặc DỪNG;
   - Q2: từng tham số rời rạc — có/không điền tham số tuỳ chọn, chọn giá trị enum, chọn id từ kết quả tool trước;
   - tham số chữ tự do thì sinh tới `</parameter>`.
   Sau đó `env.validate` (sai schema thì không gọi, đưa lỗi vào prompt) → `env.call` → kết quả đưa vào bước sau.
   Tool đọc trả danh sách rỗng thì tìm lại một lần, bỏ tham số chữ tuỳ chọn.
5. Dừng khi: chọn DỪNG, đã hỏi lại, một call lặp 3 lần, validate sai 3 lần, đủ 8 bước, hoặc gần hết 30 giây (tính cả ASR).
6. Ghi `env.get_tool_calls()`, `env.get_state_hash()` và thời gian của mẫu.

Chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`; không đọc `<split>_environment/`.

## Tái lập

- Seed cố định (0) cho Python/NumPy/PyTorch, ASR (beam search) và vLLM (`--seed 0`, `temperature 0`).
- `predict.sh` và notebook dùng chung hàm `predict.infer`, mỗi lúc chỉ một request tới vLLM.
- Mã train, dữ liệu và URL tải: xem `training_code/README.md`.

## Cấu trúc

```
predict.sh  predict.py  start_jupyter.sh  predict_notebook.ipynb  selfcheck.py  requirements.txt  README.md
agent/            vòng agent Clef-token (clef_agent.py), định dạng prompt (clef/fmt.py), bộ lọc app (clef/app_filter/)
docker/           asr_stage.py (ASR; chế độ --serve phục vụ từng mẫu), run.sh (lối chạy theo lô, chỉ để đo lúc phát triển)
saved_models/     asr/  fc/  pack/  fmt.json
training_code/    mã train + README
```
