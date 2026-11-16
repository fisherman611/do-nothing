# ZAC 2026 LifeMate — mô tả giải pháp

## 1. Luồng suy luận

```
audio các lượt user (history + câu cuối)
   │
   ▼
ASR: PhoWhisper-large + LoRA (đã gộp)        prompt = transcript các lượt trước + lời assistant, beam 5, fp16
   │  transcript
   ▼
Bộ lọc app: BM25 top-4 ∪ embedding top-5 ∪ luật (app đã dùng trong history, đóng bao x-ref)
   │  tập app ứng viên → catalog hai tầng (app ứng viên đầy đủ schema, app còn lại rút gọn)
   ▼
FC: Qwen3.5-4B (chỉ text) + LoRA (đã gộp), vLLM cục bộ, greedy (temperature 0, seed 0)
   │  vòng agent Clef-token, mỗi bước:
   │    Q1 chọn tool (1 token nhãn, allowed_token_ids) → Q2 điền từng tham số (1 token / câu hỏi)
   │    → sinh giá trị tham số tự do → env.validate → env.call → kết quả đưa lại vào prompt
   │  dừng khi model chọn DỪNG / sys.ask_missing_info / lặp call 3 lần / hết 30 s
   ▼
ToolEnv (zaic_tools của split): get_tool_calls(), get_state_hash()
```

- Mỗi mẫu đúng **một** `ToolEnv`; chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`. Không đọc `<split>_environment/`.
- N1 (`retry_empty`): tool đọc trả danh sách rỗng → gọi lại một lần, bỏ tham số chữ tuỳ chọn (thường là từ ASR nghe sai).
- Ngân sách 30 s/mẫu tính cả ASR; hết giờ thì dừng và nộp các call đã chạy.
- Mẫu chạy tuần tự, seed cố định (42 cho Python/NumPy/PyTorch, 0 cho vLLM), giải mã greedy.

## 2. Model

| Thành phần | Repo HF @ commit | Model gốc | Tham số |
|---|---|---|---|
| ASR | `fisherman611/pw-tmp-e46f` @ `85081b30df5a9de0b792aa82a7af7876da74a253` | `vinai/PhoWhisper-large` | 1.54B |
| FC (LLM) | `fisherman611/q-v5-6c1e` @ `c2db4cff5f4cb7b3e2013c20ecd85a91224ca516` | `Qwen/Qwen3.5-4B`, bỏ phần vision (chỉ text) | 4.206B, context 16,384 token |
| Embedding bộ lọc app | `fisherman611/vibi-appfilter-v2-b` @ `386ba3baf57876a556d06bb5a0eb3a05fa7944ec` | `bkai-foundation-models/vietnamese-bi-encoder` | 0.135B |
| Tokenizer FC | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (chỉ file tokenizer) | – | – |
| **Tổng** | | | **5.88B** (< 6B; LLM < 5B) |

BM25 không có tham số học. Cấu hình bộ lọc: `config/filter/pack.json` (k_bm25 = 4, k_emb = 5), từ khoá app `config/filter/keywords.json`.
Định dạng prompt FC (phải giống lúc train): `config/fmt.json` (`style v1, q2 all, q1_desc short, sys compact, rules short`).

## 3. Huấn luyện (chi tiết và URL: `training_code/README.md`)

- **ASR**: PhoWhisper-large fine-tune LoRA rồi gộp; train có prompt (transcript các lượt trước) đúng như lúc suy luận.
- **FC**: SFT LoRA (r 32, alpha 64, dropout 0.05, 12 module gồm attention, MLP và các lớp Gated DeltaNet) trên định dạng Clef-token:
  loss chọn nhãn (label smoothing 0.1 + Brier) cộng loss sinh tham số tự do; max_len 12,288.
  1. 3 epoch trên `fmtA_v4` (21,091 ví dụ), LR 1e-4, cosine.
  2. +1 epoch trên `fmtA_v5` (20,951 ví dụ), khởi tạo từ adapter bước 1, LR 2e-5, seed 1.
  3. +1 epoch trên `fmtA_v5`, khởi tạo từ adapter bước 2, LR 1e-5, seed 2; chọn checkpoint step 400, gộp vào model gốc.
- **Bộ lọc app**: fine-tune `vietnamese-bi-encoder` 2 epoch (cấu hình B) để khớp câu yêu cầu với mô tả app.

Dữ liệu tổng hợp dùng khi train (chỉ lúc phát triển, không dùng lúc suy luận): app và giao diện tool đổi tên (alias / iface) để model
tổng quát sang app lạ; nhiễu ASR (bảng lỗi nghe thật của ASR, biến thể đọc theo âm) chèn vào câu yêu cầu; audio TTS cho ASR.
Không gán nhãn public_test / private_test để train hay chỉnh giải pháp.

## 4. Chạy

```bash
bash build.sh zac2026-<team>:v1                         # kéo model (cần mạng), docker build
docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result zac2026-<team>:v1 /bin/bash /code/predict.sh
docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result zac2026-<team>:v1 /bin/bash /code/start_jupyter.sh
```

| File | Việc |
|---|---|
| `predict.sh` / `predict.py` | chạy toàn split → `/result/submission.jsonl` (id, tool_calls, state_hash, time) |
| `predict_notebook.ipynb` | cùng luồng, từng bước (seed → nạp model → đọc test → dự đoán) → `/result/jupyter_submission.jsonl` |
| `src/pipeline.py` | luồng dùng chung: ASR, khởi động vLLM, bộ lọc, vòng agent |
| `src/agent/` | vòng agent Clef-token (`clef_agent.py`, `clef/fmt.py`, `clef/core.py`) và bộ lọc app (`clef/app_filter/`) |
| `requirements.txt` | môi trường `/opt/app` (Python 3.10) |
| `requirements_fc.txt` | môi trường `/opt/fc` (Python 3.12, vLLM 0.31.0) chỉ để chạy server FC |
