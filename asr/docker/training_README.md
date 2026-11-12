# training_code — tái lập huấn luyện

Thư mục này chứa mã train của ba thành phần và công cụ dựng dữ liệu. BTC chạy lại có mạng; dữ liệu và model tải theo URL dưới đây.
Các mục đánh dấu **[ĐỘI ĐIỀN]** phải được hoàn tất trước hạn nộp Docker — chúng chưa có trong repo tại thời điểm soạn file.

> Dữ liệu và model trên HuggingFace **không được thay đổi sau hạn nộp Docker**. Ghi commit hash cho từng repo.

## 1. Nguồn tải

| Thứ | Nơi lưu | Commit | Ghi chú |
|---|---|---|---|
| ASR dùng lúc suy luận | `fisherman611/pw-tmp-e46f` | `85081b30df5a9de0b792aa82a7af7876da74a253` | PhoWhisper-large fine-tune, đã gộp LoRA |
| LLM gọi tool dùng lúc suy luận | `fisherman611/q-f48e` | `0cd1ed0dd6aaaf3c31cc3dba1b2468ab29a6b071` | Qwen3.5-4B chỉ-text + LoRA `fmtA_48e` đã gộp |
| LoRA của LLM | `fisherman611/q-f48e-lora` (thư mục `final`) | **[ĐỘI ĐIỀN]** | |
| Bộ lọc app (`pack_b`) | **[ĐỘI ĐIỀN: URL]** | **[ĐỘI ĐIỀN]** | `pack.json`, `keywords.json`, `catalog.json`, `model/` |
| Dữ liệu sinh thêm (synth, TTS, nhiễu ASR) | `fisherman611/zaic-data-5eb2dec0` | **[ĐỘI ĐIỀN]** | |
| Base model (không nằm trong Docker) | `vinai/PhoWhisper-large`, `Qwen/Qwen3.5-4B` (`851bf6e…`), `bkai-foundation-models/vietnamese-bi-encoder` | | tải lúc train |

Dữ liệu gốc: gói `train` của BTC (1 545 hội thoại). Không dùng nhãn hay transcript của `public_test` / `private_test` để train
hoặc chọn model. Các công cụ gán nhãn transcript `public_test` (chỉ để đánh giá ASR) không nằm trong thư mục này.

## 2. Các bước (seed cố định trong từng script)

### 2.1 ASR — PhoWhisper-large + LoRA
- Mã: `asr/asr_train.py`, `asr/bias_prompt.py`, `asr/finetune_v1/`, `asr/kaggle/` (chạy trên Kaggle T4), `asr/modal_*.py` (chạy trên Modal).
- Dữ liệu: audio train của BTC + audio TTS sinh thêm (xem `plans/PLAN_tts_asr_from_fc.md`, `plans/PLAN_PhoWhisper_finetune.md`).
- Lệnh, siêu tham số và số epoch của bản `e46f`: **[ĐỘI ĐIỀN]**.
- Đánh giá: 405 đoạn giữ riêng của train — `python -m metrics asr-detail …` (xem `metrics/README.md`).

### 2.2 LLM gọi tool — Qwen3.5-4B chỉ-text + LoRA (định dạng Clef-token)
1. Cắt Qwen3.5-4B còn phần text: `asr/agent/strip_qwen35.py` (bỏ `model.visual.*` 0.33B và `mtp.*` 0.12B → 4.206B).
2. Chia tập: `asr/agent/clef/splits.py` → `splits.json`.
3. Dựng dữ liệu theo bước: `asr/agent/clef/build_data.py` (+ `make_subset.py`); cờ dùng cho `fmtA_48e_20k`: **[ĐỘI ĐIỀN]**.
   File `train.jsonl.meta.json` của bộ này chính là `saved_models/fmt.json` trong image (định dạng prompt phải giống lúc train).
4. Train: `asr/agent/clef/train.py` (LoRA; loss chọn nhãn + CE cho tham số tự do), chạy qua `modal_train.py` / `lf_train.py`.
   Lệnh và siêu tham số của bản `fmtA_48e` (1 epoch): **[ĐỘI ĐIỀN]**.
5. Gộp adapter: `asr/agent/clef/modal_merge.py` hoặc `merge_local.py`.
6. Đánh giá: `asr/agent/eval_val.py` + `python -m metrics score …`.

### 2.3 Bộ lọc app — bi-encoder + BM25
- Mã: `asr/agent/clef/app_filter/train_v2.py` (fine-tune `vietnamese-bi-encoder`, cần `pyvi`), `filter.py`, `pack.py build` (đóng gói).
- Lệnh train và đóng gói của `pack_b`: **[ĐỘI ĐIỀN]**. Báo cáo: `plans/REPORT_app_filter_v2.md`.

### 2.4 Dữ liệu tổng hợp
- Kế hoạch và cách sinh: `plans/PLAN_synth_fc_data_gemini.md`, `plans/PLAN_phonetic_augment_gemini.md`, `plans/PLAN_data_augment_v3.md`.
- Dịch vụ ngoài (LLM, TTS) chỉ dùng để sinh dữ liệu lúc phát triển; suy luận hoàn toàn offline.
- Mã sinh (`synth/…`) và bảng kê file dữ liệu + checksum: **[ĐỘI ĐIỀN nếu thư mục `synth/` chưa được chép vào đây]**.

## 3. Kiểm lại kết quả suy luận

Sau khi có ba model, đặt vào `saved_models/{asr,fc,pack}` + `fmt.json` rồi chạy `bash /code/predict.sh`.
