# Plan fine-tune vinai/PhoWhisper-large cho ZAC 2026 – LifeMate (chạy trên Modal)

Cập nhật: 2026-11-07. Nơi chạy: **Modal** (workspace `vyvy372004`). Server `myserver` không dùng nữa vì đĩa đầy (`/mnt` còn ~6G) và 2 GPU đang bị job khác chiếm.
Code: `zaic/modal/` (`make_manifest.py`, `app.py`).

## 0. Bối cảnh & mục tiêu

- Bài toán: trợ lý giọng nói. Audio → ASR → agent gọi tool. Điểm chấm là **tool_calls + state_hash**, không phải WER.
- ASR sai tên người, số tiền, giờ, số điện thoại thì tool call sai theo.
- Mục tiêu: fine-tune PhoWhisper-large trên data train của BTC để giảm lỗi so với zero-shot, rồi đo ảnh hưởng lên tool-call.
- Vòng private chạy bằng **Docker** (README mục 4): model cần suy luận gọn, ổn định.

## 1. Dữ liệu

| Mục | Giá trị |
|---|---|
| Mẫu train | 1545 (`train/train.jsonl`) |
| Audio lượt cuối (có `audio_text`) | 1545 file, ~3.1h, TB 7.2s, max 22.3s |
| Audio lượt user trong `history` (có `audio_text`) | 3491 file, ~3.7h, TB 3.8s, max 17.1s |
| Tổng | **5036 utt, ~6.8h**, 16kHz mono |
| Public test | 634 mẫu, **không có transcript**; 1299 audio history cũng cần ASR |

- Max 22.3s < cửa sổ 30s của Whisper, không cần cắt.
- Nhãn **không nhất quán cách viết số** ("50 nghìn" vs "số không chín năm…"); có câu tự sửa ("5 giờ nhé, à không, 5 rưỡi"). Giữ nguyên nhãn, vì agent cần phần sửa lời.
- Dữ liệu nhỏ so với 1.5B tham số → nguy cơ overfit, ưu tiên LoRA / LR thấp.

## 2. Hạ tầng Modal

- **Volume `zaic-data`**: `manifest.json`, `train/audio/`, `train/train.jsonl` (đã upload). Public test audio (310MB) upload khi cần suy luận.
- **Volume `zaic-out`**: kết quả (`baseline.json`, `lora/`, `full/`, mỗi thư mục có model fp16 + `eval.json`).
- **Image**: debian_slim + torch 2.8.0, transformers 4.57.1, peft 0.17.1, accelerate, jiwer, librosa, soundfile. Model tải từ HF lúc chạy.
- **GPU**: baseline A100-40GB; train A100-80GB (fp32 master weights + bf16 autocast).
- Không còn ràng buộc đĩa/GPU dùng chung. Chi phí tính theo giờ GPU, nên mỗi lần chạy phải có lý do rõ.

## 3. Chia dữ liệu & chuẩn hóa

- Chia train/val **theo `sample id`** (seed 42, 8% mẫu): train 4631 utt (1422 final + 3209 history), val 405 utt (123 final + 282 history). History và lượt cuối của cùng một mẫu nằm cùng phía.
- Chuẩn hóa khi tính WER/CER: NFC, lowercase, bỏ dấu câu, gộp khoảng trắng. **Chưa** quy đổi chữ số ↔ chữ, nên WER hiện đã bị phạt oan ở các câu số (xem mục 6).
- Train giữ nguyên nhãn BTC.

## 4. Các bước

| # | Bước | Lệnh | Trạng thái |
|---|---|---|---|
| 0 | Manifest + upload data | `make_manifest.py`, `modal volume put` | Xong |
| 1 | Baseline zero-shot (go/no-go) | `modal run app.py --step baseline` | **Xong: WER 25.3%, CER 14.8%** → đi tiếp |
| 2 | LoRA | `modal run app.py --step train --mode lora --epochs 6` | **Xong: WER 7.6%, CER 4.8%** (beam 5, best epoch 3) |
| 2b | Thử loss: label smoothing / weighted CE / CTC+attention | `asr/modal_finetune_phowhisper.py` (xem bảng dưới) | **Xong: tốt nhất LS 0.1 + CTC 0.3, WER 6.6%** |
| 2c | Contextual biasing: prompt từ history | `--use-prompt --label-smoothing 0.1 --ctc-weight 0.3` | **Xong: WER 6.2% (điều kiện thi), 6.1% (prompt nhãn thật), lỗi tên 27→17, câu khớp 231→253** |
| 2c' | Contextual biasing v2: thêm `user_profile` + tên từ `tool_results` history (+ tên từ label kèm nhiễu khi train) | xem `PLAN_ASR_bias_v2.md` | Chưa |
| 2d | Bổ sung dữ liệu | xem `PLAN_data_bo_sung.md` | Chưa |
| 3 | Full fine-tune (nếu LoRA chưa đủ) | `--mode full` (LR 5e-6, grad ckpt) | Bỏ: loss đã bão hòa, capacity không phải nút thắt |
| 4 | Train lại bản cuối trên 100% data | `--full_data` (cùng cấu hình tốt nhất) | Chưa, chờ chốt 2c/2d |

Kết quả các cấu hình loss (cùng split, LoRA r=32, 6 epoch, early stop patience 2, beam 5 trên val 405 utt):

| Cấu hình | WER | CER | WER sau quy đổi số/chính tả | Ghi chú | HF |
|---|---|---|---|---|---|
| CE thường | 7.6% | 4.8% | 6.7% | eval dao động 12.7 → 15.1 → 11.0 | `fisherman611/pw-tmp-7f3a` |
| Label smoothing 0.1 | 7.0% | 4.4% | 6.3% | eval ổn định hơn | `fisherman611/pw-tmp-a81e` |
| Weighted CE β=2 (từ hiếm, số, tên) | 8.0% | 5.1% | 6.9% | không giúp: lỗi số đã ~1.5% | `fisherman611/pw-tmp-5b27` |
| LS 0.1 + CTC 0.3 | 6.6% | 4.0% | 5.9% | chạy đủ 6 epoch, eval phẳng từ epoch 3 | `fisherman611/pw-tmp-c3d9` |
| **LS 0.1 + CTC 0.3 + prompt history** | **6.2%** | **3.9%** | **5.4%** | prompt = transcript model sinh cho lượt user trước + text assistant; bỏ prompt lúc infer vẫn 6.7% | `fisherman611/pw-tmp-e46f` |

Chênh lệch dưới ~1.5 điểm nằm trong nhiễu (hai lần chạy cùng cấu hình weighted CE lệch 1.4 điểm ở epoch 3). Cả 4 model sai cùng chỗ: ~27 token tên riêng (tên nước ngoài, thương hiệu), ~125 câu sai từ nội dung; số gần như không sai. Chi tiết trong `PLAN_data_bo_sung.md` mục 0.

Hạ tầng: workspace Modal đã chuyển sang `trangntt03724` (volume `zaic-data`, `zaic-out`, secret `hf-token` tạo lại). Model cũ (CE, LS, weighted) vẫn nằm ở workspace `vyvy372004`. Chạy job bằng `modal deploy` + `Function.spawn` để tắt máy local vẫn chạy (xem docstring script).
| 5 | Đánh giá end-to-end | chạy agent với `audio_text` vs transcript ASR | Chưa |
| 6 | Đóng gói cho Docker | merge, fp16, thử faster-whisper/CTranslate2 | Chưa |

Cấu hình train: batch 16 × accum 2 = 32, warmup 10%, cosine, bf16, SpecAugment (mask time/feature 0.05), `language=vi, task=transcribe`.
- LoRA: r=32, alpha=64, dropout 0.05, target q/k/v/out_proj + fc1/fc2, LR 3e-4.
- Eval mỗi epoch, early stopping theo val WER (patience 2), giữ best checkpoint; sau train merge adapter, eval beam 5 và lưu fp16.
- Speed perturbation / nhiễu: chỉ thêm nếu val cho thấy overfit.

## 5. Tiêu chí quyết định

- Sau LoRA: so val WER với baseline 25.3%. Nếu giảm rõ rệt và đường eval không tăng lại → chuyển bước 4/5. Nếu còn cao hoặc overfit → thử full FT (bước 3).
- Cuối cùng quyết định bằng **tool-call end-to-end** (bước 5), không chỉ WER: lệch giữa chạy bằng `audio_text` gốc và bằng transcript ASR là phần lỗi do ASR.
- Chưa có agent LLM downstream để chạy bước 5; cần quyết định dùng agent nào.

## 6. Việc còn thiếu / rủi ro

| Vấn đề | Xử lý |
|---|---|
| WER bị phạt oan do khác kiểu viết số | Thêm bước quy đổi số ↔ chữ trước khi tính WER; báo thêm entity error rate (tên, số, giờ) |
| Overfit (chỉ 6.8h) | LoRA, LR thấp, early stop, SpecAugment |
| Hallucination / lặp từ trên audio ngắn | Kiểm tra trong phân tích lỗi trên val |
| Val nhỏ (405 utt) nên số đo nhiễu | Xem như chỉ báo; so sánh các cấu hình trên cùng val |
| Chi phí Modal | Chạy tuần tự, mỗi lần một cấu hình; theo dõi giờ GPU |
| Docker vòng private | Merge + fp16, thử CTranslate2; ghi version thư viện |
| Mất kết quả | Model nằm trong volume `zaic-out`; tải về local sau khi chốt bản cuối |
