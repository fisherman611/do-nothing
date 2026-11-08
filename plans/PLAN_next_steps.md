# Việc cần làm tiếp – ZAC 2026 LifeMate

Cập nhật: 2026-11-08. Tổng hợp các plan chi tiết: `PLAN_ASR_bias_v2.md`, `PLAN_SFT_qwen35_text.md`, `PLAN_tool_calling_LLM.md`, `REPORT_zeroshot_LLM.md`.

## 0. Hiện trạng

| Mảng | Đã có | Đang dở |
|---|---|---|
| ASR | PhoWhisper-large LoRA + LS 0.1 + CTC 0.3 + prompt history: **WER 6.2%** (val 405 utt, điều kiện thi) | Prompt v2 (hồ sơ + tên trong history + lexicon): code xong, dataset trên 3 tài khoản Kaggle, mới chạy smoke test. Train fp16 trên T4 chưa được kiểm |
| LLM | Zero-shot trên train (transcript gốc): Qwen3.5-4B cắt **37.5** điểm không thinking, **45.2** có thinking. Bản cắt 4.206B, tổng hệ thống 5.75B < 6B | Plan SFT xong, chưa có code |
| Hệ thống | Agent loop + scorer `metrics/`, vLLM trên Modal | Chưa đo độ trễ thật, chưa ghép ASR → LLM, chưa có Docker |

Tài nguyên: Modal còn ~$15 credit. Kaggle 3 tài khoản × 30 giờ GPU/tuần (V1 đã dùng 1.4 giờ, làm mới 14/11). myserver 2×A100-40GB rảnh nhưng ổ `/mnt` còn ~11 GB.

## 1. Thứ tự ưu tiên

| # | Việc | Phụ thuộc | Nơi chạy | Chi phí ước | Kết quả cần có |
|---|---|---|---|---|---|
| 1 | **Quyết định chỗ chạy SFT LLM** (dọn ổ myserver hay dùng Modal) | – | – | – | Chọn xong |
| 2 | ~~Smoke ASR V1/V2/V3~~ **Xong trên Modal (2026-11-08)**: chọn **V2** (xem `PLAN_ASR_bias_v2.md` mục CHỐT). Kaggle fp16 T4 vẫn chưa kiểm | – | Modal | ~$4.7 | Đã có |
| 3 | **Nhập dữ liệu ASR mới** (bạn đang sinh) → manifest/lexicon v2 → smoke nhanh → train đủ V2 (run chọn model, rồi train lại 100%) | dữ liệu mới | Kaggle T4 hoặc Modal | Modal ~$5.5/lần | Model ASR nộp; chọn (a) hay (b) lexicon |
| 4 | **Bước 0 của SFT**: kernel linear-attention, LoRA target, vLLM serve LoRA, test template từng token | 1 | myserver / Modal | < $2 | Biết train được và serve được |
| 5 | `build_sft.py` + chia val (app held-out + held-in) + chấm lại zero-shot trên đúng val | 4 | local | 0 | Dữ liệu SFT, baseline cùng tập |
| 6 | **SFT hướng A** (LoRA, không thinking) | 5 | myserver / Modal | ~$5–10 trên Modal | Điểm val so với zero-shot thinking 45.2 |
| 7 | **Benchmark độ trễ** 1 mẫu (ASR history + ASR lượt cuối + vòng LLM) trên GPU gần A30 | 3, 6 | Modal L4/A10G | ~$1 | Có kịp 30 s/mẫu không, thinking có dùng được không |
| 8 | Chọn ASR tốt nhất, train lại trên 100% train (lexicon dựng lại từ toàn bộ train) | 3 | Kaggle | ~13 giờ | Model ASR nộp |
| 9 | Chạy ASR trên toàn bộ train để có transcript lỗi thật | 8 | Kaggle | vài giờ | Dữ liệu cho SFT có nhiễu ASR (30–50% lượt user) |
| 10 | **End-to-end**: transcript ASR → agent SFT → `metrics/` | 6, 8 | Modal | ~$2 | Điểm thật khi có lỗi ASR |
| 11 | SFT hướng C (tăng cường: lượt history, nhiễu catalog, nhiễu ASR), hướng B (thinking chưng cất) nếu độ trễ cho phép | 6, 7, 9 | – | – | Điểm val app held-out tăng |
| 12 | Kiểm tra tính tất định (chạy lại cùng tập, so tool_calls) | 6 | Modal | < $1 | Đủ điều kiện luật chạy lại |
| 13 | Quantize / tối ưu serving (bạn làm) | 7, 10 | – | – | Nhanh hơn, điểm không giảm |
| 14 | **Docker offline**: vLLM + PhoWhisper + `zaic_tools`, seed cố định, chạy public_test, nộp | 10, 13 | – | – | `submission.jsonl` hợp lệ, điểm leaderboard |

Việc 2–3 (ASR, Kaggle) và 4–6 (LLM) chạy song song được.

## 2. Mốc quyết định

- **Sau việc 3**: prompt (b) lexicon chỉ dùng khi WER giảm ≥ 0.3 điểm trên run đủ dữ liệu, điểm end-to-end LLM không giảm, và thời gian vừa 30 s/mẫu; mặc định (a).
- Chạy Kaggle T4: nếu tốc độ fp16 < ~0.4 mẫu/giây thì chuyển sang Modal.
- **Sau việc 7**: tổng độ trễ tệ nhất > ~20 s thì bỏ thinking, dồn sức vào SFT không thinking.
- **Sau việc 10**: chênh điểm giữa transcript gốc và ASR lớn thì ưu tiên nhiễu ASR trong SFT (việc 11) hơn là thêm biến thể ASR.

## 3. Việc nhỏ, làm khi tiện

- Sửa `agent/score.py` sang ngưỡng fuzzy > 0.6 (hoặc chỉ dùng `metrics/`).
- Phân tích 155 bước thinking không có `</think>` mà chưa chạm giới hạn token.
- `PLAN_tool_calling_LLM.md`: phần pipeline còn ghi Qwen3-4B-2507 là mặc định ở vài chỗ.

## 4. Bảo mật và quản lý

- **Đổi token**: HF, Modal, 3 token Kaggle (`KGAT_...`) đã được dán trong chat.
- Dataset Kaggle `zaic-asr-v2-data` để riêng tư, không chia sẻ (dữ liệu cuộc thi).
- Commit / push: bạn làm.
