# Plan train ASR v2: contextual biasing mở rộng (PhoWhisper-large, LS + CTC)

Cập nhật: 2026-11-08 (thêm kết quả bước 0, code, nơi chạy). Đi kèm `PLAN_PhoWhisper_finetune.md` (bước 2c là bản v1) và `PLAN_data_bo_sung.md`.

## CHỐT (2026-11-08): train ASR bằng biến thể V2

Đã chạy smoke test cả V1, V2, V3 trên Modal (A100-80GB, mỗi biến thể một GPU, chạy song song): khoảng 1600 utt đầu của manifest, 2 epoch, eval giữa epoch 150 utt, eval cuối đủ 405 utt val ở 3 điều kiện. Tag: `lora_ls0.1_ctc0.3_prompt_pv2[_le0.5[_ad0.3]]_smoke1600e2` trên volume `zaic-out`. Tốn khoảng $4.7 credit.

| | Điều kiện | WER | CER | Lỗi tên /289 | Tên trong prompt: sai | Chép nhầm |
|---|---|---|---|---|---|---|
| V1 | (a) hồ sơ + tên history | 15.4% | 9.2% | 54 | 8 / 47 | 4 |
| | (b) + lexicon | 15.1% | 9.0% | 51 | 14 / 167 | 21 |
| | không prompt | 16.6% | 9.8% | 72 | 19 / 47 | 3 |
| **V2** | (a) | 15.0% | 9.0% | 58 | 7 / 47 | 5 |
| | (b) | **14.5%** | **8.7%** | 51 | 10 / 166 | 22 |
| | không prompt | 16.0% | 9.6% | 75 | 18 / 47 | 3 |
| V3 | (a) | 15.8% | 9.4% | 53 | 6 / 47 | 7 |
| | (b) | 15.3% | 9.0% | 52 | 11 / 169 | 25 |
| | không prompt | 16.7% | 9.8% | 79 | 20 / 47 | 4 |

Đọc kết quả (smoke chỉ train 1/3 dữ liệu × 2 epoch, WER ~15% so với 6.2% của v1 đầy đủ nên **không so được với v1**):
- **Prompt v2 có ích.** Từ không prompt sang (a): lỗi tên giảm 15–25 token, WER giảm ~1 điểm. Trong 47 tên đã có trong prompt, model sai 18–20 khi không có prompt và chỉ còn 6–8 khi có.
- **Lexicon (b) giúp nhẹ, có giá.** WER giảm 0.3–0.5 điểm ở cả ba, lỗi tên giảm 1–7 token, nhưng chép nhầm tăng 4–5 lần (4–7 → 21–25) và tốn thêm một lượt giải mã/utt.
- **Ba biến thể ngang nhau trong mức nhiễu** (WER chênh ≤ 0.9 điểm, lỗi tên chênh ≤ 5 token). V3 không tốt hơn: chép nhầm cao nhất ở (b), WER cao nhất ở (a).

**Chọn V2** vì WER thấp nhất ở (b) và nhì ở (a), có dạy model chép tên (tên đúng + tên giả) mà đơn giản hơn V3. Đây là lựa chọn dựa trên chênh lệch nhỏ; lý do chọn V2 hơn V1 là V1 không bao giờ thấy tên đúng lẫn trong danh sách lúc train. V3 bỏ, chỉ xem lại nếu bản V2 đủ dữ liệu chép nhầm nhiều ở (a).

### Công thức train chốt

| Mục | Giá trị |
|---|---|
| Model | `vinai/PhoWhisper-large` rev `b9136a44…`, LoRA r=32, alpha=64, dropout 0.05, target q/k/v/out_proj + fc1/fc2 |
| Loss | CE + label smoothing 0.1; CTC 0.3 (head phụ trên encoder, bỏ lúc suy luận): L = 0.7·CE + 0.3·CTC |
| Optim | LR 3e-4, warmup 10%, cosine, batch hiệu dụng 32 (A100-80GB: 8×4; T4: 4×8 + gradient checkpointing), SpecAugment mask time/feature 0.05, seed 42 |
| Epoch | Run chọn model: 6 epoch + early stopping patience 2 theo WER val (như v1). Bản nộp: train lại trên 100% dữ liệu với số epoch = epoch tốt nhất của run trước, tắt early stopping (val nằm trong train nên không còn ý nghĩa). Trên Kaggle T4 dùng 4 epoch (đã giảm vì tốc độ) |
| Prompt lúc train | `<|startofprev|>` + hội thoại (nhãn thật các lượt trước, ≤110 token, giữ đuôi) + hồ sơ (`user_profile`, ≤40 token) + danh sách tên (≤70 token); tổng ≤ 221 token |
| Drop | Bỏ cả prompt với xác suất 0.4; bỏ riêng hồ sơ 0.3 và danh sách tên 0.3 |
| Danh sách tên (V2) | Tên trong `tool_results` của history. Với xác suất 0.5, và khi utt có tên trong label: thêm tên đúng từ label + 3–10 tên giả cùng loại trường (lấy từ lexicon split train), xáo trộn. `all_distract_prob = 0` |
| Cờ | `--label-smoothing 0.1 --ctc-weight 0.3 --use-prompt --prompt-v2 --label-ent-prob 0.5` |
| Eval | Val cố định 405 utt (split theo sample id, seed 42, 8%), điều kiện thi: ngữ cảnh lượt trước = transcript do model sinh. Beam 5. Báo (a), (b), không prompt, kèm lỗi tên và chép nhầm |

### Điều chưa chốt (phải quyết sau khi train đủ dữ liệu)
1. **Prompt lúc suy luận: (a) hay (b) lexicon.** Mặc định (a). Chuyển sang (b) khi: WER giảm ≥ 0.3 điểm trên run đủ dữ liệu, điểm end-to-end của LLM (`metrics/`) không giảm (chép nhầm có thể làm hỏng `contacts.search`), và thời gian thêm một lượt giải mã vừa trong 30 s/mẫu.
2. **Nơi chạy bản đủ dữ liệu**: Kaggle T4 (miễn phí, 3 tài khoản, nhưng train fp16 trên T4 chưa được kiểm, tốc độ chưa biết) hay Modal A100-80GB (~$5.5/lần, credit còn ít: metered $19.24).
3. **Kết quả chỉ ở quy mô smoke.** Không dùng bảng trên để kết luận V2 > V3; cần ít nhất một run đủ dữ liệu để thấy khác biệt.

### Dữ liệu bổ sung đang được sinh: cách nhập vào pipeline
Xem `PLAN_data_bo_sung.md` (nguồn, tỉ lệ, nguyên tắc). Quy ước kỹ thuật để dữ liệu mới chạy được với prompt v2 mà không đổi công thức trên:
- **Val giữ nguyên**: 405 utt thật (split theo sample id như v1). Dữ liệu mới **chỉ vào train**, để kết quả so được qua các lần chạy.
- **Mỗi utt mới** thêm vào `manifest_v2.json` cùng các trường như utt thật: `uid, audio, text, sid, kind, split="train", prev, profile, apps, hist_ents, label_ents`. Nếu utt không có hội thoại thì `prev=[]`, `hist_ents=[]`; nếu không có hồ sơ thì `profile=""` (prompt drop đã xử lý rỗng). Điền `label_ents` = các tên riêng (người, địa điểm, thương hiệu, kênh, bài hát…) có trong câu, cùng dạng `[app, field, giá trị]`; đây là chỗ dữ liệu tên riêng tự sinh giúp V2 nhiều nhất.
- **Lexicon** (`lexicon_v2.json`) dựng lại từ split train gồm cả dữ liệu mới, dùng cho tên giả lúc train và cho điều kiện (b) lúc test. Không đưa tên từ val hay public_test vào lexicon.
- Tỉ lệ: dữ liệu ngoài miền ≤ 2× lượng trong miền (6.8h), hoặc hai giai đoạn (ngoài miền trước, trong miền sau).
- Không dùng nhãn hay pseudo-label của public/private test; không đọc `*_environment/`.
- Khi có dữ liệu mới: chạy `make_manifest_v2.py` mở rộng (cần sửa để đọc nguồn bổ sung), `modal volume put` lại manifest + lexicon, **chạy lại smoke nhanh** để chắc không lỗi, rồi mới train đủ.

## 0. Hiện trạng và mục tiêu

- Bản tốt nhất (v1, `fisherman611/pw-tmp-e46f`): LoRA r=32, LS 0.1 + CTC 0.3, prompt = transcript các lượt user trước + text assistant. Val 405 utt, điều kiện thi: **WER 6.2%**, CER 3.9%, lỗi tên 17 token.
- Prompt v1 bỏ sót hai nguồn tên riêng **có sẵn lúc test** (đã kiểm trong `public_test.jsonl`: 634/634 mẫu có `user_profile`, 1106 lượt assistant có `tool_results`).
- Độ phủ trên 3871 từ viết hoa (không đứng đầu câu) của 5036 utt train:

| Prompt gồm | Phủ |
|---|---|
| Transcript + text assistant (v1) | 18.4% |
| + tên từ `tool_results` của history | 21.5% |
| + `user_profile` | **27.6%** |

- 1545 utt (lượt user đầu tiên của mỗi mẫu) hiện có prompt rỗng; với `user_profile` thì có ngữ cảnh.
- Mục tiêu: giảm lỗi tên riêng, giữ hoặc giảm WER, không tăng lỗi "chép nhầm tên từ prompt".

## 1. Ràng buộc luật

| Nguồn | Train | Eval / infer |
|---|---|---|
| `context.user_profile`, `history` (text + `tool_results` của assistant) | Có | Có |
| Transcript lượt user trước | Nhãn thật hoặc transcript model | **Chỉ** transcript do model sinh |
| `label.tool_results` / `label.final_state` (trong `train.jsonl`) | Có, kèm tên gây nhiễu | Không |
| `*_environment/`, `state.json` | **Cấm** | **Cấm** |
| Nhãn public/private test | **Cấm** | – |

## 2. Bước 0: kết quả (đã làm, không cần train)

Trên val 405 utt của v1 (điều kiện thi: prompt dựng lại đúng như v1 thấy, tức transcript model + cắt 180 token). Tên riêng = từ viết hoa không đứng đầu câu và không đứng ngay sau `. ? ! …` (cách đếm cũ tính cả "Nhắc", "Phiếu"… nên ra 23).

| Tên sai (19 / 289 tên) | Số | Trong lexicon train-split |
|---|---|---|
| Đã có trong prompt v1 mà vẫn sai | 5 | 3 |
| Chỉ có trong tên `tool_results` history (mới) | 1 | 1 |
| Chỉ có trong `user_profile` (mới) | 2 | 2 |
| Chỉ có trong label (không có lúc test) | 9 | 8 |
| Không có nguồn nào | 2 | 1 |

Ước lượng bằng code thật (`modal/test_bias_prompt.py`, bản nháp = hyp của v1):

| Prompt lúc test | Tên trong danh sách tên của prompt | Lỗi tên nằm trong danh sách |
|---|---|---|
| (a) hội thoại + hồ sơ + tên history | 47 / 289 | 5 / 19 |
| (b) (a) + lexicon train chọn theo app + bản nháp | 177 / 289 | 7 / 19 |

Kết luận:
- Nguồn mới lúc test chỉ chạm tới khoảng 3–7 trong 19 lỗi tên. **Trần lợi ích nhỏ**: WER có thể không đổi quá mức nhiễu; chỉ số chính là lỗi tên và chép nhầm.
- 5 lỗi tên đã có sẵn trong prompt v1 mà model vẫn sai: model chưa tận dụng prompt. Đây là chỗ V2 (dạy chép tên) có thể giúp.
- **Lexicon train** (2511 mục, 35 app, chỉ dựng từ split train) là nguồn hợp lệ lúc test vì là sản phẩm dựng từ train. Phủ 37% tên trong `tool_results` history của public_test. Dùng làm điều kiện eval (b), không cần biến thể train riêng: prompt V2 lúc train (tên đúng lẫn tên nhiễu) giống prompt (b) lúc test.
- (b) tốn thêm một lượt giải mã cho mỗi utt (bản nháp), phải tính vào 30 s/mẫu.

## 3. Prompt mới

Ba phần, mỗi phần có ngân sách token riêng (thay vì cắt cả chuỗi như v1). Phần quan trọng đặt gần cuối, sát transcript:

```
<|startofprev|> [hội thoại: transcript + text assistant, giữ cuối, ≤ 110 token]
                [hồ sơ: tên người dùng, địa chỉ nhà/công ty, thành phố, ≤ 40 token]
                [tên riêng: danh sách cách nhau bằng ", ", ≤ 70 token]
<|startoftranscript|><|vi|><|transcribe|><|notimestamps|> ...
```

- Tổng ≤ 220 token (1 + 220 + 4 token ép + 200 token sinh = 425 < 448). Whisper giới hạn prompt khoảng 223 token (một nửa context decoder 448). Prompt dài hơn v1 (180), nên phải kiểm tra lại bộ nhớ và độ trễ.
- **Trích tên từ `tool_results`** của history: chỉ lấy giá trị chuỗi của các trường dạng tên: `name, title, members, channels, channel, artist, location, provider, from_name, from, destination, origin, airline, station, bank, assignee, carrier, author`.
  - Bỏ text dài (`content, subject, snippet, message`), id, giờ, số.
  - Khử trùng lặp. Ưu tiên lượt gần nhất khi vượt ngân sách.
- Không có history thì chỉ có phần hồ sơ (lượt user đầu tiên).
- Dùng **một hàm dựng prompt chung** cho train, eval và infer trong Docker (đặt trong `modal_finetune_phowhisper.py`, sau này import vào runtime). Unit test: với một mẫu public_test, prompt chỉ đọc từ `context` + `history` + transcript model.

## 4. Các biến thể

Mọi biến thể giữ nguyên cấu hình v1: LoRA r=32/alpha=64 trên q/k/v/out_proj + fc1/fc2, LR 3e-4, LS 0.1, CTC 0.3, 6 epoch, early stop patience 2 theo val WER, batch 8 × accum 4, SpecAugment, cùng split (seed 42, 8% theo sample id).

| Biến thể | Prompt lúc train | Ý tưởng |
|---|---|---|
| V0 | v1 (có sẵn, không chạy lại) | Mốc |
| **V1** | Hội thoại + hồ sơ + tên từ history | Chỉ dùng nguồn có lúc test. Train ↔ test khớp nhất |
| **V2** | V1 + với xác suất 0.5: thêm tên trong `label.tool_results` của chính mẫu (lượt cuối) hoặc của các lượt sau (lượt history), trộn 3–10 tên gây nhiễu lấy từ mẫu khác, xáo thứ tự | Dạy "chép tên khi nghe thấy, bỏ qua khi không nghe thấy". Tăng số ví dụ có tên đúng trong prompt (phủ ~72% thay vì ~28%) |
| V3 | V2 nhưng danh sách tên chỉ toàn tên gây nhiễu ở 30% ví dụ | Chống chép mù nếu V2 chép nhầm nhiều |

- Prompt drop giữ 0.4 cho cả prompt. Thêm drop riêng 0.3 cho từng phần (hồ sơ, tên) để model chạy được khi phần nào đó rỗng.
- Tên gây nhiễu lấy từ cùng loại trường (tên người cho tên người, địa điểm cho địa điểm) để khó phân biệt.

## 5. Đánh giá

Trên val 405 utt, **luôn ở điều kiện thi**: lượt user trước dùng transcript model, tên chỉ từ `context` + `history`, không dùng label.

| Chỉ số | Ý nghĩa |
|---|---|
| WER, CER, WER sau quy đổi số/chính tả | Như v1 |
| Lỗi token tên riêng | Chỉ số chính |
| Lỗi tên, chia theo "tên có trong prompt" / "không có" | Tách phần do biasing |
| **Chép nhầm**: số từ trong hypothesis lấy từ danh sách tên của prompt mà không có trong reference | Không được tăng so với V0 |
| WER khi bỏ hẳn prompt | Model không được phụ thuộc prompt (v1: 6.7%) |
| Số câu khớp tuyệt đối | Như v1 (253/405) |

- Chênh WER < ~1.5 điểm là nhiễu (đã thấy ở v1). Nhìn lỗi tên và chép nhầm là chính. Nếu cần, chạy lại 2 seed cho biến thể tốt nhất.
- Sau khi chọn: đánh giá end-to-end bằng agent LLM với transcript ASR thay cho `audio_text` (điểm `metrics/`). Đây mới là thước đo thật, vì tên sai thường làm `contacts.search` rỗng.

## 6. Code (đã viết)

| File | Nội dung |
|---|---|
| `modal/bias_prompt.py` (mới) | Logic dùng chung train / eval / Docker: `items_from_row` (một dòng jsonl → các utt), `build_prompt` (ngân sách từng phần; eval tất định, train có drop + tên label + tên nhiễu), `build_lexicon`, `select_lexicon` (char-3gram so với bản nháp), `name_stats` (lỗi tên, tách theo có/không trong prompt, chép nhầm) |
| `modal/make_manifest_v2.py` (mới) | Ghi `manifest_v2.json` + `lexicon_v2.json`. Cùng split với v1, kiểm tra các trường v1 khớp `manifest.json` (không ghi đè v1) |
| `modal/test_bias_prompt.py` (mới) | Prompt dựng được cho mọi utt public_test chỉ từ context/history (không có nhãn); độ dài ≤ 221; số liệu bước 0 |
| `modal/asr_train.py` (mới) | Toàn bộ logic train/eval tách khỏi Modal; thêm `--time-budget-h`, `--eval-items`, `--base-fp16`. `modal_finetune_phowhisper.py` chỉ còn bọc hàm này |
| `kaggle/launch.py`, `kaggle/driver_template.py` (mới) | Upload dataset riêng tư, đẩy kernel T4 cho từng biến thể / tài khoản, resume, tải output |
| `modal/modal_finetune_phowhisper.py` (sửa) | Cờ `--prompt-v2 --label-ent-prob --all-distract-prob --n-distract-max --section-drop`; eval cuối ở 3 điều kiện: (a), (b) lexicon, không prompt, kèm `name_stats`; lưu CTC head cạnh checkpoint nên **resume được khi có CTC** (trước đây tắt); `--bs --accum`; tự chọn fp16 khi GPU không có bf16; chạy được bằng `python` thường với `ZAIC_DATA` / `ZAIC_OUT`; `--max-items` cho smoke test |

Lệnh:

```
python modal/make_manifest_v2.py data/train+public_test
modal volume put zaic-data modal/manifest_v2.json manifest_v2.json
modal volume put zaic-data modal/lexicon_v2.json lexicon_v2.json
# Modal (A100-80GB):
modal deploy modal/modal_finetune_phowhisper.py
python -c "import modal; f=modal.Function.from_name('lifemate-phowhisper-finetune','train'); \
  print(f.spawn(label_smoothing=0.1, ctc_weight=0.3, use_prompt=True, prompt_v2=True).object_id)"                       # V1
  ... f.spawn(..., prompt_v2=True, label_ent_prob=0.5)                                                                   # V2
# Không qua Modal (myserver: 1 biến thể / GPU):
CUDA_VISIBLE_DEVICES=0 ZAIC_DATA=<dir> ZAIC_OUT=<dir> python modal/modal_finetune_phowhisper.py \
  --label-smoothing 0.1 --ctc-weight 0.3 --use-prompt --prompt-v2 --bs 4 --accum 8
```

Chưa chạy thử trên GPU. Bước đầu tiên là smoke test `--max-items 20 --epochs 1`.

## 7. Nơi chạy và chi phí

v1 train 5453 s (~1.5 giờ, 870 bước) + eval cuối trên A100-80GB. v2 eval cuối nặng hơn (3 điều kiện, (b) giải mã 2 lượt).

| Nơi | GPU | Chi phí | Vấn đề |
|---|---|---|---|
| **myserver** (khuyên dùng) | 2×A100-40GB, đang rảnh, bf16 | 0 | Ổ `/mnt` còn ~11 GB; PhoWhisper-large chưa có trong cache (tải ~6 GB fp32) + mỗi model gộp ~3 GB. Cần dọn ổ hoặc chỉ lưu adapter. bs 4 × accum 8 + gradient checkpointing |
| Modal | A100-80GB $2.5/h | ~$5 / biến thể, V1 + V2 ≈ $10 | Credit còn ~$15, cần để dành cho SFT LLM |
| **Kaggle (đang dùng)** | T4 16GB (P100 đã bị ngừng, CLI tự chuyển sang T4), **không bf16** | 0 | 3 tài khoản × 30 giờ GPU/tuần (đã kiểm bằng `kaggle quota`), phiên ~12 giờ. Mỗi biến thể một tài khoản; train tối đa 8 giờ/phiên rồi resume. fp16. Code: `kaggle/` (README) |

## 8. Thứ tự

| # | Việc | Kết quả |
|---|---|---|
| 1 | Chọn nơi chạy; smoke test `--max-items 20 --epochs 1` | Chạy hết đường train → eval → lưu |
| 2 | V1 và V2 song song | Bảng: WER/CER, lỗi tên, chép nhầm, ở (a), (b), không prompt |
| 3 | V3 nếu V2 chép nhầm nhiều | – |
| 4 | Train lại bản tốt nhất trên 100% data (`--full-data`); lexicon dựng lại từ toàn bộ train | Model nộp |
| 5 | End-to-end với LLM (điểm `metrics/`) | – |

## 9. Rủi ro

- **Chép nhầm tên**: danh sách tên dài làm model thay từ đúng bằng tên gần giống. Theo dõi chỉ số chép nhầm; V3 để giảm.
- **Lệch train/test** ở V2: train có tên đúng thường xuyên hơn test. Eval luôn ở điều kiện thi để phát hiện.
- **Lỗi lan truyền**: transcript sai ở lượt trước đi vào prompt lượt sau (đã có ở v1).
- **Độ trễ**: prompt dài hơn → decode chậm hơn một chút; ASR history tối đa 12 audio, phải tính vào 30 s/mẫu.
- **Lợi ích có thể nhỏ**: lỗi tên còn chủ yếu là tên nước ngoài / thương hiệu, có thể nằm ngoài cả 28% phủ. Bước 0 trả lời việc này.
