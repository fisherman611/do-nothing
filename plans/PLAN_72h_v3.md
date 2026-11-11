# PLAN 72h (v3) — 11/11 17:15 → 14/11 17:15

Căn cứ: `plans/PLAN_v3_improve.md` (thước đo M1/M2/M5, mục N/F/I/D). Thay cho `plans/PLAN_48h_v3.md` (bản 48h, 11/11 16:40).
Mốc: **public 62.76** = ASR v1 large `e46f` → bộ lọc `pack_b` → FC `fmtA_48e` + N1 `--retry-empty` (`result/submission_fmtA_48e_e46f_n1.jsonl`). Lịch sử: 59 (ASR v2 medium) → 62.04 (ASR v1 large) → 62.76 (+N1).

## 1. Kết quả mới → ưu tiên

| Kết quả | Số liệu | Ý nghĩa |
|---|---|---|
| N3: ASR v2 medium → v1 large | public 59 → **62.04** | điểm nhạy với ASR câu cuối; model lớn thắng data mới |
| M2: FC gặp lỗi ASR | val_in 76.5 → 63.5; câu cuối có lỗi −18.6; câu điều kiện −25.2 | khoảng mất điểm lớn nhất còn lại → **FC v3 ưu tiên 1** |
| N1: tìm lại khi rỗng | bật 9/634 mẫu public; M2 không đổi; **public +0.72** | đúng dự đoán (+0.3–0.7); giữ, luôn bật |
| Trần N4 (địa danh) | 1–2 ca rõ ràng trên public | hạ ưu tiên |
| Qwen3-ASR-0.6B | 1 epoch: WER 13.2% (v1 large 6.23%) | chốt sau bản 6 epoch, không đầu tư thêm |
| M5 | Q1 app mới 84.5% vs app cũ 97.0%; conf < 0.7 = 5% bước, 40% lỗi | N8 đáng làm |

Thứ tự ưu tiên: **FC v3 (chịu lỗi ASR + app lạ)** > **ASR large + data mới** > **luật lúc infer** (N8, I4, I5, nghe lại với tên từ kết quả tool) > N4. Docker bắt buộc. Bỏ: N5, N6, L1, Qwen (sau khi có số).

## 1b. Phân tích bổ sung bằng metric mới (11/11 tối)

Metric thêm vào `metrics/` (chẩn đoán, không phải điểm chính thức): `python -m metrics calls` (chọn tool vs độ đúng tham số) và
`python -m metrics asr-detail` (entity error rate theo loại, code-switching WER, RTF). Kết quả: `agent/runs/metrics/`.
Không dùng: 4 chỉ số streaming và concurrent streams (bài thi chấm file wav theo lô).

**FC `fmtA_48e` trên val_in (128 mẫu): sạch (M1) → có lỗi ASR (M2)**

| Chỉ số | M1 | M2 |
|---|---|---|
| Chọn tool: precision / recall / F1 | 0.944 / 0.926 / 0.935 | 0.910 / **0.853** / 0.880 |
| Call thiếu / call thừa | 20 / 15 | **40** / 23 |
| Call đúng tool mà đúng hết tham số | 0.909 | 0.845 |
| Tham số `id` / `enum` | 0.980 / 0.968 | 0.957 / 0.963 |
| Tham số `datetime` | 0.949 | **0.811** |
| Tham số `fuzzy` (nội dung tin nhắn, ghi chú, truy vấn) | 0.759 | **0.600** |
| Tham số `text` | 0.958 | 0.848 |
| Mẫu đúng hoàn toàn / sai chọn tool / sai tham số / cả hai | 87 / 19 / 20 / 2 | 70 / 27 / 23 / 8 |
| Câu điều kiện (13 mẫu) đúng hoàn toàn | 10 | 5 |

- 19 mẫu rơi khỏi nhóm đúng khi có lỗi ASR: 9 do chọn tool, 7 do tham số, 3 do cả hai → **hai nguyên nhân ngang nhau**.
- Lỗi chọn tool chủ yếu là **thiếu call**: vế câu bị nghe sai thì model bỏ luôn hành động của vế đó (vd "điều hoà phòng khách để 26 độ" → "điều hòa phần khách để hay xấu được" → không gọi `climate_set`), hoặc hỏi lại thay vì làm.
- Lỗi tham số: model chép nguyên từ nghe sai vào nội dung ("đổ xăng" → "Ăn đồ xanh"), nhầm tên người, sai ngày giờ. `id` và `enum` gần như không bị ảnh hưởng.
- Ngay cả khi sạch, tham số `fuzzy` chỉ đúng 76% (nội dung tin nhắn/ghi chú diễn đạt khác nhãn) → lỗi riêng của FC, không do ASR.

**ASR trên 405 đoạn giữ riêng** (183 thực thể có nhãn, thay cho phép đo cũ 55 thực thể)

| Chỉ số | v1 large `e46f` | v2 medium | Qwen 0.6B (1 epoch) | medium gốc |
|---|---|---|---|---|
| WER tất cả / câu cuối | 6.23 / 7.15 | 6.85 / 7.19 | 13.17 / 14.56 | 25.75 / 25.94 |
| WER câu có từ nước ngoài hoặc mã (117) / câu thuần Việt (288) | 5.99 / 6.36 | 6.81 / 6.88 | 13.87 / 12.82 | 30.14 / 23.53 |
| Sai thực thể có nhãn (183) | **9.3%** | 13.1% | 27.3% | 41.0% |
| – tên người (47) | 0.0% | 6.4% | 19.1% | 38.3% |
| – địa điểm (17) | 5.9% | 11.8% | 41.2% | 41.2% |
| – tên khác: bài hát, việc, món, kênh (119) | 13.4% | 16.0% | 28.6% | 42.0% |
| Số (30) | 10.0% | 23.3% | 46.7% | 90.0% |
| Mã chữ-số (34) | 0.0% | 2.9% | 5.9% | 97.1% |
| Từ nước ngoài (122) | 6.6% | 7.4% | 22.1% | 60.7% |

- v1 large hơn v2 medium rõ nhất ở **tên người và số**, khớp với việc public tăng 3 điểm.
- Sau fine-tune, câu có từ nước ngoài **không** khó hơn câu thuần Việt → vấn đề trên public là **tên chưa từng gặp**, không phải tiếng Anh.
- Lỗi còn lại của v1 large tập trung ở **tên khác** (tựa bài hát, tên việc, món ăn) và số.
- RTF v1 large trên T4 (beam 5) = 0.41; trên L40S p50 0.45 s/đoạn.

**Điều chỉnh plan từ phân tích này**

| Bước | Điều chỉnh |
|---|---|
| A1 (data FC v3) | (1) ≥ 50% mẫu có nhiễu là mẫu nhiều call, nhiễu rơi vào vế chứa hành động thứ 2, thứ 3; (2) nhãn nội dung/tên giữ bản sạch để model học sửa từ nghe sai theo ngữ cảnh (danh bạ, kết quả tool); (3) có nhiễu ở cụm ngày giờ và số, gồm cả dạng chữ ↔ chữ số |
| A3 / A5 (cổng FC v3) | thêm trên M2: recall chọn tool ≥ 0.90 (hiện 0.853), `fuzzy` ≥ 0.70 (0.600), `datetime` ≥ 0.90 (0.811) |
| B2 (TTS miền mới) | trọng tâm là tên chưa gặp: tựa/sự kiện/sản phẩm/món ăn, địa danh, số và mã; không cần riêng câu tiếng Anh |
| B3 / B5 (chọn ASR) | dùng "sai thực thể có nhãn" (183) + WER câu cuối, thay phép đo 55 thực thể |
| C (tuỳ chọn, sau C2) | xem 20 mẫu sạch sai tham số `fuzzy` để biết có sửa được bằng luật/định dạng không |
| D1 | đo RTF ASR + bộ nhớ GPU trong image (dùng `asr-detail` với `secs`) |

## 1c. Lỗ hổng của bản 48h và phần bổ sung (11/11 17:15)

| Lỗ hổng | Bằng chứng | Bổ sung (bước) |
|---|---|---|
| **App mới** gần như chưa có biện pháp và không có thước đo | 44% mẫu public gọi app mới; Q1 bước đầu 84.5% vs 97.0%; 13/18 ca "tìm thấy rồi vẫn dừng" ở app mới; val_ood đã lẫn vào train | **A0** thước đo app lạ M3b; **A1** thêm F3b (cặp app trùng chức năng) + 28 app tổng hợp `synth/newapps`; **C1** thêm I5 (gợi ý luồng search → book từ catalog) |
| **FC trên câu sạch** có trần 76.5 | M1: 19 mẫu sai chọn tool, 20 mẫu sai tham số; tham số `fuzzy` đúng 76%; FC mới train 1 epoch trên 20k/24.6k mẫu | **A0b** đọc 40 mẫu sạch bị sai; **A4** vòng 2 dùng nốt data chưa train + sửa theo A0b |
| **Chưa dùng kết quả tool để sửa lỗi nghe** | tên đúng nằm sẵn trong kết quả tool (sự kiện, danh bạ, món); N2 cũ xếp muộn và tốn công | **C2** nghe lại câu cuối với tên từ kết quả tool đưa vào prompt ASR (thay cách làm của N2), ghép với N1 |
| Không tất định | hai lần chạy cùng cấu hình lệch 6/634 mẫu; bộ chấm có chế độ trung bình 5 lần | **D1** cố định seed + greedy; **D3** chạy lại 2 lần, so khớp |
| Quyết định hỏi lại khi có lỗi ASR | M2: thiếu 8 lần `sys.ask_missing_info` | thêm chỉ số theo dõi ở cổng **A3 / A5** |

## 2. Bốn luồng chạy song song

Mỗi luồng tự nối tiếp bên trong; các luồng chạy cùng lúc vì dùng tài nguyên khác nhau.

| Luồng | Tài nguyên | Nội dung |
|---|---|---|
| **A — FC** | local (dựng data, đọc lỗi) + Modal H100 (train, đo) | thước đo app lạ → FC v3 vòng 1 → đo → vòng 2 → đo |
| **B — ASR** | Kaggle 3 tài khoản (miễn phí) + local (sinh text TTS) | L2, Qwen (đang chạy) → TTS miền mới → ASR v3 → (vòng 2 nếu cần) |
| **C — Luật lúc infer** | local + Modal vLLM (đo, dùng chung server với A) | N8 + I4 + I5 → nghe lại với tên từ kết quả tool → N4 → chọn tổ hợp |
| **D — Docker** | local + Modal (chạy thử) | image offline → 20 mẫu → 634 mẫu → kiểm tất định → bản nộp |

## 3. Các bước theo thứ tự (phụ thuộc)

| Bước | Luồng | Việc | Cần xong trước | Song song với | Tiền | Điều kiện giữ / cổng |
|---|---|---|---|---|---|---|
| B0 | B | PhoWhisper-medium gốc chạy trên audio train + TTS (10.8k đoạn) → lỗi nghe nhầm thật | – | A0, A1, B1 | $0 (Kaggle duongantbat) | đủ 10.8k transcript |
| B1 | B | *Đang chạy:* Qwen3-ASR 6 epoch (→ ~21:00), L2 = `e46f` + 2 epoch data v2.1 (→ ~23:30) | – | mọi thứ | $0 | Qwen: tiếp chỉ khi WER < 8% |
| A0 | A | **Thước đo app lạ M3b:** lấy mẫu val_in, đổi vỏ (tên app/tool/tham số/mô tả + thực thể) sang miền chưa gặp, không trùng tool của test; đo mốc `fmtA_48e` (sạch + nhiễu M2) | – | B0, A1 | ~$0.5 | chạy được trong `ToolEnv` đổi vỏ; mốc ghi vào plan |
| A0b | A | Đọc 40 mẫu **sạch** bị sai của `fmtA_48e` (19 chọn tool + 20 tham số, từ `metrics calls`) → phân loại nguyên nhân | – | B0 | $0 | bảng nguyên nhân → đưa vào A1/A4 |
| A1 | A | Data FC v3 vòng 1: nhiễu ASR từ B0 pha loãng kiểu M2, 60% mẫu, nhiễu ở câu cuối; ≥ 50% mẫu có nhiễu là mẫu nhiều call, nhiễu rơi vào vế chứa hành động thứ 2–3; nhãn nội dung/tên giữ bản sạch; nhiễu ở ngày giờ và số; câu điều kiện + nhiều yêu cầu ≥ 25%; mẫu "tìm rỗng → tìm lại → làm tiếp" và "tìm thấy → đặt/mua"; **app lạ:** mẫu đổi vỏ 2% → ~20%, **F3b** cặp app trùng chức năng (cùng câu, hai bộ app cài khác nhau → hai tool khác nhau), **28 app tổng hợp** `synth/newapps` (giữ riêng 4–5 app làm val) | B0 (phần nhiễu) | B1, C1 | $0 | không rò val/M3b; đủ 147 tool; đúng phân bố |
| C1 | C | **N8** (conf < 0.7 → hỏi lại với catalog đầy đủ) + **I4** (2–3 giả thuyết ASR câu cuối, bỏ phiếu ở bước chọn tool) + **I5** (gợi ý luồng từ catalog: tool ghi cần id của tool đọc nào → nhắc bước kế sau khi tìm thấy) | – | A1, B1 | ~$1.5 | từng thứ: M2 tăng, M1 tụt ≤ 1, M3b không tụt (so cặp) |
| B2 | B | Text TTS miền mới (2–3k câu, trọng tâm **tên chưa gặp**: sự kiện, sản phẩm, món ăn, tựa; tỉnh thành, ga, bến, sân bay; số và mã chuyến — nguồn chung, không lấy từ public) → sinh audio; giữ riêng 300 câu + 1–2 giọng làm val miền mới | – | A2 | TTS: nhỏ | nghe thử 20 câu |
| A2 | A | Train FC v3 vòng 1: `--init-adapter` từ `fmtA_48e`, ~12k ví dụ, LR 5e-5, 1 epoch | A1 | B2, B1 (đêm) | ~$10–12 | – |
| B3 | B | Chấm L2 trên 405 đoạn so với `e46f` | B1 | A2 | $0 | thắng WER câu cuối **và** tỉ lệ sai thực thể có nhãn (183 thực thể, `metrics asr-detail`) |
| S-A | – | Nếu B3 qua: ASR public bằng L2 → FC `fmtA_48e` + N1 (+C1 nếu qua) → **nộp A** | B3 | A2 | ~$1.5 | |
| B4 | B | Train ASR v3: từ bản large tốt nhất (L2 hoặc `e46f`) + data v2.1 + TTS miền mới, 2–3 epoch | B2, B3 | A2, A3, D1 | $0 (Kaggle, 8–12 h, 2 phiên nếu cần) | – |
| A3 | A | Gộp FC v3 vòng 1 → đo M1 + M2 + **M3b** + `metrics calls` | A2, A0 | B4, D1 | ~$2 | **M2 ≥ 68**, M1 ≥ 75.5, **M3b ≥ mốc + 3**, câu điều kiện tụt ≤ 15; trên M2: recall chọn tool ≥ 0.90, `fuzzy` ≥ 0.70, `datetime` ≥ 0.90, thiếu `sys.ask_missing_info` ≤ 4 (hiện 8) |
| S-B | – | Nếu A3 qua: public với ASR tốt nhất → **nộp B** | A3 | D1, C2 | ~$1 | |
| D1 | D | Image offline (PhoWhisper-large + FC gộp + bộ lọc + vLLM hoặc HF generate + `submit.py`); **seed cố định, greedy**; chạy 20 mẫu tắt mạng; đo thời gian/mẫu, RTF của ASR và bộ nhớ GPU | – | A3, B4, C2 | ~$1 | chạy offline được; ghi p50/p95, bộ nhớ |
| C2 | C | **Nghe lại với tên từ kết quả tool:** sau tool đọc trả danh sách, giải mã lại audio câu cuối với các tên vừa trả về trong prompt thiên vị của ASR → FC đi tiếp với transcript đã sửa; ghép với N1 (tìm rỗng → liệt kê → nghe lại) | C1 | D1, A4, B4 | ~$1.5 | M2 tăng, M1 tụt ≤ 1; thời gian thêm ≤ 2 s/mẫu |
| B5 | B | Chấm ASR v3 trên 405 đoạn + val miền mới (B2) | B4 | A4, C2 | $0 | thắng bản large đang dùng ở cả hai tập |
| S-C | – | Nếu B5 hoặc C2 qua: public với ASR v3 + FC tốt nhất + C2 → **nộp C** | B5, C2 | A4 | ~$1.5 | |
| A4 | A | Vòng 2: đọc lỗi FC v3 trên M2 + M3b + kết quả A0b → sửa data; thêm phần data chưa từng train (4.6k) → train tiếp 1 epoch từ vòng 1. *Lựa chọn của chủ repo:* train từ đầu ~25k × 2 epoch, bỏ val_ood khỏi train (có lại thước đo M3 gốc, đắt hơn ~$20) | A3 | C2, D1, B5 | $12–18 (hoặc ~$36) | – |
| B6 | B | (nếu B5 không qua hoặc còn quota Kaggle) ASR v3 vòng 2: thêm giọng TTS, tăng cường audio mạnh hơn, hoặc thêm epoch | B5 | A4, C3 | $0 | như B5 |
| C3 | C | N4: sửa địa danh ở mức tham số | C2 | A4 | ~$0.5 | M2 không tụt; M1 tụt ≤ 1 |
| A5 | A | Gộp + đo FC v3 vòng 2 (M1, M2, M3b, `metrics calls`) | A4 | D2, C3 | ~$2 | hơn vòng 1 trên M2 và M3b (so cặp), M1 ≥ vòng 1 |
| C4 | C | Chọn tổ hợp cuối: bật/tắt từng thứ (N1, N8, I4, I5, C2, N4) với FC + ASR tốt nhất, so cặp trên M1/M2/M3b | A5, B5/B6, C3 | D2 | ~$3 | chỉ giữ thứ không làm M1 tụt > 1 |
| S-D | – | Public với tổ hợp cuối → **nộp D** | C4 | D2 | ~$1 | |
| D2 | D | Docker với model + luật đã chọn: chạy đủ 634 mẫu offline; ngân sách thời gian (gần 30 s thì dừng, nộp call đã có); nếu chậm: giảm beam / ASR fp16 | D1, C4 | – | ~$2 | 0 lỗi; p95 < 30 s; submission khớp bản chạy online |
| D3 | D | Chạy lại D2 thêm 2 lần → so state_hash từng mẫu; đóng gói theo quy định BTC; ghi hướng dẫn chạy | D2 | – | ~$2 | 3 lần chạy khớp nhau (hoặc lệch ≤ 2 mẫu, có giải thích) |

## 4. Lịch theo giờ

| Khung giờ | A — FC | B — ASR | C — Luật infer | D — Docker |
|---|---|---|---|---|
| **11/11** 17:15–20:00 | **A0** thước đo app lạ; **A1** dựng data | **B0** sinh lỗi ASR; B1 đang chạy | – | – |
| 20:00–22:30 | A1 hoàn tất (ghép nhiễu B0); **A0b** | Qwen xong → chốt | **C1** N8 + I4 + I5 (code) | – |
| 22:30–01:00 | **A2** train vòng 1 (chạy nền) | **B3** chấm L2 → **S-A**; **B2** text TTS | C1 đo | – |
| 01:00–07:00 (không cần người) | A2 xong | B2 sinh audio | – | – |
| **12/11** 07:00–10:00 | **A3** đo → **S-B** | **B4** train ASR v3 (chạy nền) | – | **D1** dựng image |
| 10:00–15:00 | đọc lỗi vòng 1 | B4 | **C2** nghe lại với tên từ tool (code) | D1 chạy 20 mẫu, đo giờ + bộ nhớ |
| 15:00–20:00 | **A4** data vòng 2 | B4 → **B5** chấm | C2 đo | D1 sửa lỗi |
| 20:00–24:00 | A4 train vòng 2 (chạy nền) | **S-C** (ASR v3 + C2); B6 nếu cần | **C3** N4 | – |
| 00:00–07:00 (không cần người) | A4 xong | B6 | – | – |
| **13/11** 07:00–11:00 | **A5** đo vòng 2 | chấm B6 | – | – |
| 11:00–17:00 | – | – | **C4** chọn tổ hợp → **S-D** | **D2** build bản cuối |
| 17:00–24:00 | – | – | – | D2 chạy 634 mẫu offline |
| **14/11** 00:00–10:00 | – | – | – | **D3** chạy lại 2 lần, so khớp |
| 10:00–17:15 | dự phòng; cập nhật plan; đẩy code + plan (qua myserver), model + data (qua trạm → HF) | | | đóng gói nộp |

## 5. Chi phí và lượt nộp

- Modal: FC vòng 1 ~$10–12 + vòng 2 $12–18 + đo (M1/M2/M3b nhiều lượt) ~$12 + chạy public ~$5 + Docker ~$5 = **~$45–52**. Train từ đầu ở vòng 2: thêm ~$20. Nếu train FC trên myserver: còn ~$22.
- Kaggle: miễn phí; mỗi tài khoản 30 h GPU/tuần (đã dùng ~8 h ở huthnhlng, ~8 h ở andngdngtrnhhoian, ~0.5 h ở duongantbat).
- Lượt nộp: A + B + C + D = 4.
- Kỳ vọng thô: public 62.8 → ~67–71 (các mục sửa trùng loại lỗi nên không cộng thẳng).

## 6. Quy tắc

- Không tune theo public test; public chỉ để chẩn đoán và lấy điểm nộp. Không đọc `*_environment/`.
- Chọn FC bằng M1 + M2 + M3b (so cặp; M2 chỉ thấy thay đổi ≥ 4–6 điểm). Chọn ASR bằng 405 đoạn giữ riêng + val miền mới.
- Không chạy gì trên myserver khi chưa được phép. Data/model lên HF qua trạm. Monitor bằng Haiku.
- Một bước không qua cổng thì bỏ, luồng đó đi tiếp với bản đang tốt nhất; các luồng khác không phải chờ.

## 7. Cần chủ repo quyết

1. Duyệt Modal ~$45–52 (kiểm số dư trước), hoặc cho train FC trên myserver.
2. Số lượt nộp còn lại, hạn chót, hạn nộp Docker (plan giả định còn ≥ 72 giờ và ≥ 4 lượt).
3. A4: train tiếp từ vòng 1 (mặc định) hay từ đầu ~25k × 2 epoch (thêm ~$20, có lại thước đo app lạ gốc).
