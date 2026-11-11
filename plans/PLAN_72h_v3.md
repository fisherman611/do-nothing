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

## 1d. Kết quả A0b — 41 mẫu val_in SẠCH mà `fmtA_48e` vẫn sai (11/11 tối)

Đọc tay từng mẫu (nguồn: `metrics calls` trên `fmtA_48e_notau.val_in`). Mỗi mẫu xếp vào một nhóm chính:

| Nhóm | Số mẫu | Mô tả | Hướng sửa (bước) |
|---|---|---|---|
| Quyết định hỏi lại / làm | 9 | hỏi thừa (1246, 1386), không hỏi khi thiếu thông tin (0961, 1427, 1469, 1605, 1733, 1333), hỏi sai tham số (0684) | A1: thêm cặp mẫu "đủ thông tin → làm" / "thiếu → hỏi" cùng một câu, đổi một chi tiết |
| Bước đọc/ghi thừa hoặc thiếu do phải suy từ ngữ cảnh ("lịch đó", "lúc nãy", "cái đó") | 11 | 0708 (tạo lại thay vì sửa), 0998, 1156, 1193, 1046, 1154, 0966, 1316, 1593, 1612, 1714 | A1: nhiều mẫu hỏi tiếp dùng đại từ; mẫu câu điều kiện có kết quả tool |
| Nội dung diễn đạt khác nhãn (tin nhắn, ghi chú, tiêu đề) | 8 | 0468, 0647, 0730, 0833, 0814, 1133, 1236, 1404 — thường đúng ý nhưng ngắn hơn/dài hơn nhãn, token-F1 ≤ 0.6 | A1: quy ước nhất quán — nội dung chép đủ dữ kiện từ lời nói và kết quả tool; **không chỉnh được hết** |
| Tham số tuỳ chọn bỏ sót hoặc điền khác | 7 | `near`, `channel`, `date`, `days`, `location`: 0304, 0616, 1388, 1128, 1578, 1673, 1528 | A1: mẫu có/không tham số tuỳ chọn theo chi tiết trong câu |
| Số, id, đối tượng sai | 6 | tính phút từ khoảng giờ (1675: 35 thay vì 40; 1186), `rule_id` (1341), `locked` (1746), `query` (1407), thiếu `contacts.search` (1531) | A1: mẫu có tính khoảng thời gian; cân nhắc luật kiểm tra số học ở infer (nhỏ) |

- Hai nhóm đầu (20/41 mẫu) là **quyết định ở mức bước** (hỏi/làm, bước nào) — cùng loại với nhóm lỗi M2 làm tăng thêm. Nhóm nội dung (8/41) phần lớn là sai khác cách diễn đạt, một phần là trần của FC.
- Tự sửa giữa câu ("à không…") có mặt ở 4 mẫu (0961, 1154, 1128, 1236); tỉ lệ này trong data đã ~10% (§1e) nên giữ nguyên.

## 1e. Đo thành phần data hôm qua (`fmtA_48e_20k`) — hiệu chỉnh mục tiêu của A1 (11/11 19:00)

Đo trên 1417 hội thoại train (không có val_in), 7901 mẫu trong 20k và 128 mẫu val_in:

| Loại câu | Train | 20k (theo mẫu) | val_in | Kết luận |
|---|---|---|---|---|
| Điều kiện, định nghĩa hẹp (nếu / không thì) | 8.5% | 8.5% | 9.4% | – |
| Điều kiện, định nghĩa rộng (nếu / thì / trừ / nhưng / khi nào) — định nghĩa của phân tích M2 (33/128) | 34.9% | 34.9% | 28.1% | **đã đủ**; mục tiêu "≥ 25%" trong bản trước là đã đạt |
| Tự sửa giữa câu ("à không…", "khoan", "nhầm") | 10.7% | 9.1% | – | **đã đủ** |
| Nhiều call: 3+ / 2 / 1 | 41% / 28% / 24% | 40% / 29% / 25% | – | đã đủ |
| Có hỏi lại (`sys.ask_missing_info`) / không gọi tool | 14.0% / 7.2% | 17.6% / 5.8% | – | đã có đủ |
| Loại mẫu: gốc / đổi tên tool (alias) / phiên âm / đổi giao diện (iface) | – | 1417 / 1941 / 1840 / **385 (4.9%)** | – | iface còn ít; chỉ có 14 bản đổi (`synth/iface_map.json`) và 382 mẫu |

- **Điều chỉnh:** câu điều kiện **không thiếu** trong data; M2 tụt 25 điểm ở nhóm này vì lỗi nghe rơi vào từ nối điều kiện hoặc vế câu, không phải vì ít mẫu. Vì thế A1 **không tăng** tỉ lệ điều kiện, mà nhắm nhiễu vào từ nối điều kiện và vào vế thứ 2–3.
- Phần **thật sự mới** của A1: (1) nhiễu từ B0; (2) app lạ — `synth/newapps` (28 app, có catalog, bộ dựng `build.py`, `cross.py`) **chưa từng được dùng** trong data FC (meta `fmtA_48e` chỉ có `train_alias` và `train_iface`); (3) cặp mẫu hỏi/làm tương phản; (4) mẫu "tìm rỗng → tìm lại"; (5) mẫu tham số tuỳ chọn; (6) mẫu tính khoảng thời gian.
- **`synth/newapps` chỉ có mức chọn tool, không có tham số và kết quả tool** (11/11 19:30): mỗi dòng `{text, gold_tools, installed, kind, split}` — không có `arguments`, không có môi trường tool để chạy. Vì vậy chỉ dựng được **ví dụ bước Q1** (chọn tool / DỪNG) từ `train` (314 câu, 16 app, kèm 19 câu chéo app và 20 câu không cần tool); không dạy được tham số hay chuỗi nhiều bước cho app mới. Phần tham số/chuỗi vẫn phải nhờ đổi vỏ app thật (iface, có kết quả tool thật).
- **Thước đo app lạ M3b (thay cho kế hoạch A0 cũ):** thay vì đổi vỏ val_in rồi chạy trong `ToolEnv` (khó, cần môi trường cho tên tool mới), đo **độ đúng bước Q1 trên `val_newapps` (148 câu, 6 app) + `test_newapps` (138 câu, 6 app)** — 12 app không có trong train, có sẵn danh sách app cài. Cách đo giống `diag_public_q1.py`, không cần `ToolEnv`, ~$0.3 trên vLLM. Chỉ đo được chọn tool, không đo được tham số.
- Các app `test_like` trong `newapps` được thiết kế theo nhóm chức năng của catalog public test (chỉ dùng tên/nhóm app, không dùng nhãn hay câu của public).
- 3 và 4 và 6 cần sinh bằng luật từ hội thoại train (không có LLM ngoài): chỉ nên chiếm ~10–15% tổng để khỏi lệch phân bố.

## 1f. Đổi vỏ app: bỏ mục tăng iface 2% → 20% (11/11 19:15)

- Data hôm qua đã có **alias** (`synth/train_alias.jsonl`): đổi tên app, tên tool, tên tham số và mô tả sang tiếng Việt (vd `smart_home.get_status` → `nha_thong_minh.xem_thiet_bi`), 80 bản đổi, **1941/7901 mẫu (24.6%)**. Phần "tên lạ" đã được phủ nhiều.
- **iface** (4.9%) là biến đổi *cấu trúc* (gộp tool, đổi enum, thêm/bớt tham số tuỳ chọn) chỉ có 14 bản; tăng lên 20% nghĩa là lặp lại 14 kiểu đó, ít thông tin mới.
- Khoảng trống thật của app lạ trên public là **chức năng và luồng chưa từng có** (tìm sự kiện → đặt vé, đặt chỗ gửi xe, tìm công thức, tra cổ phiếu…), không phải đổi tên. Phần này được `newapps` Q1 (đã dựng) + gợi ý luồng I5 (C1) lo; không thể bù bằng đổi vỏ app thật vì app thật không có các chức năng đó.
- Điều chỉnh: **giữ nguyên iface và alias**, không làm bước "tăng đổi vỏ".

## 1g. Tiến độ A1 (11/11 19:20)

**Đã xong và kiểm chạy:**
- `agent/clef/build_newapps_q1.py`: ví dụ Q1 cho app lạ từ `synth/newapps` → `agent/clef/data/newapps_q1/` (train 885 ví dụ; val_newapps 137 + test_newapps 129 dùng đo M3b). Kiểm: 0 câu trùng văn bản train↔val/test, không app val/test trong prompt train.
- `agent/clef/asr_noise_b0.py` + cờ `--asr-b0 --asr-b0-manifest --asr-q` trong `build_data.py`: nhiễu ASR thật từ B0, pha loãng q=0.45 (hiệu chỉnh M2), tra theo văn bản câu chuẩn, mỗi lần pha loãng lại, ép có ≥ 1 lỗi, mẫu ≥ 3 call ưu tiên lỗi ở nửa sau câu, câu không có bản nghe thật thì rơi về bảng cụm cũ. Đã chạy thử với B0 giả: 117 câu dùng lỗi B0, 28 rơi về bảng cũ, 0 lỗi.
- `agent/clef/make_v3.py`: chọn tập con có coverage (`make_subset`, ~11.5k ví dụ) + thêm 885 ví dụ app lạ + kiểm rò (val_in, val/test newapps) + báo cáo thành phần. Đã chạy thử trên bộ cũ: 147/147 tool, không rò, thành phần câu điều kiện 34.9%, tự sửa 8.9%, nhiều call 41.6%, hỏi lại 13.5%.

**Còn lại (khi B0 xong ~21:30):** chép `b0.base_medium.asr.json` → dựng bản đầy đủ có nhiễu B0 (`build_data.py`, cờ như `fmtA_48e` + `--asr-p 0.6 --asr-hist-p 0.15 --asr-b0 ...`) → `make_v3.py` → `agent/clef/data/fmtA_v3`.

**Không làm trong A1 (ghi nhận để khỏi hiểu nhầm):**
- *Cặp mẫu hỏi/làm tương phản, cặp tham số tuỳ chọn, mẫu "tìm rỗng → tìm lại → làm tiếp"*: sinh bằng luật cần sửa câu nói và dựng lại chuỗi call; nhãn rất dễ sai (ask luôn kèm các bước đọc trước đó, tham số phụ thuộc kết quả tool) và không có LLM ngoài để kiểm. Nhãn sai gây hại hơn thiếu mẫu → **không làm**; để A4 quyết định sau khi đọc lỗi thật của FC v3 trên M2/M3b (có thể dùng chính lỗi đó để chọn thêm mẫu train sẵn có).
- *Tăng iface 2% → 20%*: bỏ (§1f).
- Thành phần câu điều kiện / tự sửa / nhiều call giữ nguyên vì đã đủ (§1e).

## 1h. Kết quả C1 + M3b (11/11 19:45)

Code: `clef_agent.py --n8-tau/--i5`, `eval_newapps_q1.py`. Đo trên server `fmtA_48e`:

| Phép đo | Kết quả |
|---|---|
| **M3b mốc** — bước Q1 trên 12 app lạ (`newapps`): `val_newapps` / `test_newapps` | catalog hai tầng **56.9% / 55.8%**; catalog đầy đủ 55.5% / 55.8% |
| N8 offline trên M3b (τ 0.5–1.0, cùng một lượt gọi) | val 56.9 (không đổi), test 55.8 → 56.6 (chỉ 1 câu) |
| Q1 kém chắc (conf < 0.7) trên M3b | val 18 câu (13 sai), test 23 câu (16 sai): sai nhiều nhưng hỏi lại với catalog đầy đủ **không sửa được** |
| N8 + I5 trên M1 (val_in sạch) | 76.5 → 76.5 (0 mẫu khác); N8 thăm dò 15 lần, đổi 0 lần; I5 0 lần |
| N8 + I5 trên M2 | 63.5 → 62.8 (−0.6, 1 mẫu kém hơn); N8 thăm dò 29, đổi 4; I5 0 lần |
| Public (ASR v1 large + N1 + N8 + I5), 634 mẫu | N8 thăm dò 155 lần ở 125 mẫu, **đổi 4 lần**; I5 **0 lần**; 7 mẫu khác bản 62.76, trong đó chỉ 1 do N8 (6 mẫu còn lại là dao động của server) |

- **Kết luận:** N8 và I5 không đáng đưa vào bản nộp: không tăng điểm trên M1/M2/M3b, thêm ~25% request. Lỗi bước Q1 ở app lạ **không phải do nén catalog** → chỉ sửa bằng train. Mã giữ lại, mặc định tắt. Không nộp `public_fmtA_48e_e46f_c1.jsonl`.
- I5 không kích hoạt lần nào: sau tool đọc, Q1 thu hẹp luôn nghiêng về DỪNG/hỏi lại, xác suất tool ghi không đạt 0.5.
- **M3b mốc 56%** thấp hơn nhiều so với 84.5% của M5 vì `newapps` cố ý có app dễ nhầm (hard negative) với app gốc. Đây là thước đo khó **chưa dùng để train** (12 app giữ riêng) → FC v3 có 885 ví dụ Q1 app lạ (16 app train khác) nên **M3b là cổng kiểm khái quát hoá sạch**: mục tiêu ≥ 65% trên val và test.
- I4 (bỏ phiếu n-best) chưa làm; hoãn tới khi có n-best và có cách đo.

## 1i. Trạng thái các việc chạy nền (11/11 19:47)

| Việc | Nơi chạy | Trạng thái | Dự kiến xong | Bước kế |
|---|---|---|---|---|
| B0 — PhoWhisper gốc nghe 9 854 đoạn train + TTS | Kaggle duongantbat | đang chạy (lần 2; lần 1 lỗi thiếu `/kaggle/temp`) | ~21:30 | dựng `fmtA_v3_full` → `make_v3.py` → **A1 ✅** |
| Qwen3-ASR-0.6B fine-tune 6 epoch (lr 5e-5, sửa lỗi cộng dồn loss) | Kaggle huthnhlng | đang chạy | ~21:00 | chấm 405 đoạn bằng `metrics asr-detail`; dừng hướng nếu WER > 8% |
| L2 — `e46f` + 2 epoch data v2.1 | Kaggle andngdngtrnhhoian | đang chạy | ~23:00–23:30 | B3: chấm so với `e46f` (WER câu cuối + sai thực thể 183) |
| D1 — chạy thử Docker 20 mẫu | Modal A10 | đang dựng image | ~20:30 | đo p50/p95 ASR+FC, bộ nhớ GPU |
| C1 | Modal | ✗ xong (§1h) | – | – |

Điểm nộp: 59 → 62.04 (ASR v1 large) → **62.76** (+N1). Chưa có lượt nộp mới.

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
| A0 | A | **Thước đo app lạ M3b:** độ đúng bước Q1 trên `val_newapps` + `test_newapps` (12 app lạ, 286 câu; xem §1e); đo mốc `fmtA_48e` (câu sạch) | – | B0, A1 | ~$0.5 | chạy được trong `ToolEnv` đổi vỏ; mốc ghi vào plan |
| A0b ✅ | A | Đọc 41 mẫu **sạch** bị sai của `fmtA_48e` → phân loại nguyên nhân (xong 11/11 tối, kết quả ở §1d) | – | B0 | $0 | bảng nguyên nhân → đưa vào A1/A4 |
| A1 🔶 **đang làm 11/11 19:20 — pipeline xong, chờ B0 để ghép nhiễu** (chi tiết ở §1g) | A | Data FC v3 vòng 1: nhiễu ASR từ B0 pha loãng kiểu M2, 60% mẫu, nhiễu ở câu cuối; ≥ 50% mẫu có nhiễu là mẫu nhiều call, nhiễu rơi vào vế chứa hành động thứ 2–3; nhãn nội dung/tên giữ bản sạch; nhiễu ở ngày giờ và số; giữ nguyên tỉ lệ câu điều kiện (đã ~35% theo định nghĩa rộng) và nhiều call (3+ call đã 40%) — xem §1e; nhiễu nhắm vào từ nối điều kiện (nếu/thì/không thì/trừ); mẫu "tìm rỗng → tìm lại → làm tiếp" và "tìm thấy → đặt/mua"; **app lạ:** ~~mẫu đổi vỏ 2% → ~20%~~ (bỏ, xem §1f), **ví dụ Q1 từ `synth/newapps` (đã dựng: `agent/clef/data/newapps_q1/`, 885 ví dụ train)**, **F3b** cặp app trùng chức năng (cùng câu, hai bộ app cài khác nhau → hai tool khác nhau), **28 app tổng hợp** `synth/newapps` (giữ riêng 4–5 app làm val) | B0 (phần nhiễu) | B1, C1 | $0 | không rò val/M3b; đủ 147 tool; đúng phân bố |
| C1 ✗ **đã đo 11/11 19:45 — N8 và I5 KHÔNG có tác dụng đo được (xem §1h); I4 chưa làm (không có n-best, không đo được offline)** | C | **N8** (conf < 0.7 → hỏi lại với catalog đầy đủ) + **I4** (2–3 giả thuyết ASR câu cuối, bỏ phiếu ở bước chọn tool) + **I5** (gợi ý luồng từ catalog: tool ghi cần id của tool đọc nào → nhắc bước kế sau khi tìm thấy) | – | A1, B1 | ~$1.5 | từng thứ: M2 tăng, M1 tụt ≤ 1, M3b không tụt (so cặp) |
| B2 | B | Text TTS miền mới (2–3k câu, trọng tâm **tên chưa gặp**: sự kiện, sản phẩm, món ăn, tựa; tỉnh thành, ga, bến, sân bay; số và mã chuyến — nguồn chung, không lấy từ public) → sinh audio; giữ riêng 300 câu + 1–2 giọng làm val miền mới | – | A2 | TTS: nhỏ | nghe thử 20 câu |
| A2 | A | Train FC v3 vòng 1: `--init-adapter` từ `fmtA_48e`, ~12k ví dụ, LR 5e-5, 1 epoch | A1 | B2, B1 (đêm) | ~$10–12 | – |
| B3 | B | Chấm L2 trên 405 đoạn so với `e46f` | B1 | A2 | $0 | thắng WER câu cuối **và** tỉ lệ sai thực thể có nhãn (183 thực thể, `metrics asr-detail`) |
| S-A | – | Nếu B3 qua: ASR public bằng L2 → FC `fmtA_48e` + N1 (+C1 nếu qua) → **nộp A** | B3 | A2 | ~$1.5 | |
| B4 | B | Train ASR v3: từ bản large tốt nhất (L2 hoặc `e46f`) + data v2.1 + TTS miền mới, 2–3 epoch | B2, B3 | A2, A3, D1 | $0 (Kaggle, 8–12 h, 2 phiên nếu cần) | – |
| A3 | A | Gộp FC v3 vòng 1 → đo M1 + M2 + **M3b** + `metrics calls` | A2, A0 | B4, D1 | ~$2 | **M2 ≥ 68**, M1 ≥ 75.5, **M3b (§1h) ≥ 65% cả val lẫn test (mốc 56%)**, câu điều kiện tụt ≤ 15; trên M2: recall chọn tool ≥ 0.90, `fuzzy` ≥ 0.70, `datetime` ≥ 0.90, thiếu `sys.ask_missing_info` ≤ 4 (hiện 8) |
| S-B | – | Nếu A3 qua: public với ASR tốt nhất → **nộp B** | A3 | D1, C2 | ~$1 | |
| D1 🔶 **code viết xong 11/11 19:40 (`docker/`: Dockerfile, run.sh, asr_stage.py, build.sh, README; `clef_agent --time-budget`); Dockerfile qua `docker build --check`. **Đang chạy thử trên Modal A10 (19:45)**: Modal không có A30 (đã thử: "A30 is not a valid GPU type") → A10 24 GB, băng thông thấp hơn A30 nên số đo bảo thủ; `docker/modal_d1.py --limit 20` dựng image từ chính Dockerfile; dữ liệu public trên volume `zaic-d1-data`. Lần dựng đầu lỗi vì parser Dockerfile của Modal không nhận `ARG ASR_REV=` rỗng → ghim commit `85081b3` của e46f** | D | Image offline (PhoWhisper-large + FC gộp + bộ lọc + vLLM hoặc HF generate + `submit.py`); **seed cố định, greedy**; chạy 20 mẫu tắt mạng; đo thời gian/mẫu, RTF của ASR và bộ nhớ GPU | – | A3, B4, C2 | ~$1 | chạy offline được; ghi p50/p95, bộ nhớ |
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
