# Ghi chú phân tích kèm PLAN_72h_v3 (11/11)

Các mục đo đạc, chẩn đoán và quyết định chi tiết; plan chính ở `plans/PLAN_72h_v3.md`.

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

**Hoàn tất 21:25 11/11 (B0 xong 21:04):**
- B0: `agent/runs/asr/b0.base_medium.asr.json` — PhoWhisper-medium gốc (rev 55a7e3eb, có prompt, beam 5) nghe 9854 câu train + TTS, 0 lỗi, 7149 s trên T4. WER thô so với text chuẩn 20,4%, chỉ 20,5% câu nghe đúng hoàn toàn → đủ lỗi để pha loãng về mức M2 (q = 0,45).
- Dựng `fmtA_v3_full` (6 shard song song, cờ như `fmtA_48e` + `--asr-p 0.6 --asr-hist-p 0.15 --asr-b0 ... --asr-q 0.45 --asr-table-mix 0.3`): 9905 item → 24.654 ví dụ, bỏ 4 bước. Nhiễu: 3961 item sạch (40%), 4258 câu dùng lỗi B0 thật (505 câu ≥ 3 call có lỗi rơi nửa sau), 2070 câu không có bản nghe → bảng cụm cũ, 2180 câu trộn bảng theo `table_mix`, 371 câu thay nguyên bản ASR v2.
- `make_v3.py` → `agent/clef/data/fmtA_v3`: 11.504 ví dụ tập con (1417 item gốc, 4268 item kể cả alias/phonetic/history) + 885 ví dụ Q1 app lạ = **12.389 ví dụ**; 147/147 tool (mỗi tool ≥ 4 lần), 386/407 tên kể cả alias; **0 rò** val_in và val/test newapps; tier/full catalog 7318/5071; kết thúc call/ask/stop0 = 3483/343/442; thành phần: điều kiện 34,9%, tự sửa 8,9%, hỏi lại 13,5%, ≥ 3 call 41,6%; app trong public 74,1%.
- **Soát tay 30 mẫu nhiễu (21:30):** đa số giống lỗi nghe thật (âm gần, phương ngữ: "giùm/dùm", "Zalo/đa lô", "tiếng/kín"). Hai kiểu có hại: (1) **đổi giá trị số/giờ** ("bảy giờ tối" → "ba giờ tối", "4 chỗ" → "vốn chỗ") trong 67/821 câu nhiễu (8%) — nhãn vẫn giữ giá trị sạch nên model chỉ học đoán bừa; (2) lật động từ/phủ định ("Soạn" → "Xoá", "khỏi bật" → "coi bật") ~7% theo đếm thô. Sửa (1): `asr_noise_b0.py` thêm chốt chặn `_nums` (đọc số cả chữ lẫn số 0–99; bản nhiễu đổi giá trị số → thử lại, hết `max_tries` thì giữ câu sạch). **Dựng lại 21:39:** 828/1417 item gốc có nhiễu (58%), **0 câu đổi giá trị số**, WER câu nhiễu 11,0% (trước 11,7%), 1314 ứng viên bị loại; tập con vẫn 12.389 ví dụ, 147/147 tool, 0 rò, thành phần không đổi. (2) để lại: lỗi thật của ASR cũng lật từ như vậy và không có cách phân biệt tự động không dùng nhãn; theo dõi qua M2 sau khi train.
- Lưu ý: meta của bản ghép shard phải giữ `p_item/p_hist/q/table_mix` là hằng (script ghép cộng nhầm, đã sửa tay).

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

## 1i. Trạng thái các việc chạy nền (11/11 19:47, cập nhật 20:40)

| Việc | Nơi chạy | Trạng thái | Dự kiến xong | Bước kế |
|---|---|---|---|---|
| B0 — PhoWhisper gốc nghe 9 854 đoạn train + TTS | Kaggle duongantbat | đang chạy (lần 2; lần 1 lỗi thiếu `/kaggle/temp`) | ~21:30 | dựng `fmtA_v3_full` → `make_v3.py` → **A1 ✅** |
| Qwen3-ASR-0.6B fine-tune 6 epoch | Kaggle huthnhlng | ✗ xong 20:55: WER 16.3% (kém hơn 1 epoch) → **dừng hướng Qwen** (§1n) | – | – |
| L2 — `e46f` + 2 epoch data v2.1 | Kaggle andngdngtrnhhoian | đang chạy | ~23:00–23:30 | B3: chấm so với `e46f` (WER câu cuối + sai thực thể 183) |
| D1 — chạy thử Docker | Modal A10 | ✅ smoke 5 mẫu qua (§1k) | – | D2/D3 khi có model cuối |
| Bản model cắt `qwen35_4b_text` cho A2 | Modal (CPU) | ✅ 20:35 | – | A2 |
| C1 | Modal | ✗ xong (§1h) | – | – |

Điểm nộp: 59 → 62.04 (ASR v1 large) → **62.76** (+N1). Chưa có lượt nộp mới.

## 1j. Kiểm trước A2 (train FC v3) — không tốn GPU (11/11 20:30)

| Kiểm | Kết quả |
|---|---|
| `train.py encode()` đọc 885 ví dụ Q1 app lạ (`newapps_q1/train.jsonl`) | giữ 885/885, mỗi ví dụ đúng 1 nhãn chọn tool, 0 phần `lm` (đúng thiết kế) |
| Cùng hàm trên 300 ví dụ bộ cũ (đối chứng) | giữ 300/300 |
| `--init-adapter` với adapter `fisherman611/q-f48e-lora/final` (r 32, alpha 64): dựng model rỗng (meta) từ config `q-f48e`, LoRA như `train.py` | target trùng khớp 12/12; khoá 496/496, **0 thiếu, 0 thừa, 0 lệch shape**; `set_peft_model_state_dict`: 0 unexpected, 0 missing |
| Model gốc cho train trên Modal (`/hf/qwen35_4b_text` trong volume `hf-cache`) | **chưa có ở workspace duytungihb** (F2 đã train ở kaiserihb rồi myserver) → trước A2 phải dựng lại bản cắt (`strip_qwen35`) hoặc train tiếp trên model gộp `q-f48e` với LoRA mới |

## 1k. D1 — smoke Docker trên Modal (11/11 20:30)

Image dựng từ chính `docker/Dockerfile` (Modal `Image.from_dockerfile`), GPU A10 24 GB (Modal không có A30), 5 mẫu public đầu, đi hết 4 bước.

| Kiểm | Kết quả |
|---|---|
| rc / tổng thời gian | **0** / 260 s (gồm khởi động vLLM ~3.5 phút) |
| ASR trong container | 7/7 đoạn trùng y transcript e46f đã dùng cho bản nộp 62.76 |
| Agent + ToolEnv | 5/5 mẫu, 0 lỗi, 0 hết giờ |
| `submit.py` | 5 dòng, 0 thiếu/thừa/trùng/hash rỗng, 10 call |
| So với bản nộp 62.76 | **5/5 mẫu cùng state_hash** |
| Bộ nhớ GPU đỉnh | 19.6 GB / 23 GB (vLLM `--gpu-memory-utilization 0.85`) |
| Thời gian (tham khảo, chưa tối ưu) | ASR+FC p50 12.7 s, max 16.0 s |

Sửa trong lúc chạy thử: Modal không nhận `ARG` rỗng → ghim commit `85081b3` của e46f; `python3` trần trỏ vào Python của Modal → Dockerfile ghi `/etc/fc_python`, `run.sh` dùng nó; smoke chạy cả bước nộp (`submit.py --limit`), bản đầy đủ thêm `zaic_tools --validate`.
Chưa làm: chạy đủ 634 mẫu (kiểm tất định, D3); hướng dẫn đóng gói chính thức của BTC chưa có.

## 1l. Thiết kế C3 (N4) — sửa địa danh ở mức tham số (11/11 20:50, chưa code)

**Căn cứ (chỉ chẩn đoán, không tune):** trong các lần tìm kiếm trả rỗng của bản nộp 62.76 (chạy lại call trong ToolEnv), loại có tham số nơi chốn gồm:

| Mẫu | Call rỗng | Nguyên nhân thấy được | C3 sửa được? |
|---|---|---|---|
| 0238 | `train_bus_tickets.search` → "Nạng Sơn" | ASR l/n | ✔ → Lạng Sơn |
| 0252 | `flights.search` → "QHO" | model bịa mã IATA (Quy Nhơn = UIH) | ✔ → UIH / "Quy Nhơn" |
| 0440 | `train_bus_tickets.search` Hà Nội → Hà Nội | điểm đi = điểm đến | có thể: điểm đi lấy thành phố trong hồ sơ người dùng |
| 0199 | `train_bus_tickets.search` → "SJC" | ASR nát cả câu | ✘ |
| 0187, 0287, 0331 | điểm hợp lệ (HAN→DAD, Hà Nội→Huế) | rỗng vì lý do khác (ngày, mã chuyến) | ✘ |
| 0266, 0332, 0646, 0649 | `public_transit.next_departures` stop | tên trạm không có danh sách công khai, không có tool liệt kê trạm | ✘ |

→ phần sửa được trên public chỉ **~2–3 mẫu** → kỳ vọng **+0.3–0.5**, giữ ưu tiên thấp.

**Thiết kế:**
1. **Khi nào chạy (an toàn như N1):** chỉ sau khi tool ĐỌC có tham số nơi chốn trả danh sách rỗng: `flights.search`, `train_bus_tickets.search`, `public_transit.find_routes`, `parking.find`, `maps.search_place`, `bike_rental.find_bikes`, `weather.*`. Không bao giờ sửa tool GHI (chúng dùng id lấy từ kết quả tìm kiếm). Riêng mã IATA: kiểm ngay trước call (mã 3 chữ hoa không có trong danh sách → đổi), vì sai chắc chắn.
2. **Danh bạ ứng viên (không lấy từ public test):**
   - tĩnh, gõ tay từ nguồn công khai: tỉnh/thành (tên cũ 63 + tên sau sáp nhập + bí danh: TP.HCM/Sài Gòn/HCM, Huế/Thừa Thiên Huế…), ~22 sân bay VN + mã IATA (HAN, SGN, DAD, CXR, PQC, HPH, VII, HUI, UIH, VCA, VCL, DLI, BMV, PXU, THD, VDO, VDH, TBB, CAH, VCS, DIN, VKG…) và vài sân bay quốc tế phổ biến, các ga tàu chính (Hà Nội, Sài Gòn, Đà Nẵng, Huế, Vinh, Nha Trang, Lào Cai, Đồng Đăng/Lạng Sơn, Hải Phòng, Diêu Trì/Quy Nhơn, Tháp Chàm, Phan Thiết, Thanh Hoá, Ninh Bình, Nam Định, Đông Hà, Đồng Hới, Tuy Hoà, Quảng Ngãi). File: `agent/data/places_vn.json`.
   - động theo mẫu: thành phố/địa chỉ trong `user_profile`, tên nơi chốn trong lịch sử và trong kết quả tool trước đó của cùng mẫu.
3. **So khớp theo âm:** bỏ dấu, tách âm tiết; khoảng cách sửa chữ trên âm tiết với nhóm nhầm lẫn hay gặp của ASR tiếng Việt (l/n, s/x, tr/ch, d/gi/r, v/d; vần n/ng, c/t). Nhận khi điểm ≥ 0.75 và hơn ứng viên thứ hai ≥ 0.15; không thay bằng giá trị đã thử.
4. **Luật điểm đi = điểm đến:** nếu `origin == destination` và câu không nhắc điểm đi → `origin` = thành phố trong `user_profile` (nếu khác).

**Đã code (11/11 21:50):** `agent/geo_fix.py` (danh bạ `agent/data/places_vn.json`: 30 sân bay + ~110 tỉnh/thành/bí danh + 45 ga; so khớp theo âm với bảng quy đổi l/n, s/x, tr/ch, d/gi/r, vần ng→n, t→c; ngưỡng 0,75 / cách biệt 0,15; mã IATA bịa → tìm tên nơi trong câu cuối, ngưỡng 0,85), cờ `clef_agent.py --c3` (kiểm IATA trước call; sau tool đọc nơi chốn trả rỗng — kể cả sau N1 — sửa rồi gọi lại một lần, mỗi tool một lần, không lặp bộ tham số đã thử; trường `n_c3`), `agent/c3_offline.py` (chạy lại call đã ghi trong ToolEnv chính thức, không dùng nhãn).
**Kiểm offline trên log bản 62.76** (`agent/runs/c3/c3_offline.public_n1.json`): 85 call có tham số nơi chốn, 7 trả rỗng, C3 đề xuất 2 + 1 kiểm IATA:
- 0238 "Nạng Sơn" → "Lạng Sơn": tìm lại **có kết quả** ✔ (đúng như dự đoán).
- 0252 "QHO" → "Quy Nhơn" (từ câu "đi Quy Nhơn"): call hợp lệ nhưng vẫn rỗng — người dùng nói mã chuyến RV253 + ngày, có lẽ ngày không có chuyến; chỉ sửa được tham số, không sửa được việc không có chuyến.
- 0440 Hà Nội → Hà Nội: không sửa vì thành phố hồ sơ cũng là Hà Nội (câu "vé tàu tối nay với vé chủ nhật" không nhắc điểm đi); đúng theo luật, nhận là không cứu được.
- 0187/0287/0331 (điểm hợp lệ), 0199 ("SJC" nát): không sửa, đúng ý thiết kế. Không có sửa sai (false positive) nào trên 85 call.
→ Kỳ vọng vẫn **+0,15–0,3** (1–2 mẫu/634). **Chưa đo M1/M2 online** (cần LLM ~$1): gộp vào lần đo G1 để tiết kiệm — chạy `--retry-empty --c3` cùng lúc, so với bản 62.76 theo state_hash.
5. **Hành động:** gọi lại tool đọc đúng **một lần** với giá trị đã sửa (giữ nguyên tham số khác), đưa cả hai call + kết quả vào hội thoại để model đi tiếp (như N1). Ghi `n_place_fix`. Cờ `--place-fix`, mặc định tắt.
6. **Đo:** (a) mô phỏng offline mở rộng `sim_retry_empty.py`: bao nhiêu call rỗng trên public thành có kết quả (chẩn đoán); (b) M1/M2 với cờ bật để chặn hại (dự kiến gần 0 lần kích hoạt); (c) chạy public cùng N1, so với 62.76, đọc tay các mẫu đổi. ~$1.
7. **Công:** ~1.5–2 giờ (danh bạ + so khớp + luật + mô phỏng). Phần trạm xe buýt bỏ qua (không có danh sách hợp lệ).

## 1m. Mẫu "không gọi tool" ở public — chẩn đoán + giải pháp (11/11 21:00)

**Tỉ lệ giống train:** lượt history assistant không gọi tool: public 14.9% (193/1299) vs train 14.6%; câu cuối: agent bản 62.76 không gọi tool 44/634 (6.9%) vs train 7.2% (111/1545). Trong train, "không gọi tool" = app cần dùng chưa cài, hoặc người dùng tự huỷ; **không có câu vô nghĩa**; "hỏi lại" (217 mẫu) luôn là thiếu tham số cụ thể.

**44 mẫu agent không gọi tool, chia theo dấu hiệu transcript (chỉ chẩn đoán, KHÔNG tune theo public):**

| Nhóm | Số | Ví dụ | Đánh giá |
|---|---|---|---|
| Cảm ơn / tán gẫu / "khỏi làm gì thêm" | 3 | 0209, 0263, 0341 | đúng |
| Tự huỷ | 19 | 0057 "à mà thôi, cô giáo bảo mưa vẫn tổ chức" | đa số đúng; riêng **huỷ việc này + yêu cầu việc khác** (0210 "khỏi đặt nữa, nhắc anh tạt qua chợ") là sót |
| Câu điều kiện, lượt sau của hội thoại đã có thời tiết | 2 | 0045, 0098 | **có thể đúng** (điều kiện đã biết từ lượt trước) — không xử lý |
| **Chọn từ kết quả lượt trước ở APP MỚI** | 5 | 0191 "lấy liên hoan phim ngắn đi" (lượt trước `event_tickets.search`), 0245 "lấy tàu NA3", 0262 "đặt đủ số vé", 0635 (`pantry`), 0654 (`stocks`) | **sót**: P(DỪNG) 0.47–0.87, tool ghi cần làm (`event_tickets.book`, `train_bus_tickets.book`…) chỉ 0.02–0.15 |
| ASR nghe nát | 2+ | 0424 "bật ca chống trộm", 0229 | sót do ASR |
| Khác / chưa rõ | ~13 | | |

**Vì sao chỉ hỏng ở app mới:** kiểu "chọn từ kết quả lượt trước → gọi tool ghi ngay" có **181/1417 mẫu train (12.8%)** và 25/128 val_in, và trên val_in sạch model **không** dừng sai ở bước đầu lần nào (M2: 3 lần, không phải kiểu này). Model đã học kiểu này với app cũ (`bike_rental.reserve`, `flights.book`…) nhưng không chuyển được sang tool ghi của app chưa gặp. I5 (C1) không kích hoạt lần nào vì chỉ xét tool đọc **trong cùng lượt**, còn ở đây tool đọc nằm ở **lượt trước**.

**Giải pháp (xếp theo lợi ích/công):**

| # | Giải pháp | Cách làm | Đo bằng | Kỳ vọng | Công |
|---|---|---|---|---|---|
| G1 | **I5b — nối luồng qua lượt** (luật lúc infer) | Ở bước đầu của lượt, model chọn DỪNG; lượt assistant gần nhất có tool đọc R trả danh sách không rỗng; tool ghi W cần id từ R (x-ref, `build_deps`) chưa được gọi sau R → hỏi lại Q1 thu hẹp {W, hỏi lại, DỪNG}; nếu P(W) ≥ τ thì làm tiếp với W. τ chọn trên val_in (25 mẫu kiểu này + mẫu DỪNG đúng), không chọn trên public | M1/M2 (không được làm sai mẫu đang đúng; đếm số lần kích hoạt ở mẫu DỪNG đúng); public: đếm lần kích hoạt (chẩn đoán) rồi một lượt nộp | +0.3–0.8 (≤ 5 mẫu public rõ ràng) | ~1 h + ~$1 |
| G2 | **Data FC (A4): kiểu "chọn từ lượt trước" ở app đổi tên** | Lấy 181 mẫu kiểu này, tạo bản alias/iface (đổi tên app/tool/tham số) và nhân ×2 trong tập con, để model học "lấy X đi" → tool ghi của app lạ | M3b chỉ đo bước Q1 câu đơn → cần thêm thước đo nhỏ: 25 mẫu val_in kiểu này đổi vỏ | khái quát app lạ | trong A4 |
| G3 | **Data FC (A4): "huỷ việc này + yêu cầu việc khác"** | chọn thêm mẫu train có cụm tự huỷ mà nhãn vẫn có call (đếm trước khi làm) | M1/M2 | nhỏ | trong A4 |
| G4 | ASR nghe nát | ASR v3 (B4) + nhiễu FC v3 (A1) | 405 đoạn giữ riêng | – | đã có trong plan |

Không làm: luật ép "không được DỪNG ở bước đầu" chung chung (trên val, P(DỪNG) của mẫu sai 0.83 trùng vùng mẫu đúng 0.61–0.94 → không tách được).

## 1n. Qwen3-ASR-0.6B 6 epoch — dừng hướng này (11/11 20:55)

| Model (405 đoạn giữ riêng) | WER | WER câu cuối | Sai thực thể (183) | Tên người | Số |
|---|---|---|---|---|---|
| v1 large `e46f` (đang dùng) | **6.23%** | **7.15%** | **9.3%** | 0.0% | 10.0% |
| v2 medium | 6.85% | 7.19% | 13.1% | 6.4% | 23.3% |
| Qwen 0.6B, 1 epoch, có ngữ cảnh | 13.17% | 14.56% | 27.3% | 19.1% | 46.7% |
| Qwen 0.6B, **6 epoch**, có ngữ cảnh | 16.27% | 17.34% | 25.1% | 21.3% | 23.3% |
| Qwen 0.6B, 6 epoch, không ngữ cảnh | 17.81% | 19.69% | 27.3% | 27.7% | 20.0% |

- 6 epoch **kém hơn** 1 epoch về WER (16.3 vs 13.2): loss train xuống 0.002 (học thuộc ~10.8k đoạn), không khái quát. Lỗi số giảm (46.7 → 23.3%) nhưng mọi chỉ số khác vẫn kém xa `e46f`.
- **Kết luận: dừng hướng Qwen3-ASR**, giữ PhoWhisper-large. Transcript: `agent/runs/asr/heldout.q3asr06ft_{e1,e6}_{ctx,noctx}.asr.json`; quota Kaggle huthnhlng đã dùng thêm ~4.5 h.



## 1o. Kiểm toán nhãn `fmtA_v3` (11/11 22:30) — sửa 3 lỗi, mốc M3b đổi

Công cụ (chạy lại được): `agent/clef/audit_v3.py` (cấu trúc + khớp gold), `agent/clef/audit_gold.py` (gold chạy lại trong ToolEnv chính thức), kết quả ở `agent/runs/audit/`.

**Đã kiểm và ĐẠT trên bản cuối (12.241 ví dụ):**
- Mọi ví dụ khớp gold nguồn: tên tool, tham số, khối `<tool_call>`, chữ cái đáp án Q1 trỏ đúng dòng tool, span `choice`/`lm` đúng vị trí, Q2 có/không và Q2 chọn id khớp tham số, mọi id trong nhãn đều có trong prompt, bước liền mạch. 0 lỗi.
- Gold của 1416 mẫu train gốc chạy lại trong ToolEnv: 3266/3266 call hợp lệ, kết quả trùng bản lưu. (Không so được trạng thái cuối: gold lưu cả state, ToolEnv chỉ cho hash.)
- Bản tổng hợp: phonetic giữ nguyên nhãn mẫu gốc; alias/iface chỉ khác theo thiết kế (đổi tên tool/tham số/enum). 4 id trùng trong `synth/train_iface.jsonl` (2 biến thể chung id) — không ảnh hưởng nhãn.

**Lỗi 1 — gold gốc hỏng (1 mẫu):** `train_1696` gold gửi tin tới `chat_6142` không tồn tại (kết quả `done:false`). `build_data.py` giờ bỏ mọi mẫu có call gold không thành công (kèm bản alias/phonetic/history).

**Lỗi 2 — nhiễu làm nhãn sai so với câu đầu vào.** Nhãn giữ theo câu sạch, nhưng nhiễu của medium gốc tạo ra câu đọc thành lệnh khác. Đo trên 2518 câu cuối bị nhiễu (bản trước), so với ASR thật lúc thi (e46f, 71 câu cuối có lỗi trong 123 held-out):

| | e46f thật | v3 trước | v3 sau |
|---|---|---|---|
| Tỉ lệ câu cuối có lỗi | 58% | 59% | 58% |
| Đổi giá trị số | 7,0% câu lỗi | 0 (đã chặn) | 0 |
| Đổi từ chỉ thời gian | 14,1% | 19,1% (168 item có nhãn ngày giờ) | 0 |
| Đổi phủ định / tự sửa | 4,2% | 2,9% (72 item) | 0 |
| WER câu lỗi p50 / p95 / max | 0,08 / 0,38 / 0,45 | 0,11 / 0,38 / **1,57** | 0,10 / 0,33 / 0,44 |
| Thực thể nhãn mất nguyên văn | 9,3% | 18,8% | 13,6% (chỉ còn loại gần âm) |

Ví dụ nhãn sai đã loại: "sáng mai" → "sáng này" (nhãn 06:00 ngày mai); "à thôi **khỏi** mở" → "à thôi **phải** mở" (nhãn không mở); "Phát bài Bước qua nhau đi" → "Khách bài bớt quà nhào đề" (nhãn `music.play trk_03`); "tưới cây" → "tới kênh" (nhãn content "tưới cây").
Sửa: `asr_noise_b0.noise_reject` — bản nhiễu bị loại khi đổi số, đổi từ thời gian, đổi phủ định/tự sửa, mất/thêm "nếu"/"trừ", WER câu > 0,45, hoặc một thực thể của nhãn không còn gần âm (độ giống < 0,7; thực thể ≤ 4 ký tự phải còn nguyên văn). Loại thì thử lại 10 lần, hết thì giữ câu sạch (224 câu). Lý do chặn cả những lỗi ASR thật cũng mắc (số, thời gian): các ca đó không model nào cứu được, mà dạy trên chúng chỉ làm model bớt tin câu nói rõ ràng. Từ nối "thì/còn/hoặc" vẫn cho nhiễu (5% câu lỗi, gần mức thật, nghĩa giữ nguyên).

**Lỗi 3 — nhãn Q1 app lạ sai quy ước, và thước đo M3b sai theo.** 30 tool của `synth/newapps` có tham số bắt buộc kiểu id (`pet_id`, `book_id`, `device_id`…). Theo data chính thức, bước đầu phải là tool đọc để lấy id; nhãn Q1 lại là gọi thẳng tool ghi. Trên các câu này `fmtA_48e` đúng 0/90, phần còn lại đúng 82–89%:

| M3b | Cũ (mọi câu) | Câu gold cần id | **M3b sạch (mốc mới)** |
|---|---|---|---|
| val_newapps | 78/137 = 56,9% | 0/42 | **78/95 = 82,1%** |
| test_newapps | 72/129 = 55,8% | 0/48 | **72/81 = 88,9%** |

→ "Khoảng trống app lạ" phần lớn là lỗi thước đo; model hiện tại đã chọn tool cho app lạ khá tốt. 147/885 ví dụ train loại này sẽ dạy model bỏ bước tra id → đã bỏ (còn 738). `build_newapps_q1.py` giờ tự bỏ các câu này ở mọi split; ba file `agent/clef/data/newapps_q1/*.jsonl` đã lọc (train 738, val 95, test 81). Kết luận §1h "C1 không có tác dụng trên M3b" vẫn đúng nhưng lý do đổi: phần sai còn lại nhỏ.

**Bản cuối `fmtA_v3`:** 11.503 + 738 = **12.241 ví dụ**, 147/147 tool, 0 rò; 58% câu cuối có nhiễu; thành phần: điều kiện 35,0%, tự sửa 8,9%, hỏi lại 13,6%, ≥ 3 call 41,5%.
**Còn lại, không tự động kiểm được:** (a) gold chính thức có đúng ý người nói không — chỉ kiểm được tính nhất quán với ToolEnv; (b) câu nhiễu mà một từ trong nội dung tin nhắn đổi ("tài liệu em gửi rồi" → "em nữa rồi") — nhãn giữ bản sạch, giống tình huống thi.


## 1p. G1 (I5b) + C3 đo online (11/11 22:45) — G1 không tác dụng, C3 giữ, lộ ra chuyện không tất định

Code: `clef_agent.py --i5b τ` (`prev_turn_reads`: tool đọc trả danh sách ở lượt assistant gần nhất; tool ghi cần id từ nó chưa gọi; model DỪNG ngay bước đầu → Q1 thu hẹp), `agent/analyze_g1.py`. Chạy `--retry-empty --c3 --i5b`.

**Val (τ thu thập 0,3):** M1 76,43 (tham chiếu không N1 76,51; chênh do N1 ở `train_1528`), M2 63,39 (= bản N1). I5b hỏi lại 3 mẫu, cả 3 là mẫu DỪNG đúng, P(tool ghi) 0,11 / 0,01 / 0,01 → không kích hoạt. C3 không kích hoạt lần nào trên val. → an toàn, chốt τ = 0,5.

**Public (τ = 0,5, chỉ chẩn đoán):** I5b hỏi lại 16 mẫu, **kích hoạt 0**. Ba mẫu sót đã chẩn đoán có P(tool ghi) ở Q1 thu hẹp chỉ 0,13 (0262), 0,05 (0191), 0,03 (0245) — model chắc chắn DỪNG ngay cả khi chỉ còn {tool ghi, hỏi lại, DỪNG}; không ngưỡng nào tách được khỏi mẫu DỪNG đúng trên val (0,11). **→ G1 ✗: luật lúc infer không cứu được, phải sửa bằng data (G2 trong A4: mẫu "chọn từ kết quả lượt trước" ở app đổi tên).**
**C3 trên public:** kích hoạt 2. 0238: "Nạng Sơn" → tìm lại "Lạng Sơn" có kết quả → model đặt vé tiếp (`train_bus_tickets.book`) — đúng luồng. 0252: mã bịa "QHO" → "Quy Nhơn", vẫn rỗng. → **giữ `--c3`** (an toàn trên val, +1 mẫu rõ trên public ≈ +0,15).

**Không tất định giữa hai lần chạy (quan trọng cho D3):** ngoài 2 mẫu C3 và 1 mẫu chỉ đổi thứ tự, có **13/634 mẫu khác bản 62.76 dù không luật nào kích hoạt** — khác ở phần chữ tự sinh (nội dung ghi chú/tin nhắn), mã cổ phiếu (T29 ↔ T22), có mẫu khác cả số call (0530, 0390). Cùng model, cùng seed, cùng transcript; khác là lần deploy vLLM và cách gom batch (48 worker). → **[SỬA 12/11 00:20: đếm theo state_hash thì hai lần chạy e46f chỉ lệch 2/634 mẫu; 13 mẫu kia chỉ khác chữ tự sinh / thứ tự call → nhiễu giữa hai lần chạy ≈ 0, KHÔNG phải ±0,5]**; Docker chấm thi phải chạy tuần tự hoặc cố định batch để tái lập (đưa vào D2a/D3: chạy 2 lần so state_hash).


## 1q. B3 — L2 (PhoWhisper-large từ e46f + data v2.1, 2 epoch, Kaggle 6,5 h) so với e46f (11/11 23:10)

Chấm trên 405 câu held-out bằng transcript trong `eval.json` của kernel (không cần Modal): `agent/runs/asr/heldout.L2.asr.json`, `agent/runs/metrics/asr_detail.L2.json`.

| | e46f (đang dùng) | L2 |
|---|---|---|
| WER tất cả (405) | 6,23% | 6,22% |
| WER câu cuối (123) | 7,15% | **6,65%** |
| WER lịch sử (282) | **5,40%** | 5,83% |
| WER câu có từ nước ngoài (117) | 5,99% | 6,14% |
| Câu cuối đúng hoàn toàn | 42,3% | 44,7% |
| Thực thể nhãn sai (183) | 17 (9,3%) | 18 (9,8%) |
| Tên riêng sai (289, thước đo của train) | – | 20 (6,9%); không prompt 26 (9,0%) |
| Câu đổi giá trị số | 9 (câu cuối 5) | 10 (câu cuối 6) |

Lỗi "number" của L2 trong `asr-detail` cao (8/30 so với 3/30) là do L2 hay viết số bằng chữ ("sáu rưỡi" thay "6 rưỡi"), không phải sai giá trị (đếm theo giá trị: 10 so với 9 câu).
**Kết luận:** L2 ≈ e46f. Hơn 0,5 điểm WER ở câu cuối (≈ 6 từ trên 123 câu), kém 0,4 ở lịch sử, thực thể ngang nhau — tất cả trong mức nhiễu. Data v2.1 + inter-CTC + KWS + TTS không cải thiện rõ khi train tiếp từ e46f. **Giữ e46f; không tốn lượt nộp cho L2** (chênh lệch kỳ vọng nhỏ hơn sai số ±0,5 giữa hai lần chạy, §1p). Adapter L2 nằm ở output của kernel `andngdngtrnhhoian/zaic-asr-l2`, chưa đưa lên HF.

## 1r. Bản nộp L2 + N1 + C3 (11/11 23:53)

`result/submission_fmtA_48e_L2_n1c3.jsonl`: ASR L2 (prompt v2, `asr_infer_v2` trên Modal, model gộp ở volume `zaic-out/L2_v21L_e46f`, chưa lên HF) → `pack_b` → FC `fmtA_48e` + `--retry-empty --c3`. 634 dòng, 0 lỗi schema, 0 lỗi agent, 1431 call, 43 mẫu không gọi tool.
Bước chặn: L2 trên Modal (bf16) ra WER held-out 6,29% / câu cuối 6,72% (Kaggle fp16: 6,22 / 6,65) → đường chạy đúng.
So với bản 62.76: 368/634 câu cuối có transcript khác e46f (bỏ dấu câu và hoa thường), 88/634 mẫu khác state (≈13 do nhiễu giữa hai lần chạy, 2 do C3). Chờ điểm public: thấp hơn 62.76 → giữ e46f; cao hơn ≥ 1 → chuyển L2 (Docker cần prompt v2 + đưa L2 lên HF qua Modal/myserver).
**Kết quả public (12/11 00:00): 62.60** (bản e46f + N1: 62.76). Chênh −0,16 là thật (nhiễu giữa hai lần chạy ≈ 0, xem §1s): L2 đổi state ở 83 mẫu, được và mất gần bù nhau. → **L2 không hơn e46f, giữ e46f** cho Docker và các lần nộp sau; dừng hướng train tiếp ASR large với data v2.1. Khớp với B3 (§1q).


## 1s. So bản 62.76 (e46f + N1) với 62.60 (L2 + N1 + C3) — vấn đề nằm ở đâu (12/11 00:20)

Chỉ chẩn đoán, KHÔNG có nhãn public; **không** được ghép bản nộp bằng cách chọn L2 hay e46f theo từng mẫu sau khi đọc transcript (tương đương gán nhãn tay tập test). Mọi luật ghép hai ASR phải thiết kế và đo trên val/M2. Công cụ: `agent/compare_runs.py` → `agent/runs/metrics/compare.e46f_vs_L2.public.json`.

**Tách 88 mẫu khác state:** 78 khác câu cuối, 4 chỉ khác transcript lịch sử, 1 không khác transcript, C3 góp 2 (0238, 0252), nhiễu giữa hai lần chạy chỉ ~2 (đối chứng e46f chạy 2 lần: 2/634). → **83 mẫu đổi do ASR.**

**FC lật theo transcript (độ mong manh):** 368/634 câu cuối có transcript khác → 83 mẫu đổi state (≈ 23%). Theo loại khác biệt:

| Transcript khác ở | số câu | đổi state | tỉ lệ |
|---|---|---|---|
| giá trị số | 34 | 16 | **47%** |
| tên riêng / từ nước ngoài | 149 | 38 | 26% |
| động từ hành động | 31 | 6 | 19% |
| từ thường | 130 | 22 | 17% |
| cách viết số (6 ↔ sáu) | 12 | 1 | 8% |
| từ đệm | 12 | 0 | 0% |

Đọc 83 mẫu: mỗi bên "trông đúng" ở số mẫu gần bằng nhau (nhóm số: e46f ~9, L2 ~2; nhóm tên: chia đều) → **không phải ASR nào kém hơn, mà FC đổi quyết định theo từng biến thể nhỏ của câu** — cùng hiện tượng với val sạch 76,5 → M2 63,5.

**Kiểu hỏng thấy được (cả hai bản):**
1. *Một từ nghe sai đổi app/hành động:* 0064 "Chuyển cho chủ nhà…" → "Kiểm tra chỗ nhà…" → L2 tắt đèn thay vì chuyển tiền; 0601 "đặt giùm hai vé" → "tắt giùm" → `my_bookings` thay vì `book`; 0136 "26 độ" → "16 độ".
2. *Thực thể nát → tìm rỗng → chuỗi chết:* 0356 "Stand Up Comedy" → "startup company" (L2 dừng sau 2 lần tìm rỗng); 0669 (e46f tìm "cà phê muối" rỗng 3 lần); 0567. N1 không cứu được vì truy vấn sai từ gốc.
3. *Dừng sau tool đọc (do nhiễu):* public: kết thúc bằng tool ĐỌC e46f 13,9%, L2 15,1%. So cùng loại trên val_in: gold 10,9% (14/128); dự đoán câu sạch 14,1% nhưng **0 mẫu** gold kết thúc bằng ghi mà model dừng ở đọc; dự đoán M2 (có nhiễu ASR) 16,4% với **3 mẫu dừng sớm** (train_0272, 1016, 1271). → không phải lỗi của FC trên câu sạch; xuất hiện khi transcript nhiễu (ví dụ public L2: 0212, 0251, 0571, 0416). Phần chênh còn lại trên public có thể do phân bố câu hỏi khác train. FC v3 (train có nhiễu) nhắm đúng chỗ này; chưa đủ căn cứ để thêm data riêng ở A4.
4. *Số:* đổi giá trị số lật kết quả 47% số lần; cả hai ASR đều nghe sai số (L2: 26 → 16 độ, "sáu giờ" → "mười sáu giờ"; ở 0252 L2 nghe "mùng 9" và tìm ra chuyến, e46f nghe "19" và rỗng).
5. *Transcript lịch sử cũng lái quyết định:* 4 mẫu chỉ khác lịch sử vẫn đổi state (0256, 0480, 0600, 0413); không trùng 2 mẫu e46f tự lệch giữa hai lần chạy (0390, 0530) → là tác động thật.
6. *Luồng mua hàng của app mới:* `grocery/ecommerce.checkout` không có trong train (gold train 0 lần), agent gọi 11 (e46f) / 16 (L2) lần; 9/88 mẫu khác nhau có checkout → model không chắc quy ước của app mới.
Tỉ lệ khác: hỏi lại 12,0% (train 14,0%), không gọi tool 6,9% (train 7,2%) — gần train; các mẫu "không gọi tool" đã chẩn đoán ở §1m.

**Kết luận (tách theo thứ sửa được):**
- *ASR quyết định, FC không cứu được:* sai giá trị số (lật 47%, nhóm lật nhiều nhất) và thực thể nát dẫn tới tìm rỗng. Ở nhóm số, đọc tay e46f "trông đúng" ~9 mẫu, L2 ~2 → **L2 kém hơn ở số**, thêm một lý do giữ e46f. Chốt chặn nhiễu của v3 cố ý loại nhiễu đổi số/thời gian/phủ định, nên FC v3 sẽ không đổi được nhóm này.
- *FC v3 nhắm được:* một từ nghe sai làm đổi app/hành động trong khi phần còn lại của câu vẫn chỉ đúng app (0064, 0601), và dừng sau tool đọc khi câu nhiễu (val M2: 3/128).
- *Chưa ai lo:* quy ước app mới (checkout) — cần mẫu luồng mua hàng ở app đổi tên (A4).
Ghép hai ASR (ROVER/chọn theo độ tin) chỉ làm nếu đo trên val/M2.


## 1t. Thiết kế data ASR v3 (12/11 00:40) — căn cứ cho B2/B4/B5

**Vì sao L2 không giúp (ràng buộc thiết kế):** L2 đổi ba thứ cùng lúc — prompt v1 → v2, data v2.1 (TTS + tăng cường), CTC/inter-CTC/KWS — và ra kém hơn ở số (§1s). Không tách được nguyên nhân. → ASR v3 chỉ đổi **data**: từ `e46f`, giữ prompt v1, không thêm loss phụ, LR thấp, 1 epoch, phát lại audio train thật để không quên.

**Mục tiêu lấy từ lỗi của e46f trên 405 đoạn giữ riêng (không lấy từ public):**

| Nhóm | e46f sai | Tác động | Phần trong data mới |
|---|---|---|---|
| `other_name` (sự kiện, sản phẩm, khách sạn, quán…) | 16/119 (13%) | tìm rỗng → chuỗi chết | ~1.2k câu, cùng `foreign_word` |
| `foreign_word` | 8/122 | như trên | (gộp ở trên) |
| giá trị số / giờ | 9/405 câu (câu cuối 5/123) | lật kết quả 47% (§1s) | ~800 câu |
| động từ hành động đầu câu | (thấy ở §1s) | đổi app / hành động | ~300 câu |
| `person`, `code`, `place` | 0/47, 0/34, 1/17 | đã tốt | không thêm |

v2.1 đã cắt câu số vì A1 thấy "sai số thật ≈ 0" (đa số là khác cách viết). Đảo **một phần**: số ít khi sai nhưng mỗi lần sai rất đắt.
**Quy ước nhãn số:** nhãn train trộn hai dạng (768 câu viết chữ, 207 câu chữ số) → nhãn TTS theo đúng tỉ lệ đó; cổng chấm số theo **giá trị** (`asr_noise_b0._nums`), không theo WER thô.
**Chống rò:** tên mới viết độc lập; **không** dùng bất cứ tên nào đã đọc trong public (§1m, §1s, `compare.*.json`, log chạy public); giữ luật loại câu trùng held-out của `REPORT_tts_text.md`.
**Val miền mới:** 300 câu giữ riêng, dày số + tên, đọc bằng **giọng không có trong train** (DashScope có 4 giọng tiếng Việt: 3 cho train, 1 cho val).
**Cổng (B5):** câu cuối held-out không tệ hơn 7,15%; lỗi giá trị số ≤ 9/405; `other_name` < 16/119; câu có từ nước ngoài không tệ hơn; val miền mới thắng e46f ở số và tên. Qua hết mới chạy public.
**Lịch:** text cục bộ 12/11 sáng → audio (cần chủ repo duyệt DashScope) → Kaggle T4 ~4–6 h → cổng → nộp muộn nhất 14/11. **Kỳ vọng 0 đến +1 điểm**; chạy song song, không chen A2 (FC v3 vẫn ưu tiên 1).
**Bên infer song song:** C2 (giải mã lại câu cuối với tên vừa trả về từ tool) đánh thẳng vào "tên nát → tìm rỗng" mà không cần train.


## 1u. Đọc chọn lọc `review/01`, `review/02` trên myserver (12/11 00:20)

**Đã kiểm lại:** hướng dẫn Docker gốc của BTC (`docs/Docker_Guide_ZAC2026_LifeMate.pdf`, `rule.md`) **không có** trên myserver lẫn máy local; bản Docker viết lại (`predict.py`, notebook…) **không có** trên myserver (người review làm ở clone khác, chưa commit). `pyvi` **có** trong môi trường chạy bản 62.76 (điểm đo đúng) nhưng **thiếu** trong `docker/Dockerfile` → đã thêm `pyvi==0.1.1` + kiểm lúc build. Số dư Modal ≈ $20 (review ghi) chưa kiểm được.
**Đã lỗi thời trong review/01:** M3b 56% (giờ 82,1/88,9 sau §1o); hai lần chạy lệch ~6 mẫu (thực tế 2/634 theo state, §1s); L2/G1/C1 đã đo và loại; data FC v3 đã dựng.

**Nhận (vào plan):**
| Đề xuất | Vào bước | Lý do nhận |
|---|---|---|
| Docker theo hướng dẫn BTC (`/code/predict.sh`, `time`, notebook tuần tự, hai model cùng GPU, `training_code/`, bỏ ENTRYPOINT, pyvi, deadline từng request) | **D0** thay D2a, lên đầu | P0: bản nộp không chạy được thì mọi điểm vô nghĩa |
| Đóng băng tái lập (URL+commit HF, `pack_b`/meta vào repo, ghim phiên bản) | **D4** | BTC chạy lại train |
| Nhiễu từ chính ASR lúc thi thay cho ASR khác (B4 của review) | A4 (1) | khớp nghi ngờ của mình ở §1o/§1s về B0 (medium gốc) |
| Cặp tương phản app trùng chức năng (GenTool) | A4 (2) | trúng 14/41 lỗi nhầm app cùng chức năng; `newapps` đã có 60 câu `neg`, mở rộng được bằng mẫu ghép |
| Che tên tool bằng chuỗi vô nghĩa ~10% (Hammer) | A4 (3) | rẻ, chỉ áp tầng 1 có mô tả |
| Hỏi/làm bằng luật cơ học (bỏ tham số bắt buộc → ask) | A4 (4) | nhãn an toàn (khác dạng tự do mình đã từ chối ở §1g); giữ "làm" là lớp lớn nhất theo cảnh báo When2Call |
| Khớp ngữ âm tên từ kết quả tool | **C2 mới** (mở rộng `geo_fix`) | không train, đánh vào "tên nát → tìm rỗng"; cùng họ với C3 đã đo an toàn |
| Chẩn đoán lệch vị trí phương án DỪNG | C4 | miễn phí |

**Từ chối / hoãn:**
| Đề xuất | Lý do |
|---|---|
| "Nghe lại audio với tên trong prompt ASR" (C2 cũ của plan) | 3 paper: prompt từ khoá cho Whisper chưa train làm tăng WER; L2 đổi prompt cũng kém đi (§1s) |
| Train ASR biết dùng prompt từ khoá (B-Whisper) | đổi định dạng prompt = lặp lại sai lầm L2; ASR v3 (§1t) chỉ đổi data |
| N-best ASR trong prompt FC | đổi định dạng → dựng lại data + train lại; Whisper `generate` không trả n-best thật |
| Môi trường giả chạy được (AgentScaler/APIGen-MT) | đúng hướng nhưng 1–2 ngày + LLM ngoài; chỉ cân nhắc nếu D0 xong trước 13/11 |
| Trie biasing, KL sạch–nhiễu, HEART, HarnessFlow, Harness², DPO | tham số/thời gian/không tất định |
| Thêm mẫu DỪNG / "không gọi tool" | paper và §1m đều cảnh báo làm model dừng nhiều hơn |

**Về `get_state` trên public (vùng xám review/01 §4):** các script gán nhãn ASR (`qwen_cloud_label.py`, `label_remaining.py`) không nằm trong pipeline nộp; không đưa vào `training_code/`; không model nào được chọn dựa trên nhãn đó.


## 1v. N1b (mở rộng N1 cho query bắt buộc) + kết luận về C2 khớp ngữ âm + deadline từng request (12/11 00:30)

**Căn cứ:** trong gold train, tool đọc có tham số chữ (`*.search`, `search_place`) **không bao giờ** trả rỗng (0/1545 hội thoại, kể cả history) → rỗng trên public toàn do ASR nghe sai hoặc món không có; không có quy ước gold "rỗng thì làm gì" để học → chỉ cứu được bằng luật lúc infer. N1 (event_tickets, query tuỳ chọn) đã cho +0.72 theo đúng cách này.
**Phát hiện:** `grocery/recipes/ecommerce.search` bắt buộc `query` (validate loại `""`), nhưng `zaic_tools` nhận `"*"` (và `" "`, `"a"`) là **liệt kê toàn bộ** (`match_items`). → **N1b** trong `relax_empty`: không bỏ được tham số nào thì tìm lại với `query="*"`; vẫn loại `contacts`. Cùng cờ `--retry-empty`.
**Chẩn đoán trên log 62.76 (không nhãn):** N1b kích hoạt 11 mẫu (grocery 8, recipes 2, ecommerce 1). Món đích có trong danh sách ở ~4 mẫu: 0240 "Slim Cây 75" → "Bàn phím cơ SlimKey 75"; 0198 "in mật nguyên hạn" → "Yến mạch nguyên hạt"; 0669 "két chuối kiểu robot cà phê" → "Bánh pancake chuối kiểu Bluebird Cafe"; 0390 nguyên liệu cá áp chảo. 7 mẫu còn lại món thật không có trong cửa hàng (danh sách chỉ 5–10 món) → FC có thể chọn sai, nhưng các mẫu đó hiện đang DỪNG/tìm lặp = 0 điểm sẵn. Kỳ vọng +0.3–0.6; **chỉ đo được bằng một lượt public** (val không có tìm rỗng), gộp vào lượt nộp kế (S-B).
**C2 khớp ngữ âm (review/02 B2) — kết luận sau khi đọc dữ liệu:** danh sách của mọi tool chỉ 4–10 mục, FC chọn đúng 7/7 lần sau khi N1 liệt kê (0214, 0253, 0289, 0336, 0356, 0423, 0520) → khớp ngữ âm chỉ thêm giá trị khi danh sách dài hoặc FC chọn sai, chưa thấy trên public. Với `contacts` (0221 "Lan" ↔ "Liên"): khớp ngữ âm cũng không nhận (độ giống < 0.75) và liệt kê danh bạ rồi để FC chọn người nhận tiền là rủi ro sai hướng → **giữ loại contacts**. → C2 hạ ưu tiên: chỉ làm nếu sau N1b còn thấy FC chọn sai trong danh sách.
**Deadline từng request (D0, review/01 §3.4):** `request_json(deadline=)` — mỗi request chờ tối đa tới hạn tuyệt đối của mẫu (= ngân sách − giây ASR − lề), quá hạn ném `BudgetExceeded` → `run_one` đặt `timed_out`, giữ call đã chạy, không tính lỗi. Trước đó: 600 s × 8 lần. Chỉ bật khi có `--time-budget` (Docker); chạy online không ảnh hưởng.

## 1w. Xử lý trước/sau ASR (12/11 01:10) — đo xong, gần như không còn chỗ

- **Audio trước ASR:** 300 file train vs 300 public: cùng 16 kHz mono, âm lượng trung vị −26 dB, lặng đầu ~0,45 s, dài nhất 20 s (public), 0 file > 30 s → không có gì để chuẩn hoá; lọc ồn/cắt lặng chỉ làm lệch so với lúc train e46f.
- **Lỗi còn lại của e46f (405 held-out, `agent/asr_analyze.py`):** 257 cụm; từ thường nghe sai 54%, từ đệm 11%, mất/thừa từ 14%, tên riêng 9%, cách viết số + dấu 11%, sai giá trị số 1%. Lỗi có từ nội dung: 88, chỉ **11 (12%)** có từ đúng trong ngữ cảnh được phép (kết quả tool lịch sử 6, catalog 3, hội thoại 3, hồ sơ 3); 19 lỗi có từ đúng nằm sẵn trong prompt ASR mà vẫn sai.
- **Sửa tham số chữ của tool ghi theo tên trong kết quả tool (ý "việc 5" / C2 mở rộng):** quét log 62.76: 374 tham số chữ của tool ghi, 15 ứng viên gần âm (≥ 0,75) với một tên trong kết quả tool — **gần như toàn sửa nhầm**: "canh chua" ≈ "cà chua", "cờ ba" ≈ "Ông Ba", "cà phê với mình" ≈ tên sự kiện "hẹn cà phê với Minh", nội dung tin nhắn diễn đạt lại tên đúng. Model đã chép đúng tên từ kết quả tool khi cần. → **Bỏ**, kỳ vọng ≈ 0, rủi ro cao.
- **Bảng "hay sai → sửa" trên transcript:** không có nguồn sạch (audio train trong mẫu của e46f, held-out là thước đo, public bị cấm). Nguồn sạch duy nhất: e46f nghe 6 198 đoạn TTS (e46f chỉ train audio thật v1) → dùng làm **nhiễu cho data FC (A4 mục 1)**, không làm bộ sửa.
- **Còn lại (ưu tiên thấp):** chấm lại n-best ASR bằng Qwen (Whisper `generate` không trả n-best thật, phải tự viết beam); chuẩn hoá phương ngữ (đo M2 trước, FC đã học câu thô); chuẩn hoá số (không lợi).
- **Kết luận:** cơ chế sửa sau ASR an toàn duy nhất là khi có **tín hiệu rõ** — tìm kiếm trả rỗng (N1, N1b, C3). Phần còn lại phải đến từ ASR train lại (B2–B5) hoặc data FC (A4).

## 1x. Bài học từ các paper trong review/02, đặt cạnh số đo của mình (12/11 01:20)

**Chẩn đoán miễn phí (review gợi ý):** nhãn DỪNG ở bước đầu trong data: `fmtA_48e_20k` 12,3%, `fmtA_v3` 10,7% (cao hơn gold câu cuối 7,2% vì có mẫu phụ history "cảm ơn"); ~10% đúng mức Hammer khuyên. Trên public bản 62.76 dừng ngay bước đầu 6,9% ≈ gold 7,2% → **model không dừng quá tay nói chung**; lỗi DỪNG tập trung ở app mới (20/41) → do thiếu hiểu biết app mới, không phải do tỉ lệ nhãn DỪNG. Không giảm mẫu DỪNG.
**Paper xác nhận điều mình đã đo:** prompt từ khoá cho Whisper chưa train thì hại (↔ L2 đổi prompt, kém đi); LLM tự sửa transcript hại với ASR mạnh (↔ bỏ bộ sửa); khớp ngữ âm luôn kèm sửa nhầm (↔ 15/15 ứng viên ở §1w là sửa nhầm); bỏ phiếu n-best không train lợi rất nhỏ; cùng một model nhìn lại lần hai không cứu được nhãn xác suất gần 0 (↔ N8, I5, G1 — P(tool đúng) ≤ 0,13); Gorilla: train với tài liệu nhiễu, không lọc cứng (↔ catalog hai tầng + decoy đã làm).
**Paper làm đổi việc cần làm:** nhiễu từ chính ASR lúc thi chuyển giao tốt hơn ASR khác → A4 (1): cho e46f nghe 6 198 đoạn TTS (ngoài mẫu với e46f) làm nhiễu FC; train trên 1-best có nhiễu có thể không hơn câu sạch (Sun et al.) → hạ kỳ vọng FC v3, cổng A3 là bắt buộc; When2Call: trộn đều hỏi/làm làm model 4B dè dặt → giữ "làm" là lớp lớn nhất (model hiện hỏi lại 12% so với gold 14%, không hỏi quá tay); GenTool, Hammer → A4 (2), (3).
**Đáng cân nhắc nhưng đắt:** môi trường app tổng hợp chạy được (AgentScaler: Qwen3-4B 43,9 → 65,6 trên ACEBench-zh — con số mạnh nhất và cùng cỡ model) đánh đúng lỗi "tìm thấy rồi dừng" ở app mới; bản rẻ đã có một phần (alias/iface = app train đổi tên, có ToolEnv thật) nhưng không dạy được app có cấu trúc khác. 1–2 ngày + LLM ngoài → chỉ làm nếu D0 xong sớm.
**Thử rẻ:** hoán vị vị trí phương án DỪNG trong Q1 trên M3b (PriDe) ~$1 — nếu tỉ lệ chọn DỪNG đổi theo vị trí thì có lệch theo nhãn, sửa được lúc infer bằng trung bình hai thứ tự.
Lưu ý: số liệu paper trong review lấy từ bản tóm tắt tự động, chưa kiểm toàn văn.

## 1y. Đo PriDe (khử lệch vị trí Q1) + bản nộp N1 + N1b + C3 (12/11 01:50, Modal workspace `thang-nt3092004`)

Hạ tầng: profile Modal đã đổi sang `thang-nt3092004`; tạo secret `llm-api-key` (01:07); workspace mới mặc định image builder 2026.10 mà client 1.6.1 không hỗ trợ → bật server với `MODAL_IMAGE_BUILDER_VERSION=2025.06`; FC kéo thẳng từ HF `q-f48e@0cd1ed0`. Server đã stop.
**PriDe — KHÔNG giữ:** M3b (176 câu sạch, val + test): thứ tự gốc 85,2% (khớp mốc 82,1/88,9), thứ tự đảo 83,0%, trung bình hai thứ tự 83,0%; số lần chọn DỪNG: gốc 1, đảo 3 → **không có lệch về phía DỪNG ở vị trí cuối**. M1 76,5 (1 tốt hơn, 3 kém hơn), M2 63,9 (+0,5, 1/1) — trong nhiễu. Chi phí: request/mẫu 7,2 → 10,1, latency p90 ~2× → không đáng. Không chạy public B.
**Bản nộp A = e46f + N1 + N1b (`--retry-star`) + C3:** `result/submission_fmtA_48e_e46f_n1b_c3.jsonl`, 634 dòng, 0 lỗi schema, 1438 call. So với 62.76: khác state 9 mẫu — N1b 4 (0240 SlimKey 75 → `add_to_cart prod_08`; 0669 lưu công thức pancake chuối kiểu Bluebird Cafe; 0198, 0379 thêm vào giỏ — chưa chắc đúng món), C3 2 (0238 đặt vé Lạng Sơn; 0252), nhiễu giữa hai lần chạy 4 (0136, 0390, 0530, 0516; server ở workspace khác). N1b kích hoạt 10, 6 mẫu còn lại không đổi state.
Kỳ vọng nộp: 62.76 + ~0,3–0,6 (N1b 2–4 mẫu, C3 1 mẫu), nhiễu ±0,3.

**Kết quả nộp A (12/11): 62.70** (bản 62.76 = e46f + N1). Chênh −0.06, nhỏ hơn giá trị một mẫu, tức 9 mẫu khác nhau gần như triệt tiêu: các sửa N1b/C3 không được tính thêm điểm (hoặc chỉ bù cho mẫu mất do nhiễu chạy). Kết luận: N1b và C3 **không đem lại điểm public**; các mẹo suy luận không train lại coi như đã cạn. Bản nộp tốt nhất vẫn là `result/submission_fmtA_48e_e46f_n1.jsonl` (62.76). Đòn bẩy còn lại: A2 (train FC trên fmtA_v3).

## 1z. Tìm paper/industry lần 2 (12/11, 2 agent: ASR-số, FC-chịu-lỗi) + đo số trên held-out
**Đo số held-out (e46f, 405 câu):** chỉ 9 câu đổi giá trị số (2,2%): 2 chỉ khác dạng viết (0664 SĐT đọc từng chữ = cùng giá trị; 0515 "hai chín"↔"29"), 3 sai giá trị thật (1366 bốn lăm→bốn mươi; 1521 ba mươi→"Ba mười"; 0699 rơi chữ số cuối SĐT), 4 do từ khác nghe thành số (sớm→tư, ni→hai…). Decode không dùng `no_repeat_ngram_size`. → ITN/chuẩn hoá dạng số sửa tối đa ~2/405 → **không đáng làm**; lỗi số thật quá ít để train riêng.
**ASR (không thêm gì mới đáng làm):** NeMo ITN tiếng Việt (thiếu "chục", buổi), Peyser 1907.01372 (TTS chứa số; dạng nói + ITN chỉ thắng ở chuỗi số dài), VLSP 2025 (đội nhất chuyển nhãn sang dạng nói; không ai đo riêng số). Không có writeup ASR Zalo.
**FC (ứng viên cho data A2):** ToolACE-MT 2508.12685 (lọc trajectory bằng kiểm định; bỏ kiểm định mất ~8 điểm BFCL MT) — mình đã replay gold (§1o), phần mới là bỏ mẫu kết thúc ngay sau search khi user còn yêu cầu hành động; FunReason-MT 2510.24645 (chuỗi buộc chạy tới tool đích); EPA 2409.06263 (dồn nhiễu vào từ quyết định nhánh và span tham số); PhoATIS_Disfluency 2209.08359 (VinAI, mẫu tự sửa lời tiếng Việt thật, public); OLISIA 2304.11073 (DSTC11 hạng 1: đa dạng giá trị entity +7 JGA, mô phỏng TTS→ASR chỉ +0,1–1,1, paraphrase LLM −2–3); 2609.17848 (Qwen3-4B: LoRA > full FT; RL ≈ SFT). Không có công trình nào về câu điều kiện dưới nhiễu.
**Kiểm hai ý trước A2 (12/11):** (a) ToolACE-MT "lọc mẫu dừng sau search": fmtA_v3 có 834 mẫu DỪNG ngay sau tool đọc; đọc mẫu thì gần hết là câu hỏi thông tin (giá vàng, thời tiết, số dư) → đúng; lọc theo động từ hành động còn 125 mẫu, phần lớn là regex bắt nhầm ("đang bật") → data không dạy dừng sớm có hệ thống, ý (a) gần như không có việc. (b) Tự sửa lời: đã có 550 ví dụ / 123 hội thoại có "à không / thôi khỏi / nhầm…" ở câu cuối → PhoATIS chỉ thêm đa dạng mẫu câu, lợi nhỏ. **Kết luận: train A2 với data hiện tại, không chờ.**

## 1z2. Độ khó của public đo không cần nhãn (12/11)
Không gán nhãn test (thể lệ). Thay bằng so 2 ASR (e46f vs L2) trên audio public, câu cuối 634: lệch ở 58% câu nhưng WER giữa 2 model TB ~5%; 27 câu lệch > 20%. Theo đặc điểm (có / không): phương ngữ 84 câu, 8,7% vs 5,1%, câu lệch > 10%: 30% vs 17%; tự sửa lời 62 câu, 7,6% vs 5,4%; số 194 câu, 6,3% vs 5,2%; tên Latin 3,9% vs 5,7% và câu > 30 từ 4,7% vs 5,8% (không khó hơn). Đối chiếu: data FC v3 có 12,2% câu cuối có từ địa phương (public 13,2%) → FC không thiếu; TTS v2 có 1 068/6 198 câu có từ địa phương nhưng 4 giọng CosyVoice không chỉnh giọng vùng (`instruction` False), Huế 147, Cần Thơ 129. Held-out (audio thật của train): e46f WER 12,7% câu có từ địa phương vs 11,9% → khoảng cách nhỏ, cần đo theo vùng người nói. Bộ đếm theo danh sách từ, chỉ là xấp xỉ. → B5, A4 (4) trong plan.
**Đo theo vùng (vĩ độ người dùng trong context train):** held-out e46f WER Bắc 11,3% (65 câu) · Nam 12,4% (47) · Trung 13,1% (59); câu cuối 10,2 / 11,8 / 12,0. Train: Bắc 791, Nam 435, Trung 319 hội thoại. Khoảng cách có nhưng nhỏ (1–2 điểm, mẫu ít) → B5 đáng làm ở mức vừa, không phải đòn bẩy lớn.

## 1z3. Nguồn TTS cho data ASR v3 (12/11 sáng)
**Audio TTS v2 so với audio thật** (mẫu 52/84/65 file): sr đều 16 kHz; dải tần 99% năng lượng 2,9 kHz vs 3,4–3,6 kHz; RMS −18,4 dB vs −25,7 dB (to hơn 7 dB); đỉnh 0,70 vs 0,42; 2% file cắt đỉnh. 4 giọng CosyVoice `longan*_v3.1`, `instruction` tắt → **không có giọng vùng**, chỉ có từ địa phương (1 068/6 198 câu). → Mọi TTS mới phải chuẩn hoá RMS/đỉnh về mức audio thật.
**Alibaba (workspace Singapore, key trong `.env`):** `qwen-audio-3.1-tts-flash` và `-plus` sinh được qua `/api/v1/services/audio/tts/SpeechSynthesizer`, giọng `longanhuan_v3.1`; 1 câu 88 ký tự = 122 token (flash) / 139 (plus); `qwen-audio-3.1-tts-next` (không có tên `flash-next`) cần `text_prompt`, 1,7–2,2k token/câu, có mô tả giọng vùng. **Clone giọng (`voice-enrollment`, `qwen-voice-enrollment`) bị `AccessDenied.Unpurchased` ở cấp tài khoản** dù key đã có quyền; không bật được từ console → bỏ hướng clone.
**ElevenLabs (61k credit):** thư viện có 346 giọng tiếng Việt có nhãn vùng (Trung 38, Nam 110, Bắc 114, chuẩn 416, nhãn tự khai); 165 preview + `index.csv` ở `data/voices_regional/`; sinh thẳng bằng `eleven_flash_v2_5` chạy được, 8 mẫu ở `probe_elevenlabs/` (~350 credit). **Quyết định:** giọng chuẩn → DashScope flash (quota 1M token); giọng vùng miền → ElevenLabs (B5); `tts-next` là dự phòng (B5b). Chưa nghe mẫu nào, chủ repo nghe và chọn giọng. Script: `synth/regional_tts/`.

## 1z4. Augment nhiễu cho ASR v3 (12/11)
Đo nền ồn (khung 25 ms, phân vị 95 − 10): audio train SNR ~41 dB (nền −60 dBFS), public ~35 dB (−55 dBFS), TTS v2 ~55 dB (−68 dBFS), ElevenLabs ~43 dB. Public hơi ồn hơn train một chút; không có nhạc nền hay tiếng ồn lớn. `modal/asr_train.py::augment_wav` (đổi tốc độ 0.9/1.1, nhiễu trắng/hồng SNR 10–25 dB, vang RT60 0,15–0,5 s), bật với `audio_aug=0.5` ở v2 medium và **L2** → mẫu nhiễu ồn hơn audio thật 10–30 dB: thêm một nghi vấn cho việc L2 không lên điểm (L2 đổi nhiều thứ cùng lúc, không tách được). e46f không bật. **ASR v3:** SNR 30–45 dB, p ≈ 0,3, chủ yếu cho mẫu TTS; giữ tốc độ; bỏ vang; SpecAugment giữ 0,05.

## 1z5. Survey augment nhiễu cho ASR + đo đặc tính audio thật (12/11)
**Đo (≈60 file mỗi nguồn, khung 25 ms; phổ tương đối so với dải mạnh nhất):**
| | im lặng đầu | phổ NHIỄU 0-300/300-1k/1-2k/2-3k/3-4k/4-6k/6-8k | phổ TIẾNG cùng dải |
|---|---|---|---|
| train | 0,60 s | 0/−1/−5/−4/−6/−16/−39 | −6/0/−8/−9/−17/−28/−53 |
| public | 0,60 s | 0/−2/−4/−3/−5/−14/−39 | −5/0/−8/−8/−15/−27/−48 |
| TTS v2 | 0,10 s | 0/−11/−19/−22/−29/−36/−40 | 0/−4/−16/−24/−28/−36/−44 |
| ElevenLabs | 0,03 s | 0/−11/−12/−17/−21/−24/−26 | 0/0/−14/−15/−23/−34/−42 |
Kết luận: (1) **train và public cùng một kiểu kênh**: nhiễu nền gần phẳng tới 4 kHz rồi tắt ở 6 kHz; public chỉ ồn hơn ~5 dB (SNR ~35 vs ~41). (2) Nhiễu trắng/hồng của `augment_wav` **sai hình phổ**. (3) TTS thiếu năng lượng 1–4 kHz so với tiếng thật (−16 vs −8 ở 1–2 kHz) → nghe đục; ElevenLabs còn giữ năng lượng trên 6 kHz mà audio thật không có. (4) TTS gần như không có im lặng đầu câu (0,03–0,1 s so với 0,6 s).
**Paper/thực tế:** công thức chuẩn là MUSAN + RIR + đổi tốc độ 0.9/1.1 (Kaldi), nhưng các mức SNR 0–20 dB là cho miền ồn; paper chỉnh xác suất theo SNR của miền đích; Whisper-base Pashto: tốc độ + nhiễu (p 0,7) −7,25 điểm WER khi data ít; TTS cho ASR cần "nhiễu đại diện" + nhiều người nói; khoảng cách thật–tổng hợp vẫn lớn (2410.12279). Không thấy paper lấy nhiễu từ đoạn im lặng của audio thật rồi trộn vào TTS — nhưng đây là cách khớp miền đích trực tiếp nhất, và hợp lệ vì chỉ dùng audio train.
**Công thức ASR v3 (thay B4 cũ):** cho mẫu TTS (luôn): EQ khớp phổ tiếng thật (tăng 1–4 kHz, cắt thấp > 6 kHz) → đệm 0,4–0,8 s im lặng đầu/cuối → trộn **nhiễu thật lấy từ đoạn im lặng của audio train** ở SNR 30–42 dB → RMS −25,7 dB. Cho mọi mẫu (p ≈ 0,3): đổi tốc độ 0.9/1.1; nhiễu thật SNR 28–38 dB (phủ mức public ~35). Bỏ vang và nhiễu trắng/hồng. SpecAugment giữ 0,05.

## 1z6. Thiết kế ASR nhiều bản + đánh giá ý tưởng 2 ASR song song (12/11)
Ý tưởng chủ repo: train thêm ChunkFormer-RNNT và Gipformer 10 epoch, đưa 2 transcript vào Qwen để tự viết lại theo ngữ cảnh, rồi vào FC. Đánh giá: (a) hai model CTC/RNNT **không nhận ngữ cảnh lượt trước**; (b) 10 epoch trên ~14 h audio sẽ học thuộc (Qwen 0.6B 6 epoch: 13,2 → 16,3 WER, §1n) → 2–3 epoch có dừng sớm; (c) **bước LLM viết lại tự do là phần rủi ro nhất**: hại với ASR mạnh (review 02 §4), bỏ phiếu không train lợi ~1,6%, 15/15 ứng viên sửa ngữ âm ở §1w đều sai, thêm 1 lượt LLM/mẫu; (d) ý cốt lõi "FC thấy nhiều cách nghe" thì đúng, nhưng phải theo Sun et al.: **đưa N-best vào prompt và train FC trên định dạng đó**, không viết lại trước. Beam 5 của e46f đã cho nhiều bản, 0 chi phí; ASR thứ hai chỉ thêm khi đo được nó bù lỗi mà beam chưa bù. → B6 (3 bước có cổng) + A4 (5). Train từ đầu không khả thi với ~14 h audio.

## 1z7. Dựng data ASR v3 — text, kho nhiễu, xử lý audio (12/11 trưa)
**Text** `synth/tts_text_v3/` (`entities_v3.py`, `num_v3.py`, `templates_v3.py`, `build_v3.py` → `tts_text_v3.jsonl`): 3 500 câu = tên/từ nước ngoài 1 200, số 800, động từ dễ nhầm 300 (giọng chuẩn) + phương ngữ Trung 600, Nam 600. Val: 293 câu chuẩn + 220 phương ngữ (thực thể của val tách riêng, 15% cuối mỗi danh sách). Số sinh từ GIÁ TRỊ, nhãn theo quy ước train (đo: giờ 83% chữ, tiền 72%, độ 50%, ngày 85%, SĐT 88%); Nam dùng "ngàn"/"lẻ"/"hăm"; không nói đơn vị thì nhãn chữ. Chống rò: held-out (6 từ liền, từ chỉ có ở held-out: bỏ 147) + **public** (5-gram và tên viết hoa trong transcript public e46f/L2, chỉ để lọc: bỏ 232). Lời đọc không còn chữ số/viết tắt Latin.
**Giọng val B2:** flash chỉ có 4 giọng tiếng Việt và cả 4 đã ở TTS v2 → giữ `longanfengyue_v3.1` cho val, **bỏ 1 301 clip v2 train của giọng này** (v2 train còn 3 922). `tts-next` không dùng làm val: đọc ~2,2 từ/s (thật ~3,7), có đoạn lặng 2 s.
**Kho nhiễu** `data/noise_bank/`: 2 850 đoạn (21 phút) cắt từ đoạn im lặng audio train, bỏ 405 held-out.
**Xử lý TTS** `augment_v3.py` (fit/apply/check): EQ theo LTAS (đo, lệch tới +12 dB ở 2–5 kHz), cắt > 6,5 kHz, đệm lặng, nền nhiễu thật định dạng phổ, RMS −25,7. Lần sửa: nối nhiễu phải nối mềm + bỏ DC; kho nhiễu ghép có phổ phẳng → định dạng theo phổ nhiễu thật. Kết quả 100 mẫu v2: RMS −25,7 (thật −26,2), SNR 41 (40), phổ tiếng lệch < 5 dB, phổ nhiễu < 3 dB. Mẫu nghe: `data/tts_v3/listen/`.
**Audio chuẩn** đang sinh: `synth_dashscope_v3.py` → `data/tts_v3/wav_raw/` (flash, 3 giọng train + 1 giọng val).
**12/11 chiều — B2 audio xong:** 2 300/2 300 câu (54 lỗi mất kết nối, chạy lại đủ), 134 430 token flash (13% quota 1M), 2,24 h audio; sau xử lý: RMS −25,7 (thật −26,2), SNR 39 (40), lặng đầu 0,59 s (0,62), phổ tiếng/nhiễu lệch < 5 dB. TTS v2 xử lý lại xong (6 198 file, `data/tts_v2/wav_v3proc/`). Manifest `data/tts_v3/tts_manifest_v3.json` (`make_manifest_v3.py`): train TTS 5 940 (v2 3 922 + v3 2 018), val v3 282, val_voice 1 540, holdout v2 736. `modal/asr_train.py` thêm `--real-repeat`, `--aug-v3` (mặc định giữ hành vi cũ; bản trước ở `asr_train.py.bak_v2`); smoke test augment_v3 OK. Còn: B5 audio (chờ chọn giọng), kiểm audio bằng e46f, đóng gói DATA cho Kaggle/Modal (thư mục `tts_v2p/`, `tts_v3/`, `tts_v3el/`, `noise_bank/`).
**12/11 chiều — chọn giọng B5 + smoke:** e46f (Kaggle T4, greedy, không prompt) nghe 2 câu thử × 20 giọng: Trung WER 0,07–0,15, Nam 0,12–0,21; cả 10 giọng Nam đều bị nghe "cho bả nghen" → "cho ba ngàn" (lỗi phương ngữ thật của e46f). Chọn theo độ khó tăng dần (`data/voices_regional/selected_voices.json`): Trung train Ethan 0,07 / Zara 0,10 / Nathan 0,15, val Jay 0,10 (Huế–ĐN); Nam train Tram 0,12 / Tâm 0,18 / Hung 0,21, val Cẩm Hồng 0,21. Smoke 16 câu (~555 credit) `data/tts_v3/smoke_el/{raw,processed}`; sau xử lý (EQ riêng `eq_eleven.json`): RMS −25,7, SNR 37, phổ lệch < 5 dB. **Lỗi text đã sửa:** mẫu "{QTY} vé" ra "sáu ký vé", "hai vé vé" — 114 câu sửa tại chỗ (giữ id; bản trước `tts_text_v3.before_vefix.jsonl`), 30 audio chuẩn sinh lại; mẫu đổi sang slot `{NVE}` cho lần dựng sau. B6: Gipformer đang chạy; ChunkFormer OOM ở T4 (batch 1 800 s) → chạy lại batch 240 s.
**B6 bước 2 — zero-shot trên 405 held-out (Kaggle, 12/11):** chuẩn hoá chung (thường, bỏ dấu câu, số → chữ). e46f 5,77% (câu cuối 6,47%, 246 câu đúng hết); Gipformer1.5-68M greedy 15,68% / beam4 15,55% (câu cuối 15,26%, 100 câu đúng); ChunkFormer-RNNT-large 22,13% (câu cuối 25,48%, 78 câu đúng). Bù cho 159 câu e46f sai: Gipformer đúng hết 10, ít lỗi hơn 25; ChunkFormer 7 / 17; **câu cuối: mỗi model chỉ đúng hết 1/66 câu e46f sai**. WER oracle (biết trước câu nào tốt hơn) 4,95% / 5,29% — không đạt được trong thực tế. → **Không qua cổng** (≤ 10% và bù ≥ 1/3): dừng B6 bước 3, không fine-tune, không gắn ASR thứ hai; A4 (5) chỉ dùng beam của e46f. Transcript: `scratchpad/b6k/out_{gip,chunk}/`.
**12/11 chiều — B5 audio xong:** tài khoản ElevenLabs mới (10 000 credit, payg, không mở rộng được giới hạn → không phát sinh phí). 1 200/1 200 câu phương ngữ, 0 lỗi, **8 010 credit** (thực tế ~0,1 credit/ký tự với eleven_flash_v2_5 trên tài khoản này, không phải 0,5), 1,34 h audio. EQ refit trên 200 clip (`eq_eleven.json`); sau xử lý: RMS −25,7 (thật −26,2), SNR 39 (40), lặng đầu 0,53 s (0,62), phổ tiếng lệch ≤ 7 dB ở 0–150 Hz, ≤ 5 dB phần còn lại. Manifest v3: **train TTS 6 941** (v2 3 922 + v3 chuẩn 2 018 + phương ngữ 1 001), val v3 282 + 199, val_voice 1 540, holdout v2 736. Tổng train ASR v3 = 4 631 thật (×2) + 6 941 TTS = 11 572 câu khác nhau, ~16 200 lượt/epoch.
**12/11 chiều — bổ sung Huế + miền Tây:** Jay (Huế–ĐN) chuyển từ val sang **train** (97 câu); val Trung đổi sang giọng **Vinh** (Quảng Nam, chưa dùng) với **97 câu text mới** (dựng lại seed khác, lọc trùng + chống rò; bản trước `tts_text_v3.before_hue.jsonl`); thêm **36 câu miền Tây** (Khánh Tú — ĐBSCL, Nam — Bến Tre; text Nam train cũ, giọng mới). Credit 9 673/10 000. Manifest: train TTS **7 074** (phương ngữ 1 134 = Trung 600 + Nam 534), val phương ngữ 199 (Vinh, Cẩm Hồng). **Tổng train ASR v3: 4 631 thật (×2) + 7 074 TTS = 11 705 câu, ~16 300 lượt/epoch.** Đã đồng bộ lên myserver `data/asr_v3/`.
**12/11 tối — B4 bắt đầu (Modal H100, workspace thang-nt3092004):** volume `zaic-asr-v3-data` (2,3 GB), app `lifemate-asr-train-v3` (`modal/modal_asr_v3.py`). Smoke 24 hội thoại: chạy được, bước đầu 22,5 s (khởi động), sau đó ~2 s/bước (batch hiệu dụng 32) → ước ~510 bước/epoch ≈ 20–30 phút + eval 405 câu ~2–3 phút; 6 epoch ≈ 3 h ≈ $12–14 (chỉ từ 2 bước, cần xác nhận ở epoch 1). Chạy thật: call `fc-01M777QYXZ35KZ34PHXW6Y37RX`, tag `lora_ls0.1_ctc0.3_prompt_tts_v3base`, tối đa 6 epoch, patience 1, eval đủ 405 câu, giới hạn 8 h.
**12/11 tối — chuẩn bị cổng B4 và data A4 (chưa chạy, chờ B4 + Kaggle N-best):**
- `agent/asr_gate_v3.py`: so ASR v3 với e46f trên 405 held-out (WER tất cả/câu cuối/lịch sử, theo vùng, câu đúng hết, câu sai giá trị số, thực thể có trong câu) + tuỳ chọn TTS val giọng chưa nghe. Tự kiểm (e46f so với chính nó): 5,77% / câu cuối 6,47% / Bắc 5,55, Trung 6,96, Nam 5,01 / 246 câu đúng / 9 câu sai số / 12 trên 148 thực thể sai.
- `agent/clef/nbest_to_noise.py`: N-best e46f → `e46f_tts.asr.json` + manifest + bảng cụm `asr_subst_e46f_tts.json` + `e46f_tts.nbest.json`; in WER theo nhóm TTS và B6 bước 1.
- `agent/clef/make_a4.sh`: đúng lệnh fmtA_v3, đổi nguồn nhiễu: câu thật giữ lỗi PhoWhisper-medium (e46f đã train trên audio đó), câu TTS dùng e46f; bảng cụm = v2 ∪ e46f-TTS (có lỗi phương ngữ). **Giữ bộ lọc số**: không có "cách nghe khác" thì nhãn sạch + số sai là không học được. `agent/clef/merge_shards.py` (chép từ scratchpad).
- **A4 (5) "cách nghe khác" cần đổi cả lúc infer** (ASR trả N-best, `clef_agent` đưa vào prompt, Docker) → quyết riêng sau khi có số B6 bước 1.
**12/11 tối — chuẩn bị sau B4:** `modal/modal_asr_infer_v3.py` (app riêng `lifemate-asr-infer-v3`, L40S): chép lời đúng điều kiện thi như `agent/asr_modal.py` (beam 5, prompt = transcript chính model các lượt trước) cho held-out 405 + public 1 933 + TTS val (giọng chưa nghe) trong 1 lần gọi; audio public đã lên volume `zaic-asr-v3-data`. Tự kiểm với e46f trên 3 đoạn held-out: **trùng nguyên văn 3/3** với transcript e46f cũ → so ASR v3 với e46f là cùng điều kiện. Ghi chú: **log train từng epoch của e46f đã mất** (workspace Modal cũ; repo HF chỉ có model; L1 large-từ-gốc lỗi sau 10 phút) → B4 là log large-từ-gốc đầu tiên; mốc so sánh duy nhất là cổng đo cuối. B4 epoch 1–4: 9,26 / 8,76 / 7,83 / 7,18% (v2 medium cùng epoch: 11,82 / 9,34 / 8,45 / 8,42%).
**B4 KẾT QUẢ (12/11 15:40) — không qua cổng, giữ e46f.** Modal H100, dừng sớm sau epoch 5 (WER train-eval 9,26 / 8,76 / 7,83 / **7,18** / 7,97%), bản tốt nhất epoch 4. Eval cuối của script (beam 5, prompt = transcript chính model): WER 6,89%, CER 4,31% (prompt oracle 6,66%, không prompt 7,59%). Cổng `asr_gate_v3.py` (cùng chuẩn hoá, cả hai beam 5, cùng điều kiện thi):
| held-out 405 | e46f | ASR v3 (B4) |
|---|---|---|
| WER tất cả | 5,77% | 6,43% |
| câu cuối | 6,47% | 6,68% |
| lịch sử | 5,13% | 6,21% |
| Bắc / Trung / Nam | 5,55 / 6,96 / 5,01% | 6,30 / 7,89 / 5,34% |
| câu đúng hết | 246 | 233 |
| câu sai giá trị số | 9 | 12 |
| thực thể sai (148) | 12 | 15 |
→ thua ở mọi mục, kể cả vùng Trung (data phương ngữ không bù được). Nhận định: train lại từ PhoWhisper-large gốc với ~16k lượt × 4 epoch chưa bằng e46f (log train e46f đã mất nên không biết e46f train bao lâu); epoch 5 xấu đi khi LR còn ~25% (CER tăng vọt → vài câu lặp/bịa). Chi phí B4 ~1,8 h H100 ≈ $7–8. Model: volume `zaic-asr-v3-out/lora_ls0.1_ctc0.3_prompt_tts_v3base` (không đẩy HF). Dự phòng còn lại: train tiếp từ e46f 1 epoch, LR thấp, cùng data v3 (~$3–4).
**12/11 chiều — N-best e46f xong (Kaggle, 2 × T4 mỗi tài khoản, 10 236 câu, ~70 phút):** sửa 2 lỗi: (1) `WhisperForConditionalGeneration.generate` nhân đầu vào thêm `num_return_sequences` lần TRƯỚC beam search (5 × 5 = 25 chuỗi/câu) → tràn T4; gọi `GenerationMixin.generate` trực tiếp với 4 token đầu; (2) chia 2 GPU, mỗi GPU một worker. WER 1-best của e46f (không prompt, beam 5): held-out 6,28%; TTS v2 xử lý lại 4,98%; TTS v3 chuẩn 6,36%; **phương ngữ ElevenLabs Nam 13,73%, Trung 18,19%** (câu đúng hết 8,8% / 3,3%) → e46f yếu rõ với giọng vùng tổng hợp; nguồn nhiễu FC phương ngữ dồi dào. Bảng cụm e46f-TTS 738 cụm (gộp với v2: 1 078). **B6 bước 1:** held-out 1-best sai 175 câu; **55 câu có bản 2–5 đúng hết** (câu cuối 17/74); WER oracle N-best 4,45% vs 6,28% → A4 (5) "cách nghe khác" có dư địa thật (nếu FC học được cách chọn). A4 data đang dựng (`make_a4.sh`, log `agent/clef/data/make_a4.log`). B4b (train tiếp từ e46f, LR 1e-4, 1 epoch, thật ×3) đang chạy: call `fc-01M77F0E8H2NQEXNWVBT8NMJ47`.
**A4 data dựng xong (`agent/clef/data/fmtA_a4`, `make_a4.sh`):** 12 241 ví dụ (11 503 + 738 app lạ), 0 rò, 147/147 tool, thành phần giống v3 (điều kiện 35%, tự sửa 8,9%, hỏi lại 13,6%, ≥3 call 41,5%). Nhiễu: 7 718 câu từ bản nghe thật (b0 thật + e46f-TTS) + 4 296 bảng cụm, 4 011 mục sạch; bộ lọc giữ nguyên (loại 2 029 đổi số, 4 366 đổi giờ, 731 phủ định...). 1 024/5 005 câu cuối khác v3 (vd "cho bả nghen", "giờ báo thức" → "giờ bới thóc"; có cả đổi động từ "Đặt" → "Đã" — lỗi e46f thật, bộ lọc không chặn động từ). Kiểm nhãn: `audit_v3` 0 lỗi; `audit_gold` giống hệt v3 (G3_repeated_call 2 — đã biết từ A1). **Sẵn sàng train ở A4 (sau A2).**
**B4b KẾT QUẢ (12/11 tối) — không qua cổng, tệ hơn cả B4.** Train tiếp từ e46f, LR 1e-4, 1 epoch, thật ×3, cùng data v3 (Modal H100, ~45 phút kể cả 1 lần container khởi động lại). Train-eval epoch 1: 7,92%; loss train TB 2,38 (cao bất thường khi đi từ e46f → nghi head CTC khởi tạo ngẫu nhiên đẩy gradient vào encoder lúc đầu). Eval cuối (beam 5, điều kiện thi) 7,45%. Cổng: WER 6,96% (e46f 5,77), câu cuối 7,36 (6,47), lịch sử 6,60 (5,13), Bắc/Trung/Nam 6,59/8,76/5,88, câu đúng 235 (246), sai số 12 (9), thực thể sai 19 (12). → **1 epoch với data + công thức v3 làm e46f xấu đi ~1,2 điểm.** Nghi phạm: (1) TTS (~34% lượt), đặc biệt 600 câu ElevenLabs miền Trung chất lượng kém (e46f WER 17–20% vs 7% người Trung thật); (2) head CTC mới; (3) augment_v3. Chưa tách được; đối chứng rẻ nhất: e46f + chỉ audio thật, cùng công thức (~$1,5). **Quyết định đề xuất: dừng hướng ASR, giữ e46f**, dồn sang A2/A4 (FC). Chi phí ASR hôm nay trên Modal: B4 ~$7–8 + B4b ~$3 + suy luận/upload ~$1.

## 1z8. Bài học data 12/11 và kế hoạch data v4 (ASR) / A6 (FC)
**e46f (ASR v1, bản đang nộp) train trên gì:** `modal/manifest.json` — **4 631 câu audio THẬT của BTC** (1 422 hội thoại; 405 câu/123 hội thoại giữ riêng), **không có TTS**, prompt v1 (transcript lượt trước), LoRA r32, label smoothing 0,1, CTC 0,3 (`modal/modal_finetune_phowhisper.py`). Log train đã mất (workspace Modal cũ). Mọi bản có TTS (v2 medium, L2, B4, B4b) đều không vượt e46f.
**Vì sao data hôm nay "chưa chuẩn":** (1) ElevenLabs Flash v2.5 giọng miền Trung: e46f nghe sai 17–20% (người Trung thật: 7%) → giọng tổng hợp lơ lớ, không phải phương ngữ thật; (2) miền Nam Flash 13–14% cũng gấp 2,5× người Nam thật; (3) chưa có cổng kiểm từng nguồn TTS trước khi cho vào train; (4) B4b đổi 3 thứ cùng lúc (data, head CTC mới, aug_v3) nên không tách được nguyên nhân. Chủ repo: **không đẩy data sáng nay lên HF.**
**Kế hoạch:** B7 (ASR v4) và A6 (FC) trong PLAN §3 — cốt lõi: **đối chứng công thức trước**, **cổng WER từng nguồn TTS** (≤ 1,5× audio thật cùng vùng), thử `eleven_v4`/`v4_turbo` cho giọng vùng, TTS ≤ 25% lượt, bỏ/warm-up head CTC khi train tiếp từ e46f; FC: dựng lại bảng nhiễu A4 loại giọng Trung ElevenLabs. Ưu tiên vẫn là FC (A2/A4) → B7 chỉ khi còn ≥ 1 ngày.
**D4 — đẩy data FC lên HF (12/11 tối, từ myserver):** `fisherman611/fc-8099` `ffabd9a7…` (fmtA_v3, fmtA_a4, newapps_q1 + meta/report; meta đã thay đường dẫn scratchpad bằng `agent/clef/app_filter/pack_b`); `fisherman611/x-8eb9` `c9b6fb29…` (b0.base_medium.asr.json). Cả hai public, đã kiểm không có key/đường dẫn tạm. **Data ASR v3 không đẩy** (chưa chuẩn, theo chủ repo).

## 1z9. Phân tích lỗi e46f trên 405 held-out + câu hỏi "e46f chưa converge, train tiếp?" (12/11 tối)

**Nguồn:** `asr/agent/runs/asr/nbest_e46f/nbest_B.json` (phần `heldout`, beam 5, lấy bản 1); script `asr/agent/asr_errors_e46f.py`; báo cáo đầy đủ + phụ lục 71 câu: `asr/agent/runs/asr/REPORT_e46f_heldout_errors.md`. Chuẩn hoá local không đổi số → chữ (thiếu `vi_num`) nên WER 6,75% so với 5,77% của `gate_v3.json`; cấu trúc lỗi không đổi.

**Số liệu:** câu cuối 7,80% (123) / lịch sử 5,80% (282); Bắc 6,77 / Nam 5,66 / **Trung 7,97**. 218/405 đúng hết; 187 sai: 99 sai 1 từ, 49 sai 2, 39 sai ≥ 3. **Câu cuối: 44 đúng, 8 chỉ khác dạng viết/từ đệm, 71 sai thật (58%)** → đúng chỗ M2 tụt 76,5 → 63,5.

**Loại lỗi (257 cụm):** từ thường nghe nhầm 63,5% (`tạm dừng → tập trường`, `vặn → đặt`, `pháp chế → khách chị`); số 9,2% nhưng gần hết là dạng viết `9 ↔ chín`, **sai giá trị thật ≈ 0**; tên Latin/từ nước ngoài 8,9% (`Brightpath → Bia Pratt`, `Ryan Cooper → ra hàng khách với`, `Mọt Sách → họp khách`, `Nhã Văn → Chảo Phan`); mất từ 8,2% (phương ngữ Trung `rứa con`, `chi`, `nghe con`); thừa từ 5% (`train_1009` bịa cả câu đầu — nghi ảo giác từ prompt); từ đệm 5% (vô hại).

**Beam 2–5 bù:** 66/187 câu sai có bản đúng hẳn, 108/187 có bản tốt hơn; **câu cuối 15/71 đúng hẳn, 37/71 tốt hơn** (khớp B6 bước 1: 55 câu / câu cuối 17/74 với chuẩn hoá khác).

**Kết luận:** (1) lỗi đắt nhất là **thực thể tên riêng ở câu cuối** → tìm rỗng → chuỗi chết, N1/N1b không cứu (khớp §1s); (2) số không còn gì để sửa; (3) 63% "từ thường" không có cấu trúc → chỉ sửa bằng audio thật thêm, mà không có; (4) dư địa còn lại là **FC chịu lỗi 1–2 từ** (A2) và **FC thấy beam 5** (A4 (5)).

**"e46f chưa converge, train tiếp được không?" — Không, với căn cứ:**
- e46f dùng công thức `PLAN_PhoWhisper_finetune.md`: LoRA r32/α64, LR 3e-4, **tối đa 6 epoch, early stop patience 2**, giữ best. Log từng epoch của e46f đã mất (§1z7), nhưng cấu hình chị em `c3d9` (cùng data, cùng loss, chỉ thiếu prompt) **chạy đủ 6 epoch, eval phẳng từ epoch 3**; full fine-tune bị bỏ vì "loss đã bão hoà, capacity không phải nút thắt".
- Nút thắt là **4 631 câu thật** (6,8 h), không phải số epoch. Thêm epoch trên cùng data chỉ học thuộc (Qwen 0.6B 6 epoch: 13,2 → 16,3, §1n).
- Train tiếp đã thử 2 lần với data thêm: L2 (+2 epoch v2.1) → public 62.60 < 62.76; B4b (+1 epoch v3) → WER xấu đi 1,2 điểm. Cả hai đổi nhiều thứ cùng lúc nên không kết luận được "train tiếp trên data v1 với LR thấp" có hại hay không — đó là **B8**.
- **B8 (tuỳ chọn, $0):** e46f + chính data v1, không head CTC mới, không aug_v3, LR 2e-5, 2 epoch, patience 1, cổng `asr_gate_v3.py`. Kỳ vọng ≈ 0 điểm; giá trị là đóng dứt điểm câu hỏi và tách nguyên nhân B4b (data hay công thức). Chỉ chạy trên GPU miễn phí (Cheaha A100 / Kaggle) và **không chen A2**.

**Hạ tầng mới (12/11 tối):** vào được **Cheaha** (UAB) qua tunnel Cloudflare (`ssh cheaha-cf`, job Slurm `cf-tunne` giữ tunnel trên node CPU `c0154`, 12 h); SSH thẳng `cheaha.rc.uab.edu:22` bị timeout. A100: partition `amperenodes` (2 GPU/node, 12 h) và `amperenodes-medium` (2 ngày); lúc kiểm có 1 node idle, 18 node mix. Cách xin: `sbatch -p amperenodes --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 12:00:00 job.sh`. Dùng cho A2/A4/B8 thay Modal; cần đưa data `fmtA_v3` và model gốc `qwen35_4b_text` lên đó trước. Lịch mới ở PLAN §4b.
