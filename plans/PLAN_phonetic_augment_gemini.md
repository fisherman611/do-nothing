# Plan augment phiên âm từ nước ngoài cho data FC (Gemini teamwork)

Cập nhật: 2026-11-09. Đi kèm `PLAN_data_bo_sung.md` (lỗi tên riêng của ASR) và `PLAN_FC_clef_qwen35.md`. Người thực hiện: nhóm agent Gemini. Người duyệt: chủ repo.

## 0. Mục tiêu và lý do

Lúc chạy thật, model FC đọc **output của ASR (PhoWhisper)** chứ không đọc transcript chuẩn của BTC. Trong `train.jsonl`, từ nước ngoài luôn được viết theo chữ gốc: SJC, RON95, VTV6, Zalo, email, Daniel Lee, Greenhaven Foods... ASR thì có thể ra **dạng phiên âm tiếng Việt** của các từ đó, nhất là tên riêng và thương hiệu lạ. Model FC chưa từng thấy dạng này.

Việc cần làm: với mỗi mẫu train có từ nước ngoài, sinh thêm bản câu trong đó các từ nước ngoài được viết bằng **phiên âm tiếng Việt chuẩn**. Phiên âm gồm các âm tiết tiếng Việt viết đúng chính tả, tức dạng mà một model ASR tiếng Việt có thể viết ra. **Nhãn tool call giữ nguyên** (mục 5), để model học cách hiểu từ phiên âm và vẫn gọi đúng bằng tên gốc.

Ví dụ:

| Gốc | Phiên âm chuẩn |
|---|---|
| Vàng SJC bao nhiêu một lượng? | Vàng ét giây xê bao nhiêu một lượng? |
| Bật kênh VTV6 giùm tôi | Bật kênh vê tê vê sáu giùm tôi |
| nhắn SMS cho anh Tuấn | nhắn ét em ét cho anh Tuấn |
| Nhắn Zalo cho chị Lan | Nhắn Da lô cho chị Lan |
| gửi email cho Ngân | gửi i meo cho Ngân |
| nhắc gặp Daniel Lee | nhắc gặp Đa ni en Li |
| tiền Greenhaven Foods mới chuyển | tiền Grin hây vần Phút mới chuyển |

**Quy mô ước tính:**
- Bộ lọc thô đếm được 733/1545 mẫu có từ nước ngoài ở ít nhất một lượt user, 490 mẫu có ở câu cuối.
- Bộ lọc này có nhầm các từ Việt không dấu như Con, Gia, Vy, Huy. Số thật có lẽ khoảng 500–650 mẫu.
- Chỉ augment mẫu thuộc split **train** (`asr/agent/clef/splits.json`, khoảng 518 mẫu theo bộ lọc thô).
- Mỗi mẫu tạo 1–2 biến thể, tức khoảng 700–1000 mẫu mới.

## 1. Luật (bắt buộc)

- **Chỉ dùng** `data/train+public_test/train/train.jsonl`.
- **Không mở `public_test/`** dưới bất kỳ hình thức nào. Không dùng transcript hay nhãn ASR của audio public_test, kể cả để tham khảo (AGENT.md dòng 24–27).
- **Chỉ augment mẫu có `id` thuộc `splits.json["train"]`.** Không augment `val_in` / `val_ood`, để val không bị rò rỉ.
- Không đọc `*_environment/`. Nhãn đã có trong `train.jsonl`: `label.tool_calls`, `tool_results`, `final_state`.

## 2. Đầu ra

```
synth_phonetic/
├── terms.json             danh mục từ nước ngoài + phiên âm (bước 1–2)
├── train_phonetic.jsonl   mẫu augment, cùng schema train.jsonl
└── reports/               QA (mục 6)
```

Mỗi dòng của `train_phonetic.jsonl` là bản sao của mẫu gốc. Chỉ các trường sau được thay đổi:

- `id`: `<id gốc>#ph1`, `<id gốc>#ph2`.
- `audio_text`: câu cuối, đã thay từ nước ngoài bằng phiên âm.
- `history[*].audio_text` của các lượt user: thay theo tỉ lệ ở mục 4.
- `aug`: trường mới, mô tả đã thay gì, ví dụ:

  ```json
  {"replacements": [{"turn": -1, "from": "Daniel Lee", "to": "Đa ni en Li", "kind": "person"}]}
  ```

`context`, `label` và mọi nội dung của assistant hay tool **không đổi một ký tự**.

## 3. Bước 1: lập danh mục từ nước ngoài (`terms.json`)

1. **Lọc tự động** trên mọi lượt user của các mẫu train. Bắt:
   - chữ in hoa, hoặc chữ kèm số (SJC, RON95, VTV6, E5, QR);
   - chữ có f/j/w/z;
   - tên riêng không dấu theo kiểu tiếng Anh;
   - danh sách từ tiếng Anh thông dụng: email, file, online, team, check, meeting, deadline, ok, app, workshop, pizza...
2. **Gemini duyệt danh sách:**
   - Loại từ Việt không dấu bị bắt nhầm (Con, Gia, Vy, Huy, Thu, Nay, Hay...).
   - Gán `kind` cho mỗi từ:

     | `kind` | Ví dụ |
     |---|---|
     | `acronym` | SJC, SMS, QR, VOV |
     | `code` | RON95, E5, RV301, mã đơn |
     | `channel` | VTV3, HTV7, S+ Sport |
     | `person` | Daniel Lee, Emily, Nakamura |
     | `brand` | Greenhaven Foods, MinhMart, JetXanh, Nexora, Zalo |
     | `english_word` | email, file, deadline, check |
     | `email_url` | dạng "a còng ... chấm ..." |
3. **Ghi theo cụm đầy đủ** (ví dụ "Greenhaven Foods", không tách riêng "Greenhaven") kèm số lần xuất hiện.
4. **Không phiên âm `email_url`:** BTC đã có quy ước riêng cho loại này, giữ nguyên.

## 4. Bước 2–3: phiên âm và tạo biến thể

### 4.1 Quy tắc phiên âm chuẩn

Viết thành **các âm tiết tiếng Việt cách nhau bằng dấu cách**, đúng chính tả, không gạch nối. Tên người và thương hiệu viết hoa chữ đầu mỗi âm tiết của tên. Từ thường viết chữ thường.

| kind | Cách phiên âm | Ví dụ |
|---|---|---|
| acronym | đọc từng chữ cái theo tên chữ cái tiếng Việt thông dụng; viết tắt đọc liền thành từ thì phiên âm như từ | SJC → ét giây xê; SMS → ét em ét; QR → kiu a; VOV → vê ô vê; ATM → a tê em |
| code | chữ cái đọc theo tên chữ, số đọc thành chữ | RON95 → rôn chín lăm; E5 → e năm; RV301 → a vê ba không một |
| channel | như acronym cộng số đọc thành chữ; phần tiếng Anh phiên âm | VTV6 → vê tê vê sáu; HTV7 → hát tê vê bảy; S+ Sport → ét cộng Xpót |
| person | phiên âm theo cách người Việt đọc tên đó | Daniel Lee → Đa ni en Li; Emily → Ê mi li; Nakamura → Na ca mu ra; Sophie → Xô phi |
| brand | phiên âm theo cách đọc tiếng Anh phổ biến ở Việt Nam; phần tiếng Việt giữ nguyên | Greenhaven Foods → Grin hây vần Phút; MinhMart → Minh Mát; Zalo → Da lô |
| english_word | phiên âm thông dụng | email → i meo; file → phai; online → on lai; team → tim; check → chéc; meeting → mít tinh; deadline → đét lai; ok → ô kê; pizza → pi da |

Ràng buộc:

- **1–2 cách phiên âm cho mỗi từ.** Cách thứ hai chỉ dùng khi có cách đọc phổ biến khác thật sự, ví dụ giây/gi cho J, ét/ết cho S. Không bịa thêm.
- **Âm tiết hợp lệ:** mỗi âm tiết phải là âm tiết tiếng Việt viết đúng chính tả. Ngoại lệ là các cụm phụ âm đầu kiểu "Gr", "Xp" khi đó là cách viết phiên âm thông dụng.
- **Không trùng tên khác:** cách phiên âm không được trùng với tên khác trong state, history hay kết quả tool của chính mẫu đó (mục 6.4).

### 4.2 Tạo biến thể

- **`#ph1`:** thay **mọi** từ nước ngoài ở câu cuối bằng cách phiên âm thứ nhất. Mỗi từ nước ngoài trong history bị thay với xác suất 0.3.
- **`#ph2`:** chỉ tạo khi mẫu có ≥ 2 từ nước ngoài, hoặc có từ có 2 cách phiên âm. Thay **ngẫu nhiên 1–2 từ** (cách phiên âm thứ hai nếu có), giữ nguyên các từ còn lại. Mục đích là để model gặp câu lẫn cả chữ gốc lẫn phiên âm.
- **Chỉ thay từ nước ngoài.** Không viết lại phần khác của câu. Đây không phải paraphrase.

## 5. Bước 4: xử lý nhãn

- **`label.tool_calls` giữ nguyên,** kể cả tham số có chứa từ nước ngoài. Model phải học từ "Đa ni en Li" gọi `contacts.search` với `query` "Daniel Lee", từ "vê tê vê sáu" gọi `tv.tune` với `channel` "VTV6". Tên chuẩn phải lấy từ:
  - catalog, ví dụ enum kênh;
  - kết quả tool, ví dụ danh bạ;
  - hiểu biết chung.
- **Tham số tự do** (nội dung tin nhắn, tiêu đề lịch, ví dụ `sms.send.content` = "mai gửi mình file nhé") giữ giá trị **chuẩn** như nhãn gốc.
- **Trường hợp không suy ra được:** nếu từ phiên âm thì một người đọc kỹ cũng không đoán lại được từ gốc, và từ gốc không xuất hiện ở đâu khác mà model nhìn thấy (history, kết quả tool), thì **bỏ thay thế đó**. Ví dụ là thương hiệu tự đặt rất lạ ở tham số tìm kiếm. Không sửa nhãn cho khớp với phiên âm.

## 6. QA (tự động, bắt buộc)

Script `synth_phonetic/qa.py`, báo cáo ghi vào `reports/`:

1. **Diff:** mỗi dòng chỉ khác mẫu gốc ở các trường cho phép (mục 2). `label`, `context` và các lượt assistant/tool phải trùng y hệt (so hash).
2. **Danh sách thay thế:** mọi đoạn khác nhau giữa câu gốc và câu mới phải có trong `aug.replacements`, và `to` phải có trong `terms.json`.
3. **Âm tiết hợp lệ:** mọi âm tiết trong `to` qua bộ kiểm tra âm tiết tiếng Việt (bảng phụ âm đầu, vần, thanh). Ngoại lệ phải nằm trong danh sách cho phép ở `terms.json`.
4. **Không trùng tên khác:** không trùng tên nào trong state, history hay kết quả tool của mẫu.
5. **Giám khảo LLM** (model khác với model sinh) chấm 1–5 điểm. Bỏ biến thể dưới 4 ở bất kỳ tiêu chí nào.
   - Phiên âm có đúng cách người Việt đọc từ đó không.
   - Người đọc kỹ có suy ra được từ gốc và làm đúng yêu cầu không.
6. **Chống rò rỉ:**
   - không có `id` thuộc `val_in` / `val_ood`;
   - grep không thấy `public_test`.
7. **Duyệt tay:** chủ repo duyệt 50 biến thể ngẫu nhiên. Ngưỡng ≤ 3 lỗi.

## 7. Phân công nhóm Gemini

| Vai | Việc | Đầu ra | Bàn giao cho |
|---|---|---|---|
| Term miner | Bước 1: lọc, duyệt, gán `kind` | `terms.json` chưa có phiên âm | Phonetizer |
| Phonetizer | Bước 2: 1–2 cách phiên âm mỗi từ theo mục 4.1 | `terms.json` đầy đủ | Augmenter, Reviewer |
| Augmenter | Bước 3–4: tạo `#ph1` / `#ph2`, ghi `aug`, bỏ thay thế không suy ra được | `train_phonetic.jsonl` nháp | Verifier |
| Verifier | Mục 6.1–6.4 và 6.6, loại biến thể lỗi | `train_phonetic.jsonl` sạch, `reports/` | Chủ repo |
| Reviewer (model khác) | Mục 6.5, review `terms.json` | điểm, nhận xét | Verifier |

Thứ tự:

1. Bước 1–2 trên toàn bộ từ.
2. Đợt thử 30 mẫu qua bước 3–4 và QA. Chủ repo duyệt.
3. Chạy toàn bộ.
4. Tích hợp:
   - `build_data.py` đọc thêm `train_phonetic.jsonl` như mẫu train (cờ `--phonetic`). Mẫu `#ph*` theo split của `id` gốc.
   - Build lại data, train Clef, so sánh val_in / val_ood có và không có augment.

## 8. Tiêu chí chấp nhận

- `terms.json` phủ ≥ 95% từ nước ngoài thật trong các mẫu train (Reviewer kiểm 100 câu ngẫu nhiên).
- QA mục 6.1–6.4 và 6.6 không có lỗi nào. Giám khảo LLM: ≥ 90% biến thể đạt ≥ 4. Duyệt tay ≤ 3/50 lỗi.
- Báo cáo có:
  - số mẫu gốc được augment;
  - số biến thể;
  - số thay thế bị bỏ vì không suy ra được;
  - phân bố theo `kind`.

## 9. Rủi ro

- **Phiên âm không thống nhất giữa các mẫu,** ví dụ cùng "Daniel" lúc "Đa ni en", lúc "Đen ni eo". Xử lý: mọi phiên âm đều lấy từ `terms.json`, không sinh tự do trong từng mẫu.
- **Model học đoán ngược quá đà,** tự đổi những từ Việt bình thường thành tên nước ngoài. Xử lý:
  - giữ nguyên mẫu gốc trong train;
  - tỉ lệ mẫu gốc : biến thể khoảng 1 : 1.5 trên phần có từ nước ngoài;
  - kiểm tra val_in không giảm.
