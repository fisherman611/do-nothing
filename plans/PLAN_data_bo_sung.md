# Plan bổ sung dữ liệu cho ASR (PhoWhisper-large) – ZAC 2026 LifeMate

Cập nhật: 2026-11-07. Đi kèm `PLAN_PhoWhisper_finetune.md`. Code: `asr/modal_finetune_phowhisper.py`, phân tích lỗi: `asr/analysis/`.

## 0. Vì sao phải bổ sung dữ liệu

Bốn cấu hình loss (CE, label smoothing, weighted CE, CTC+attention) chênh nhau dưới 1 điểm WER và **sai cùng chỗ**. Loss đã bão hòa; phần lỗi còn lại là do model chưa từng nghe đủ kiểu dữ liệu đó. Kết quả trên val (405 utt, model tốt nhất LS 0.1 + CTC 0.3, WER 6.6%, WER sau quy đổi số/chính tả 5.9%):

| Loại lỗi | Số lượng | Ảnh hưởng tool call | Sửa được bằng |
|---|---|---|---|
| Từ nội dung sai (khoan → không, pin → biến, lấy → lén, chốt → chút) | ~125 câu, trong đó ~30–40 câu đổi nghĩa lệnh | Cao | **Thêm dữ liệu** (mục 2.1–2.3) |
| Tên riêng, thương hiệu (Greenhaven, Brightpath, Lemon Tree, Daniel Lee, Isabel Moreno, Nhã Văn) | 27/267 token tên sai; tên chưa thấy trong train sai 35% | Cao | Prompt từ history (sửa ~14/27) + **dữ liệu tên riêng** (mục 2.4) |
| Câu dài, nói tự nhiên, giọng vùng (rứa/nớ/mô/hông) bị nát cả đoạn | ~10–15 câu | Cao khi là lượt cuối | **Dữ liệu nói tự nhiên, nhiều vùng** (mục 2.2) |
| Rơi chữ số trong dãy dài (số điện thoại) | 8/489 token số | Cao nhưng hiếm | Dữ liệu số (mục 2.5) + kiểm tra độ dài ở hậu xử lý |
| Kiểu viết số 5 ↔ năm, chính tả hoá ↔ hóa | 14 câu | Không | Hậu xử lý, không cần dữ liệu |
| Từ đệm, đại từ (nhé ↔ nha, mình ↔ anh) | 38 câu | Không | Bỏ qua |

Dữ liệu hiện có: 5036 utt, 6.8h, 1545 hội thoại. Public test có 1933 audio **chưa nhãn**.

## 1. Nguyên tắc

- Dữ liệu bổ sung phải **giống phân phối thi**: câu lệnh cho trợ lý, giọng nói tự nhiên, có tên riêng/số/giờ, 3–20 giây. Dữ liệu đọc sách báo (VIVOS, Common Voice) chỉ giúp phần âm học, không giúp phần lệnh.
- Dữ liệu ngoài miền **không được lấn át** 6.8h trong miền. Giới hạn ≤ 2× lượng trong miền, hoặc train 2 giai đoạn: ngoài miền trước, trong miền sau (vẫn LoRA).
- Mọi bộ thêm vào phải đo trên **cùng val 405 utt** bằng `asr/analysis/compare_models.py` (WER quy đổi + lỗi tên + lỗi số), không chỉ WER thô.
- Kiểm tra luật BTC trước khi dùng dữ liệu ngoài (README cuộc thi chưa nói rõ; AGENT.md chỉ cấm đọc `state.json` và `*_environment/`).
- Giữ nguyên quy ước nhãn của BTC (số bằng chữ số lẫn chữ), thống nhất ở hậu xử lý.

## 2. Các nguồn dữ liệu, xếp theo thứ tự nên làm

### 2.1 Pseudo-label public test — **KHÔNG ĐƯỢC LÀM** (vi phạm luật)

- `AGENT.md`: "Không gán nhãn public_test hoặc private_test, bằng tay hay qua dịch vụ/model bên ngoài, để huấn luyện hoặc tinh chỉnh giải pháp." Pseudo-label bằng chính model của mình cũng là gán nhãn test để train → bỏ mục này.
- Được phép: **dữ liệu tổng hợp** (TTS, LLM sinh câu) ở giai đoạn phát triển, miễn inference offline. Các mục 2.2–2.5 vẫn hợp lệ.

### 2.2 Dữ liệu tiếng Việt nói tự nhiên, nhiều vùng miền (ưu tiên 2)

- Mục tiêu: câu nói đời thường, giọng Huế/Nam, có ngập ngừng, tự sửa lời. Val sai nặng đúng ở loại này.
- Nguồn công khai có thể dùng: Bud500 (`linhtran92/viet_bud500`, ~500h, nhiều vùng, nhiều chủ đề), VLSP ASR 2020/2021 (có phần nói tự nhiên), VietSpeech/VietMed nếu phù hợp. Common Voice vi và VIVOS là giọng đọc, chỉ dùng ít.
- Cách dùng: lọc câu 3–20 giây, ưu tiên câu có từ địa phương (rứa, nớ, mô, hông, nhứt, dùm) và câu dạng yêu cầu/câu hỏi; lấy 10–15h, không lấy hết.
- Train 2 giai đoạn: LoRA trên hỗn hợp (ngoài miền + trong miền) 2 epoch, rồi tiếp 3 epoch chỉ trong miền. So với bản chỉ trong miền trên cùng val.
- Rủi ro: lệch miền kéo WER trong miền lên; đo trước khi chốt.

### 2.3 Tăng cường trên dữ liệu có sẵn (ưu tiên 2, rẻ)

- Speed perturbation 0.9 / 1.0 / 1.1, thêm nhiễu nền (MUSAN hoặc nhiễu nhà/đường phố), âm vang phòng nhẹ, thay đổi âm lượng. Nhân 6.8h thành ~20h "mới" không cần nhãn mới.
- Nhắm vào overfit và câu dài bị nát. Không giúp tên riêng.
- Làm on-the-fly trong `DS.__getitem__` (torchaudio / audiomentations), không cần sinh file.

### 2.4 Dữ liệu tên riêng, thương hiệu, địa điểm (ưu tiên 1 cho tool call)

- Tên sai chủ yếu là tên nước ngoài đọc kiểu Việt (Greenhaven, Brightpath, Marigold, Isabel Moreno, Daniel Lee, Lemon Tree, Taipei, Edura, OrbitCloud). Model chưa thấy đủ cách người Việt phát âm các tên này.
- Lập danh sách thực thể từ train: tên người, tên quán, tên gói dịch vụ, kênh TV, tên app, hãng (lấy từ `audio_text` và lời assistant, không lấy từ `*_environment/`). Dự kiến vài trăm tên.
- Sinh câu lệnh có chứa tên bằng template từ các mẫu trong train ("nhắn cho X là…", "đặt xe tới quán X", "hủy gói X"), rồi:
  - (a) **TTS nhiều giọng** (giọng Bắc/Trung/Nam, nam/nữ) để có audio. Đây là cách chuẩn khi thiếu audio chứa tên; cần ít nhất 3–4 giọng và thêm nhiễu/speed để không bị "giọng máy". Giới hạn ≤ 2h để không kéo model về giọng TTS.
  - (b) Nếu có người: thu 1–2 giờ do thành viên nhóm đọc, ưu tiên hơn TTS.
- Kỳ vọng: giảm lỗi tên chưa thấy (hiện 35%) và lỗi tên gặp lần đầu ở lượt cuối (13/27 lỗi mà prompt không cứu được).
- Lưu ý: tên trong public/private test có thể khác tên trong train; TTS giúp model quen với **kiểu** tên nước ngoài, không chỉ tên cụ thể.

### 2.5 Số, giờ, tiền, số điện thoại (ưu tiên 3)

- Lỗi số hiện ít (1.5% token) nhưng mỗi lỗi làm hỏng tool call. Loại sai: rơi chữ số trong dãy 10 số.
- Sinh câu template có số điện thoại 10 chữ số, số tiền (nghìn/triệu), giờ (kém, rưỡi, 15 phút), ngày tháng; TTS như mục 2.4, trộn chung.
- Nhãn: theo đúng cách BTC đang ghi (số điện thoại viết bằng chữ từng số, tiền có thể số hoặc chữ) để không lệch quy ước.

### 2.6 Chưa cần

- Full fine-tune, LoRA r lớn hơn: capacity không phải nút thắt.
- Dữ liệu đọc (VIVOS, Common Voice) khối lượng lớn: sai miền.
- Text-only (chỉ transcript, không audio): Whisper không dùng được trực tiếp; chỉ có ích nếu làm rescoring bằng LM, hiện chưa làm.

## 3. Thứ tự thực hiện và cách đo

| # | Việc | Chi phí | Đo bằng |
|---|---|---|---|
| 1 | Train LS + CTC + prompt trên split hiện tại (code đã có) | ~2.5 USD | WER real-prompt vs 6.6%; lỗi tên vs 27 |
| 2 | ~~Pseudo-label public test~~ — bị cấm theo AGENT.md | – | – |
| 3 | Tăng cường on-the-fly (speed, nhiễu) | ~2.5 USD | Đường eval có bớt overfit không |
| 4 | Danh sách thực thể + TTS 1–2h câu có tên/số | 1 ngày làm + ~3 USD | Lỗi tên chưa thấy, lỗi số |
| 5 | Bud500/VLSP lọc 10–15h, train 2 giai đoạn | ~5 USD | Câu dài, giọng vùng |
| 6 | Đo end-to-end tool call với agent (chưa chọn agent) | – | Số thật để quyết định dừng ở đâu |

Mỗi bước giữ lại thư mục kết quả riêng trên `zaic-out` và đẩy lên HF như các bản trước. Chỉ gộp bước nào có cải thiện trên val vào bản train 100% data cuối.
