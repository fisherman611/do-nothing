# PLAN: sinh audio train ASR bằng TTS từ (một phần) data text của FC

Ngày: 2026-11-10. Cụ thể hoá `PLAN_data_augment_v3.md` §4 (Việc D) cho nguồn text đã có sẵn trong repo FC. Train ASR theo
`PLAN_PhoWhisper_finetune.md` / `PLAN_ASR_bias_v2.md` (`asr/asr_train.py`, `asr/modal_finetune_phowhisper.py`).
Luật: không dùng câu nào của public/private test; audio tổng hợp **chỉ vào train**, không vào val/test.

## 0. Mục tiêu

PhoWhisper-large fine-tune v1 đạt WER 6.2% trên val 405 utt **cùng phân phối train**. Lỗi còn lại và rủi ro ở test nằm ở:
tên nước ngoài/thương hiệu, số (tiền, điện thoại, giờ), từ của 11 app lạ. Train audio chỉ có ~6.8 h / 5036 utt. Mục tiêu:
thêm **8–15 h audio tổng hợp**, ≥ 60 giọng, tập trung đúng ba nhóm lỗi trên; đo bằng val thật (WER, CER, lỗi tên, lỗi số).

## 1. Nguồn text: cái gì dùng được, cái gì không (đếm 2026-11-10)

| Nguồn | Câu (kể cả history) | Câu **mới** khác train | Dùng cho ASR? |
|---|---|---|---|
| `synth_phonetic/train_phonetic.jsonl` (#ph1/#ph2) | 4299 | **985** (~19.5k từ ≈ **1.9 h**) | **Có, giá trị nhất**: cho cặp (dạng nói, dạng viết) — xem §2 |
| `synth/train_alias.jsonl`, `train_iface.jsonl`, `val_alias.jsonl` | 6032 | **0** | **Không**: chỉ đổi tên tool/catalog, lời người dùng y hệt train |
| B2 / test-like của Gemini (khi có) | ~3000 hội thoại | tất cả mới | Có: lời cho app lạ, từ vựng lĩnh vực mới |
| Câu theo mẫu có ô thực thể (sinh bằng code + LLM, `PLAN_data_augment_v3` §4.2.1) | tuỳ | mới | Có: tên, số điện thoại, tiền, giờ — lặp mỗi thực thể hiếm 20–50 lần |
| Viết lại 5036 transcript train (LLM) | 5036 | mới về diễn đạt | Có, nhưng xếp sau: không thêm thực thể mới |
| 652/1545 câu train có số/giờ/tiền | – | – | Dùng làm **mẫu** để sinh biến thể cách đọc số |

Kết luận: trong data FC hiện có, **chỉ phonetic cho text mới**; alias/iface không có gì cho ASR. Phần lớn khối lượng (8–15 h)
phải đến từ B2, câu mẫu thực thể, và viết lại.

## 2. Điểm then chốt: phonetic cho đúng cặp (nói, viết) mà ASR cần

Nhãn ASR của BTC viết từ nước ngoài theo chữ chuẩn (`SMS`, `Zalo`, `Deluxe`); người nói đọc kiểu Việt (`ét em ét`, `Da lô`,
`đê lúc`). Biến thể #ph1/#ph2 có đúng hai dạng đó cho cùng một câu (`aug.replacements` nối từ gốc ↔ phiên âm):

- **Text đưa vào TTS** = câu phiên âm (`ét em ét`) → TTS đọc đúng cách người Việt nói.
- **Nhãn ASR** = câu gốc chữ chuẩn (`SMS`) → ASR học "nghe *ét em ét* thì viết *SMS*".
- Với số: text TTS đổi `120000` → "một trăm hai mươi nghìn" (bộ chuẩn hoá số tiếng Việt, PLAN_v2 §C), nhãn giữ quy ước
  BTC (train viết lẫn chữ và chữ số → giữ như câu gốc).

Đây là lợi ích riêng của nguồn phonetic mà viết lại câu bằng LLM không có. Mỗi câu phonetic đọc bằng 3–5 giọng khác nhau
→ 985 câu × 4 giọng ≈ 7.5 h từ riêng nguồn này.

## 3. TTS

**Quyết định (2026-11-10): dùng TTS API đã có sẵn của chủ repo** (key do chủ repo giữ, không để trong repo; đặt qua biến
môi trường `TTS_API_KEY`). Script sinh audio (`asr/tts/synth.py`) viết theo giao diện chung `tts(text, voice, speed) → wav 16 kHz`
để đổi engine không phải sửa pipeline. Các việc còn lại với engine này:

- Lấy **danh sách giọng** của API (id, giới tính, vùng miền); mục tiêu ≥ 20 giọng từ API, bù bằng tốc độ 0.9/1.0/1.1 và augment
  âm thanh (§4.5) để đa dạng. Ghi `voice` vào manifest để sau lọc theo giọng.
- Kiểm **giới hạn gọi** (ký tự/tháng, request/phút) và **chi phí**: đợt 1 ≈ 985 câu × 4 giọng + 2000 câu × 2 giọng ≈ 8000 câu
  × ~110 ký tự ≈ 0.9 triệu ký tự. Cache theo hash(text, voice, speed) để không gọi lại.
- **Kiểm license đầu ra** của API cho việc train model (đọc ToS một lần, ghi kết luận vào đây). Audio chỉ dùng lúc phát triển,
  không có trong Docker nộp.
- Nghe thử 30–50 câu có tên nước ngoài phiên âm và số đọc bằng chữ trước khi sinh hàng loạt: nếu API tự "sửa" phiên âm về
  cách đọc tiếng Anh thì thử viết phiên âm có gạch nối hoặc tách âm tiết.

Dự phòng nếu API thiếu giọng hoặc hết quota: VieNeu-TTS v3 Turbo (Apache-2.0 theo `PLAN_data_augment_v3`, kiểm model card),
F5-TTS-Vietnamese (CC-BY-NC-SA, chỉ cho dữ liệu phát triển). Không clone giọng từ VIVOS/viVoice/PhoAudiobook (NC); clone từ
người quen phải có đồng ý.

## 4. Quy trình

1. **Chọn text** (script `asr/tts/select_texts.py`):
   - 985 câu phonetic mới (cả lượt cuối và history), kèm nhãn chữ chuẩn từ `aug.replacements`;
   - câu mẫu thực thể: danh sách tên (từ `terms.json` + tên trong `tool_results` train + tên Gemini sinh), số điện thoại
     10 số, tiền (nghìn/triệu/rưỡi), giờ (kém/rưỡi/15), ngày; mỗi thực thể hiếm 20–50 lần, mỗi câu 1–2 thực thể;
   - B2 khi có; viết lại transcript train (xếp sau).
   - Loại trùng với train và với nhau; **không lấy gì từ public_test**.
2. **Chuẩn hoá cho TTS**: số/giờ/tiền → chữ nói (nhiều cách đọc: "hai trăm", "hai trăm nghìn", "200k" đọc "hai trăm ka");
   từ nước ngoài → dạng phiên âm; chèn ngập ngừng/tự sửa ("à không, bảy rưỡi") ở ~15% câu. Nhãn ASR giữ dạng viết.
3. **Sinh audio**: mỗi câu 2–5 giọng ngẫu nhiên; 16 kHz mono; tốc độ 0.9/1.0/1.1.
4. **Lọc vòng ngược** (bắt buộc): chạy PhoWhisper v1 trên audio sinh ra; bỏ câu có CER > 10% so với **dạng nói** hoặc mất
   thực thể; ghi tỉ lệ loại theo engine/giọng (giọng nào bị loại > 30% thì bỏ giọng).
5. **Augment âm thanh** lúc train: nhiễu 5–20 dB SNR, RIR, codec điện thoại/Opus, SpecAugment (đã có trong v1).
6. **Manifest**: cùng schema với `asr/make_manifest_v2.py`, thêm trường `source` (`phonetic|entity|b2|rewrite`),
   `engine`, `voice`, `spoken_text`, `label_text`. Audio để ở volume Modal `zaic-data/synth_audio/`.

## 5. Train ASR

- Recipe v1 (`PLAN_ASR_bias_v2` §1): LoRA r=32/α=64 trên q/k/v/out_proj + fc1/fc2, LR 3e-4, LS 0.1, CTC 0.3, early stop
  theo val WER; prompt = transcript các lượt user trước + text assistant.
- Trộn **thật : tổng hợp = 1 : 1 → 1 : 2**, không quá 1 : 3; lặp dữ liệu thật; **epoch cuối chỉ dữ liệu thật**.
- LoRA decoder cho phần tổng hợp hoặc LR encoder thấp hơn, để encoder không học "âm thanh TTS".
- Val: 405 utt thật của v1, **không có audio tổng hợp**. Thêm chỉ số: lỗi tên (token), lỗi chuỗi số, tách lượt cuối/history,
  và WER trên tập câu phonetic thật trong train (các câu train có từ nước ngoài) để thấy tác dụng trực tiếp của §2.
- Chọn checkpoint theo val thật. Sau đó chạy `clef_agent.py --asr` với transcript của model mới trên val_in/val_ood để đo
  tác động lên điểm tool-call (đây mới là điểm thi).

## 6. Dùng lại cho FC (PLAN_data_augment_v3 §4.3)

Transcript của ASR mới trên audio train + audio tổng hợp = lời người dùng **có lỗi ASR thật**. Dùng ở 30–50% lượt khi build
data FC v2 (`build_data.py --asr <json>`), tham số vàng giữ dạng chuẩn → model FC học sửa lỗi ASR. Đây cũng là nguồn
query cho bộ lọc app (`PLAN_app_filter_finetune` §2.0 đợt 3).

## 7. Quy mô, chi phí, thứ tự

| Bước | Khối lượng | Ước |
|---|---|---|
| T1 | TTS API có sẵn: lấy danh sách giọng, kiểm quota/chi phí/ToS, viết `asr/tts/synth.py` (giao diện chung + cache), nghe thử 30–50 câu | nửa ngày |
| T2 | `select_texts.py` + chuẩn hoá số/phiên âm (dùng chung bộ chuẩn hoá của PLAN_v2 §C) | 1 ngày |
| T3 | Sinh audio đợt 1 qua API: 985 câu phonetic × 4 giọng (~7.5 h) + 2000 câu thực thể × 2 giọng (~3 h) ≈ 8000 câu, ~0.9 triệu ký tự | giới hạn bởi quota API; chạy song song 4–8 request, có cache |
| T4 | Lọc vòng ngược bằng PhoWhisper v1 | ~20 phút GPU |
| T5 | Train ASR v2 (thật + tổng hợp 1:1), so với v1 trên val thật; đo tool-call với `--asr` | ~2–3 h A100, ~$8 |
| T6 | Đợt 2 khi có B2: thêm lời app lạ, train lại | sau C1 |

Tiêu chí chấp nhận v2: WER val ≤ v1 (6.2%), **lỗi tên và lỗi chuỗi số giảm ≥ 20% tương đối**, điểm tool-call trên bản
ASR không giảm. Nếu WER tăng mà lỗi tên giảm: xem lại tỉ lệ trộn và bước "epoch cuối chỉ dữ liệu thật".

Rủi ro: TTS đọc sai chính tên nước ngoài cần dạy (kiểm bằng lọc vòng ngược và nghe mẫu 50 câu); model học giọng máy (giới
hạn tỉ lệ trộn, nhiều giọng, augment âm thanh); API ít giọng hơn mục tiêu ≥ 60 (bù bằng tốc độ + augment, hoặc thêm engine
dự phòng); quota/ToS của API (T1 phải xong trước T3).
