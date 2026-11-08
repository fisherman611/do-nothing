# Plan dữ liệu v3: tập test ASR, tool tổng hợp, audio tổng hợp, và các bổ sung để ASR + FC generalize

Cập nhật: 2026-11-08. Đi kèm `PLAN_ASR_bias_v2.md`, `PLAN_FC_clef_qwen35.md`, `PLAN_data_bo_sung.md`, `PLAN_next_steps.md`. Survey nguồn ở mục 9.

## 0. Mục tiêu và luật

Mục tiêu: ASR và FC chạy tốt trên **app, tool và tên riêng chưa thấy** (public test có 11 app lạ, 410/634 mẫu có cài ít nhất một app như vậy; private test có thể có app lạ khác).

| Việc | Hợp lệ? | Ghi chú |
|---|---|---|
| Gắn nhãn ASR public_test (Qwen + duyệt tay) **chỉ để đánh giá** | Có | Không train, không chọn checkpoint/siêu tham số theo tập này (AGENT.md: "để huấn luyện hoặc tinh chỉnh") |
| Sinh tool/app mới tự thiết kế + hội thoại tổng hợp để train | Có | AGENT.md cho phép dữ liệu tổng hợp lúc phát triển, miễn inference offline |
| Dùng TTS / LLM / API ngoài để sinh dữ liệu train | Có | Chỉ lúc phát triển, không có trong Docker. Kiểm license đầu ra (mục 4.1) |
| **Sinh dữ liệu train từ nhãn tập test** (kể cả viết biến thể) | **Không** | AGENT.md dòng 25, 27: không biến nhãn test thành dữ liệu train/tuning. Thay bằng nguồn ở mục 2–3 |
| Sinh dữ liệu train từ **catalog / code app của public_test** (`public_test/zaic_tools/apps/*`, `public_test_tools_catalog.json`) | **Vùng xám, mặc định không** | Đó là input chứ không phải nhãn, nhưng là tối ưu theo chính tập test. Giữ ngoài thì public test còn là thước đo thật cho app lạ (proxy của private test). Nếu muốn dùng: hỏi BTC trước |
| Đọc `*_environment/` / `get_state` của test | Không (trừ qua `ToolEnv` để gắn nhãn đánh giá) | Lượt gắn nhãn Qwen dùng `get_state` của public_test để lấy hotword; kết quả đó **chỉ** dùng để đánh giá |

## 1. Việc A: tập test ASR (đang làm)

Hiện trạng: 1.933 audio public_test có nhãn Qwen (`data/results/asr/qwen-cloud/public-test-asr-flash/`), 56 câu duyệt tay, UI `asr/review_ui.py`.

Số đã đo (2026-11-08): nhãn Qwen có WER **~25%** trên 100 câu train (nhãn thật), hotword không giúp (25.4% → 25.5%). PhoWhisper v1 của mình đạt **6.2%** trên val. Vì vậy nhãn Qwen chưa duyệt **không dùng để chấm PhoWhisper được**: sai số của nhãn lớn hơn sai số của model.

Kế hoạch:
1. **Duyệt tay là bắt buộc.** Chỉ dùng câu đã duyệt làm tập test.
2. **Chọn câu nào để duyệt:**
   - Ưu tiên câu lượt cuối, vì đó là thứ LLM dùng để gọi tool.
   - Lấy một mẫu **ngẫu nhiên** khoảng 300 câu, để ước lượng không thiên lệch.
   - Thêm các câu mà Qwen cũ, Qwen mới và PhoWhisper v1 lệch nhau về tên riêng hoặc số.
3. **Tránh neo theo nhãn máy:** UI hiện cả 3 hypothesis cạnh nhau, tô phần khác nhau; người duyệt nghe lại audio rồi tự gõ khi các bản lệch nhau.
4. **Quy ước nhãn** theo BTC (số viết lẫn chữ và chữ số như train). Chấm bằng chuẩn hoá tiếng Việt: số ↔ chữ, hoá ↔ hóa, bỏ dấu câu. Báo cả WER thô và WER chuẩn hoá.
5. **Chỉ số:** WER, CER, lỗi tên riêng, lỗi chuỗi số (đặc biệt số điện thoại), tách theo lượt cuối và lượt history.
6. **Vai trò:** chỉ báo cáo cuối và kiểm tra lệch phân phối train → test. Chọn model vẫn theo val 405 utt của train.

## 2. Việc B: app và tool tổng hợp (EnvScaler / AgentScaler)

Ý tưởng: tool mới phải **chạy được thật** để có kết quả tool và `state_hash` đúng, không để LLM bịa kết quả. `zaic_tools` có sẵn khung `App` (`apps/base.py`: `read`, `create`, `find`, `noop`, `@read/@write`). Một app như `parking.py` chỉ khoảng 50 dòng.

1. **Package riêng `synth_tools/`:** copy `train/zaic_tools` (runtime giống bộ chấm), thêm thư mục `apps_synth/`. Mỗi app gồm:
   - catalog JSON đúng format (`id, name_vi, group, description, tools[{name, kind, description, parameters, returns}]`, dùng `x-ref`, `enum`, `x-one-of` như catalog thật);
   - class Python trên `App`;
   - hàm sinh state ban đầu (danh sách bản ghi tiếng Việt, id dạng `prefix_NN`).
2. **Danh sách lĩnh vực:**
   - Dựng cây lĩnh vực đời sống Việt Nam: y tế, giáo dục, giao thông, nhà cửa, mua sắm, giải trí, tài chính, công việc, thể thao, du lịch, hành chính công…
   - Mỗi lĩnh vực nhờ LLM đề xuất 3–8 app, mỗi app 2–6 tool (catalog thật: trung bình khoảng 3,2 tool/app).
   - **Không lấy tên hay lĩnh vực theo 11 app của public test.** Kiểm bằng một danh sách loại trừ tự động.
3. **Độ phức tạp vừa phải** (ToolACE: dữ liệu độ khó trung bình tốt nhất). Cần đủ các kiểu tool như dữ liệu thật:
   - read có lọc, write cần id lấy từ một read trước (`x-ref`);
   - tham số enum, tham số tuỳ chọn, nhóm `x-one-of`;
   - trường hợp "noop": id không tồn tại, hết số dư…
4. **Kiểm tra tự động** (EnvScaler):
   - AST và lint;
   - agent tester gọi các call đúng và sai mà không nhìn code;
   - checker so kết quả với code và diff của state;
   - bỏ app có tỉ lệ qua < 0,85.
5. **Quy mô:** 60–120 app (EnvScaler thấy phần tăng lớn nhất nằm ở khoảng 20 môi trường đầu), chi phí khoảng 1 USD/app khi sinh bằng LLM API.
6. **Dùng app tổng hợp để mở rộng tập test FC:** giữ khoảng 15 app tổng hợp làm **val_synth_ood**. Tập này cùng với val_ood (app thật bị giữ lại) là thước đo generalize.

## 3. Việc C: hội thoại và trajectory tổng hợp (APIGen-MT)

Thay cho "augment từ nhãn tập test": augment từ **nhãn train + app tổng hợp**.

1. **Blueprint:**
   - Đi ngẫu nhiên trên đồ thị phụ thuộc tool (read → write theo `x-ref`), ưu tiên write. Ghép thêm persona (`user_profile` kiểu Việt Nam), `current_time` và tập app đã cài (trộn app thật của train với app tổng hợp).
   - Kết quả: yêu cầu người dùng, chuỗi call vàng, và state cuối có được khi **chạy thật** chuỗi call.
2. **Lọc:**
   - format, call hợp lệ theo `validate`;
   - chạy lại chuỗi call ra đúng state cuối;
   - hội đồng LLM chấm xem yêu cầu có khớp chuỗi call không, sửa tối đa 3 vòng (APIGen-MT: tỉ lệ blueprint qua lọc tăng từ 28% lên 70%).
3. **Nhiều lượt:** có history như dữ liệu thật (lượt assistant có `tool_calls` + `tool_results`). Ghép ngược các nhiệm vụ đã kiểm thành nhiệm vụ dài hơn (reverse recombination).
4. **Lời người dùng tiếng Việt nói tự nhiên:**
   - Sinh bằng LLM theo phong cách `audio_text` của train: khẩu ngữ, "nhé/nha", tự sửa lời, tên riêng, số nói bằng chữ.
   - Mỗi yêu cầu 2–3 cách nói, đây là "biến thể nhãn".
5. **Dữ liệu âm, khoảng 10%** (Hammer, When2Call):
   - thiếu thông tin bắt buộc → `sys.ask_missing_info`;
   - không có tool phù hợp, hoặc bỏ tool đúng khỏi catalog → không gọi gì;
   - yêu cầu ngoài phạm vi.
6. **Augment trên 1.545 mẫu thật** (rẻ, làm trước):
   - đổi tên hàm và tên tham số thành chuỗi ngẫu nhiên ở 30–50% mẫu (Hammer function masking: Qwen 4B tăng F1 trung bình 42 → 71);
   - viết lại mô tả tool;
   - xáo thứ tự catalog;
   - thêm 3–10 tool nhiễu gần giống;
   - đổi tên riêng và số trong yêu cầu, đồng thời sửa tham số vàng cho khớp.
7. **Tỉ lệ trộn:** chưa có số công bố, phải chỉnh theo val_ood. Bắt đầu với thật : tổng hợp ≈ 1 : 1, rồi thử 1 : 2.

## 4. Việc D: audio tổng hợp

### 4.1 Chọn TTS

| Model | Dùng cho | License đầu ra | Ghi chú |
|---|---|---|---|
| **VieNeu-TTS v3 Turbo** | **Chính** | Apache-2.0; clone giọng người thật cần có đồng ý | 25 giọng có sẵn Bắc/Trung/Nam (Trung chỉ 2), clone từ 3–8 s, nhanh (RTF ~0,01–0,02). Có chuẩn hoá số (sea-g2p) |
| F5-TTS-Vietnamese | Engine thứ hai (trộn để không học "giọng máy" của một engine) | CC-BY-NC-SA (phi thương mại) | Clone tốt; cần transcript của audio mẫu |
| MiniMax Speech, Viettel AI, FPT.AI (API) | Tuỳ chọn, thêm đa dạng giọng | Phải đọc ToS trước | Viettel 16 giọng ba miền |
| ElevenLabs | **Không dùng** | Điều khoản được cho là cấm dùng đầu ra để train ML (chưa xác minh) | – |
| CosyVoice 3, Fish-Speech, Spark-TTS, Index-TTS | Không | – | Không có tiếng Việt |
| Model omni (Qwen3-Omni…) | **Không dùng để sinh audio** (đầu ra giọng nói được cho là chưa có tiếng Việt, chưa xác minh). Dùng được cho phần text: viết lại câu, thêm ngập ngừng | – | – |

Giọng mẫu để clone: ưu tiên ghi âm 30–60 người quen có đồng ý, mỗi người khoảng 10 s, đủ ba miền. Common Voice vi là CC0. VIVOS, viVoice, PhoAudiobook là NC/nghiên cứu: không công bố audio clone hay embedding giọng.

### 4.2 Sinh audio cho ASR

1. **Text:**
   - Lời người dùng của mục 3 (app tổng hợp + augment) và bản viết lại của 5.036 transcript train.
   - Câu theo mẫu có ô thực thể: tên người, tên nước ngoài đọc kiểu Việt, thương hiệu, số điện thoại, giờ, tiền. Mỗi thực thể hiếm xuất hiện 20–50 lần.
   - **Không dùng câu của tập test.**
2. **Chuẩn hoá trước khi đưa vào TTS:** số, giờ, tiền đổi sang chữ nói. Nhãn ASR vẫn giữ quy ước của BTC.
3. **Biến thể nhãn từ một ý:**
   - viết lại câu;
   - chèn ngập ngừng và tự sửa lời ("à không, 7 giờ");
   - các cách đọc số khác nhau (đọc từng số hay theo nhóm; "2 triệu rưỡi" hay "2.500.000 đồng");
   - mỗi biến thể đọc bằng giọng khác nhau.
4. **Lọc vòng ngược:** chạy PhoWhisper trên audio sinh ra. Bỏ câu có CER > 10% hoặc mất thực thể (lọc kiểu này: WER MLS 13,5% → 6,9%).
5. **Quy mô và trộn:** 10–20 h, ≥ 100 giọng (nhiều giọng hơn là nhiều giờ trên mỗi giọng). Thật : tổng hợp ≈ 1 : 1 đến 1 : 2, không quá 1 : 3, lặp lại dữ liệu thật nhiều hơn.
6. **Augment trên audio tổng hợp:** tốc độ 0,9/1,0/1,1, nhiễu 5–20 dB SNR, RIR, codec điện thoại/Opus, SpecAugment.
7. **Train ASR:**
   - LoRA, giảm LR của encoder (hoặc chỉ LoRA decoder cho phần tổng hợp) để encoder không học đặc điểm âm thanh của TTS;
   - epoch cuối chỉ dùng dữ liệu thật;
   - chọn checkpoint theo WER và lỗi tên trên val thật. **Không đưa audio tổng hợp vào val hay test.**
   - Kỳ vọng theo các bài: WER giảm 5–15% tương đối, lỗi tên hiếm giảm 30–57% tương đối.

### 4.3 Dùng cùng audio đó cho FC

Đây là phần "một audio sinh ra dùng cho cả hai" của bạn:
- Chạy **ASR của chính mình** (PhoWhisper v1/v2) trên audio tổng hợp, và trên audio train nếu chưa có, để có **transcript có lỗi ASR thật**.
- Dùng transcript đó làm lời người dùng khi train FC, nhưng giữ tham số vàng ở dạng chuẩn để model học sửa lỗi.
- Chỉ dùng transcript lỗi ở 30–50% lượt. Với tool calling, 40–57% lỗi "text đúng, voice sai" là sai giá trị tham số.

## 5. Bổ sung cho model ASR

| # | Việc | Lý do |
|---|---|---|
| 1 | **Sửa bug sắp xếp uid** trong `transcribe_conv` (`u10` xếp trước `u2`) | Ảnh hưởng 50 utt public_test và lúc nộp |
| 2 | **Tăng tên nhiễu trong prompt** theo B-Whisper: hiện 3–10 tên, B-Whisper dùng 25–150 tên, xác suất không có tên đúng 0,3, bỏ cả prompt 0,2, trọng số CE 1,1 cho token tên | B-Whisper: lỗi từ hiếm giảm 45,6% tương đối, từ chưa thấy giảm 60,8%. Ngân sách 70 token hiện tại chỉ chứa khoảng 15–20 tên, nên cần thử nới ngân sách phần tên (giảm phần hội thoại) |
| 3 | **Chỉ số tên chưa thấy + báo động giả:** tách val theo tên, đo nhận đúng tên có trong danh sách và tỉ lệ chép nhầm khi danh sách chỉ có tên nhiễu | Không có thì không đo được generalize |
| 4 | **Sửa tên sau ASR bằng độ giống phát âm** với danh sách tên trong `tool_results` của history và hồ sơ, chỉ sửa khi độ tin thấp | Rẻ, tất định; bài của Apple: lỗi thực thể giảm 73,6% tương đối |
| 5 | **Tăng tốc:** faster-whisper / CTranslate2 fp16 (int8 phải đo lại WER), batch các audio history, prompt ngắn cho history và prompt đầy đủ cho lượt cuối; so beam 1 với beam 5 | Dư thời gian cho LLM trong 30 s |
| 6 | Tuỳ chọn: đưa top-3 hypothesis kèm độ tin cho LLM | Có ích khi chuỗi số hay tên mơ hồ |

## 6. Bổ sung cho model FC

| # | Việc | Lý do |
|---|---|---|
| 1 | **val_ood chặt:** bỏ app giữ lại khỏi `installed_apps`, catalog và history của mẫu train (hiện 601/1.158 mẫu train cài app giữ lại, 151 mẫu có call tới chúng trong history), giữ đủ khoảng 8 app như plan | val_ood 65,0 hiện tại lạc quan |
| 2 | Bỏ câu gần trùng giữa val_in và train (17/128) hoặc báo riêng | val_in 78,4 lạc quan |
| 3 | Mốc **SFT sinh text thường** trên cùng dữ liệu | Còn thiếu để chọn Clef hay SFT thường |
| 4 | Function masking + tool nhiễu + xáo catalog (mục 3.6) | Generalize sang tool lạ |
| 5 | Dữ liệu âm khoảng 10% (mục 3.5) | Zero-shot Clef chỉ hỏi lại 29/217 lần cần hỏi |
| 6 | Lời người dùng có lỗi ASR (mục 4.3) | Đầu vào thật là transcript ASR |
| 7 | Sau SFT, tuỳ chọn: GRPO với reward = khớp `state_hash` + khớp tool call (ToolRL, EnvScaler: Qwen3-4B BFCL-MT 34,9 → 38,0) | Chỉ khi còn thời gian và credit |

## 7. Bảng đánh giá

| Thành phần | Tập | Dùng để |
|---|---|---|
| ASR | val 405 utt (train) | Chọn model |
| ASR | public_test đã duyệt tay | Báo cáo, kiểm tra lệch phân phối. Không chọn model |
| FC | val_in (đã bỏ trùng), val_ood chặt, val_synth_ood | Chọn model, tỉ lệ trộn |
| FC | public_test leaderboard | Tham khảo app lạ thật (vòng public cho phép nộp) |
| End-to-end | val_in/val_ood với transcript ASR thật | Điểm thật; quyết định ưu tiên ASR hay FC |

## 8. Thứ tự

| # | Việc | Phụ thuộc | Chi phí ước |
|---|---|---|---|
| 1 | Sửa bug uid; val_ood chặt + bỏ trùng val_in; end-to-end FC với ASR v1 trên val | – | Thấp |
| 2 | Duyệt tay khoảng 300 câu ngẫu nhiên + câu lệch nhau (mục 1) | – | Công người |
| 3 | Augment trên 1.545 mẫu thật (masking, tool nhiễu, đổi tên/số) + dữ liệu âm → train lại FC, so với mốc SFT thường | 1 | 1 lần train FC |
| 4 | `synth_tools/`: 10 app thử → kiểm tra tự động → 60–120 app | – | ~1 USD/app |
| 5 | Pipeline APIGen-MT trên app tổng hợp → train FC | 4 | Sinh dữ liệu + 1 lần train |
| 6 | TTS: VieNeu (+F5) sinh 2 h thử → lọc vòng ngược → smoke ASR; được thì lên 10–20 h | text từ 3, 5 | GPU sinh audio |
| 7 | ASR v2 + tên nhiễu kiểu B-Whisper + audio tổng hợp, train đủ dữ liệu | 6 | 1 lần train ASR |
| 8 | Transcript ASR trên audio tổng hợp/train → FC có nhiễu ASR | 6, 7 | – |
| 9 | Tăng tốc (faster-whisper, batch history), đo 30 s/mẫu trên A30 | 7 | – |
| 10 | Tuỳ chọn: sửa tên theo phát âm sau ASR, N-best cho LLM, GRPO | 7, 8 | – |

Mốc dừng: sau bước 3, nếu val_ood chặt không tăng thì xem lại chất lượng augment trước khi đầu tư bước 4–5. Sau bước 6, nếu 2 h TTS làm WER val thật tăng thì đổi engine/giọng hoặc giảm tỉ lệ trước khi lên 10–20 h.

## 9. Rủi ro

- **Luật:** dùng catalog/code public_test hay nhãn test vào train làm hỏng tính hợp lệ và mất thước đo app lạ. Có danh sách loại trừ và ghi nguồn cho mọi bộ dữ liệu sinh.
- **Học theo phong cách dữ liệu tổng hợp** (tên tool, cách đặt câu, giọng TTS): dùng masking, trộn nhiều engine/giọng, epoch cuối chỉ dữ liệu thật, chọn model theo val thật.
- **Trajectory sai mà vẫn qua lọc:** chỉ giữ trajectory tái tạo được đúng state khi chạy lại; lấy mẫu kiểm tay khoảng 50 trajectory mỗi đợt.
- **Chép nhầm tên** khi danh sách tên dài hơn: theo dõi chỉ số báo động giả ở mục 5.
- **Độ trễ:** prompt dài hơn và thêm bước sửa tên → đo trên A30 trước khi bật.
- **License TTS:** F5-TTS-Vietnamese là NC; ElevenLabs bị loại; API khác phải đọc ToS.

## 10. Nguồn (survey 2026-11-08; số liệu là số tác giả báo cáo, chưa kiểm lại)

- Tool tổng hợp và generalize:
  - ToolACE https://arxiv.org/abs/2409.00920
  - EnvScaler https://arxiv.org/abs/2601.05808
  - AgentScaler https://arxiv.org/abs/2509.13311
  - APIGen-MT https://arxiv.org/abs/2504.03601
  - Hammer https://arxiv.org/abs/2410.04587
  - When2Call https://arxiv.org/abs/2504.18851
  - ToolRL https://arxiv.org/abs/2504.13958
  - From Text to Voice https://arxiv.org/abs/2605.15104
- ASR:
  - B-Whisper https://arxiv.org/html/2502.11572v2
  - CB-Whisper https://aclanthology.org/2024.lrec-main.262.pdf
  - Truy xuất thực thể theo phát âm (Apple) https://arxiv.org/pdf/2409.15353
  - Bộ chuẩn hoá cho ngôn ngữ ngoài tiếng Anh https://arxiv.org/html/2409.02449v4
  - faster-whisper https://pypi.org/project/faster-whisper
- TTS và audio tổng hợp:
  - VieNeu-TTS https://github.com/pnnbao97/VieNeu-TTS
  - F5-TTS-Vietnamese https://huggingface.co/hynt/F5-TTS-Vietnamese-100h
  - TTS cho từ ngoài từ điển https://arxiv.org/abs/2011.11564
  - Khoảng cách thật/tổng hợp https://arxiv.org/abs/2410.12279
  - Hard-Synth https://arxiv.org/abs/2411.13159
  - Thích nghi miền chỉ từ text + TTS https://arxiv.org/abs/2309.10707
