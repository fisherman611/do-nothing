# LifeMate (ZAC 2026) — Đề xuất cải thiện từ paper

**Ngày lập:** 11/11/2026 · **Đọc kèm:** `01_hien-trang-va-van-de.md` (số đo và vấn đề hiện tại)
**Nguồn:** 6 bản review trong `Works/Paper Research /` + tìm kiếm trên mạng (3 luồng: ASR, FC chịu lỗi ASR, FC với app lạ).

> **Cách đọc.** Chưa có đề xuất nào được đo trên hệ thống của mình. Số liệu là của paper, phần lớn trên benchmark tiếng Anh với model 7–13B. Các trang được đọc qua công cụ tóm tắt; mục ghi *(chỉ abstract)* hoặc *(chỉ snippet)* là chưa đọc toàn văn. Kiểm lại con số trước khi trích. Cột "Độ tin" là nhận định về mức phù hợp với pipeline của mình, không phải kết luận của paper.

---

## 1. Tóm tắt

Ràng buộc quyết định mọi thứ: còn dư ~0.11B tham số (không thêm được model), 30 s/mẫu, chạy offline tất định, hạn Docker 16/11.

| # | Việc | Sửa lỗi nào (theo báo cáo 01) | Loại | Chi phí | Độ tin |
|---|---|---|---|---|---|
| 1 | Khớp ngữ âm tên nghe sai với tên trong kết quả tool; khi tìm rỗng thì thử lại bằng các giả thuyết ASR hạng 2–5 | Tìm rỗng rồi dừng (25 mẫu); tham số chép từ nghe sai | Luật lúc infer, không train | ~1 ngày code, 0 tham số | Cao |
| 2 | Data FC: che tên tool bằng chuỗi ngẫu nhiên + cặp tương phản cho app trùng chức năng | App lạ: 84.5% vs 97%; 14/41 lỗi sang app trùng chức năng | Data, cùng vòng train FC v3 | Dùng LLM ngoài sinh data | Cao |
| 3 | App tổng hợp có môi trường giả chạy được, sinh chuỗi "tìm → đặt" | Tìm thấy rồi vẫn dừng (18 mẫu) | Data, cùng vòng train | 1–2 ngày | Cao về hướng, trung bình về mức tăng |
| 4 | Đưa 3–5 giả thuyết ASR của câu cuối vào prompt FC và train trên định dạng đó | Bỏ hành động khi vế câu bị nghe sai (−18.6 điểm) | Prompt + train | Một vòng LoRA | Trung bình–cao |
| 5 | Bộ ba hỏi / làm / dừng từ cùng một câu, giữ "làm" là lớp lớn nhất | Hỏi/làm sai chỗ (9/41), gọi thừa (29) | Data, cùng vòng train | Sinh bằng luật + LLM ngoài | Trung bình–cao |
| 6 | Train tiếp ASR để **biết dùng** prompt từ khoá (có từ đúng lẫn từ nhiễu), rồi nghe lại câu cuối với tên từ kết quả tool | Tên chưa gặp sai 13.4% | Train ASR (Kaggle) + infer | 1 lượt train, +0.5 s/mẫu | Cao về hướng |

Ba cảnh báo từ paper đối với plan hiện tại của nhóm:

1. **Prompt từ khoá cho Whisper chưa train thì hại.** Ba paper cùng báo: danh sách từ trong prompt làm tăng recall thực thể nhưng **tăng WER tổng** (10.3 → 12.1; 13.39 → 16.70). Bước C2 trong plan ("nghe lại với tên từ kết quả tool") nên đi sau bước train ở mục 6, hoặc dùng dạng "câu nói tự nhiên" (mục 3.A2).
2. **Nguồn nhiễu khác máy nhận dạng thì chuyển giao kém.** Một paper thấy nhiễu sinh từ ASR khác kém hơn nhiễu từ chính ASR dùng lúc thi. Plan B0 đang dùng PhoWhisper gốc làm máy sinh lỗi. Một paper khác thấy train trên 1-best có nhiễu **không hơn** train trên câu sạch; lợi ích đến từ nhiều giả thuyết.
3. **Thêm mẫu "không gọi tool" quá tay làm model dừng nhiều hơn.** Các paper báo đánh đổi rõ: tăng mẫu từ chối thì độ chính xác gọi tool giảm. Hệ thống đang có 20/41 lỗi app lạ là chọn DỪNG.

## 2. Sáu paper trong thư mục `Paper Research`

| Paper | Dùng được gì cho LifeMate | Kết luận |
|---|---|---|
| **Gorilla** (APIBench, RAT) | Ý "train có tài liệu nhiễu để model không bị retriever kéo sai" — hệ thống đã làm một phần (`tier1_for` thêm app nhiễu khi dựng data). Bảng 2 của paper (91 → 47 → 17 khi retrieval kém dần) ủng hộ quyết định **không lọc cứng**, giữ catalog hai tầng | Đã áp dụng phần chính; không cần làm thêm |
| **Toolformer** | Model tự học khi nào gọi API bằng dữ liệu tự sinh, tối đa 1 call mỗi input, cần fine-tune theo bộ tool cố định | Không áp dụng: khác bài toán (nhiều bước, app lạ) |
| **Beyond Prompts / PRISM** | Hữu ích nhất trong sáu paper. (a) Phân lỗi vào đúng lớp sửa: lỗi cơ học → middleware, lỗi quy trình → prompt/data. (b) Chỉ cho phép ba kiểu middleware: sửa lặng lẽ, chặn và báo lỗi, chặn khi thiếu điều kiện. (c) Tách ba tập repair / gate / scorecard để chọn bản sửa | Áp dụng thủ công: `env.validate` = chặn và báo lỗi (đã có); N1 và việc #1 ở trên = sửa lặng lẽ; dùng M1/M2/M3b làm gate, public chỉ để chấm cuối. Không chạy bộ tối ưu tự động (cần LLM lớn bên ngoài lặp nhiều vòng, không kịp) |
| **HEART** | Planner kiểm tra đủ thông tin, Verifier bốn tiêu chí, lập lại kế hoạch tối đa 3 lần | Không áp dụng nguyên trạng: thêm 3–4 lượt LLM mỗi bước, và Verifier bằng chính model 4B dễ lặp lại lỗi của nó. Phần dùng được đã có: phản hồi lỗi có cấu trúc (`validate` → prompt), ngân sách thử lại (3 lần) |
| **HarnessFlow** | Tìm kiếm phiên bản harness bằng cây + coding agent + bộ chấm tự động. Thư viện thành phần có: nén kết quả tool, kiểm schema, phân loại tool đọc/ghi | Các thành phần đó hệ thống đã có (`clip_result`, `validate`, `kind`). Vòng tìm kiếm tự động: không kịp trước 16/11 |
| **Harness²** | Tự sửa harness ngay trong từng task, không cần đáp án | Không áp dụng: tốn nhiều lượt chạy mỗi mẫu, không tất định, vượt 30 s |

## 3. Paper tìm thêm trên mạng

### A. ASR — thực thể và từ chưa gặp

| # | Kỹ thuật | Paper | Paper báo cáo | Cách gắn vào hệ thống | Rủi ro | Độ tin |
|---|---|---|---|---|---|---|
| A1 | **Fine-tune có điều kiện prompt, kèm từ nhiễu.** Mỗi câu train có một từ hiếm đúng + 25–150 từ nhiễu trong prompt; bỏ từ đúng với xác suất 0.3, bỏ cả danh sách 0.2 | B-Whisper ([2502.11572](https://arxiv.org/abs/2502.11572), SLT 2024); KG-Whisper ([2406.02649](https://arxiv.org/html/2406.02649), Interspeech 2024) | WER từ hiếm 23.7 → 11.2, từ chưa gặp 60.0 → 16.0, WER tổng 10.3 → 8.9 (TB 11 tập tiếng Anh). Whisper **chưa train** + prompt: WER tổng xấu đi | Train tiếp LoRA ASR: prompt = history như hiện tại + đoạn từ khoá (thực thể vàng + động từ hành động; nhiễu = tên khác từ kết quả tool). 0 tham số thêm | LR lớn trên 10k câu dễ làm model chép từ nhiễu; ngân sách prompt ~224 token chia với history | Cao |
| A2 | **Prompt dạng câu nói tự nhiên** thay cho danh sách từ | CB-Whisper ([2309.09552](https://arxiv.org/abs/2309.09552), Interspeech 2024) | Whisper-large-v2 không fine-tune, 2 tập nội bộ: lỗi 4.4 → 2.9 và recall thực thể 87.3 → 97.7; danh sách trần thì lỗi tăng lên 4.7 | Không train: viết kết quả tool thành một lượt assistant giả ("Lịch của bạn có: …") đúng định dạng history mà ASR đã quen, rồi giải mã lại câu cuối (+~0.5 s) | Tập nội bộ; thực thể luôn có mặt trong audio | Trung bình–cao |
| A3 | **Trie cộng điểm trong beam search** cho cụm từ trong danh sách, thu hồi điểm nếu không hoàn tất cụm | [2508.17796](https://arxiv.org/html/2508.17796v2) (APSIPA 2025); [2509.09196](https://arxiv.org/html/2509.09196v1) (Interspeech 2025) | WER từ cần thiên vị 9.86 → 6.11, không hại từ khác (large-v3, beam 10). Danh sách dài thì kém dần | Viết `LogitsProcessor` riêng giữ trạng thái trie theo từng beam (bộ có sẵn của HF không thu hồi điểm). Chỉ dùng với danh sách ngắn | Kích hoạt nhầm từ gần âm; chỉnh điểm thưởng trên 405 đoạn dễ quá khớp; ~1 ngày code | Trung bình |
| A4 | **TTS có trọng tâm + trọng số loss cho token thực thể** | B-Whisper; [2509.09196](https://arxiv.org/html/2509.09196v1) | whisper-small + 10 h audio tổng hợp: WER 30.9 → 16.3 | Đã có trong plan (B2). Phần bổ sung: gazetteer tỉnh thành, từ ngắn dễ nhầm đặt trong câu "nếu… thì…" | Khoảng cách tổng hợp–thật | Trung bình |

Riêng tiếng Việt: PiDA ([2606.12911](https://arxiv.org/html/2606.12911), Interspeech 2026) phân loại lỗi thay thế của PhoWhisper-large: ~56% là từ nước ngoài/ngoài từ vựng, 12% nhầm nguyên âm, 9.5% nhầm phụ âm, **dưới 4% chỉ sai thanh** → bộ sửa dấu/thanh chỉ giải quyết phần nhỏ. Không tìm thấy công trình thiên vị ngữ cảnh nào cho PhoWhisper.

### B. FC chịu lỗi ASR và khớp thực thể

| # | Kỹ thuật | Paper | Paper báo cáo | Cách gắn vào hệ thống | Rủi ro | Độ tin |
|---|---|---|---|---|---|---|
| B1 | **Nhiều giả thuyết ASR trong prompt + LoRA trên định dạng đó** | Sun et al., Findings ACL 2024 ([2311.07418](https://arxiv.org/abs/2311.07418)); Dighe et al. ([2309.04842](https://arxiv.org/abs/2309.04842)) | SLURP, LLM 13B: 5–10 giả thuyết hơn 1-best khoảng 3–6 điểm F1, phần tăng chủ yếu là **recall**. Không fine-tune thì N-best không giúp, có khi hại. Model rất nhỏ bị hại | Beam 5 đã có sẵn 5 giả thuyết (`num_return_sequences`). Chỉ áp cho câu cuối. Dạng gọn: 1-best + các đoạn khác nhau ("huyết \| huỷ"). Phải có trong data train FC | Model 4B nằm giữa cỡ được lợi (13B) và cỡ bị hại (≤ 3B); N-best của Whisper hay gần trùng nhau | Trung bình–cao |
| B2 | **Khớp ngữ âm với danh sách ứng viên** | Raghuvanshi et al., EMNLP 2019 demo ([D19-3011](https://aclanthology.org/D19-3011/)); Pusateri et al. (Apple, [2409.06062](https://arxiv.org/abs/2409.06062)); RECOVER ([2603.16411](https://arxiv.org/html/2603.16411)); Asano et al. ([2501.06129](https://arxiv.org/html/2501.06129v1)) | Gọi theo tên, danh bạ 100k: recall@1 0.10 (chữ) → 0.26 (+ngữ âm) → 0.33 (+N-best). Apple: giảm 33–39% WER trên thực thể hiếm. Mọi paper đều ghi nhận **sửa nhầm** là cái giá phải trả | Không train. Sau khi tool đọc trả danh sách: so từng tên với các đoạn trong transcript bằng khoảng cách âm tiết (âm đầu / vần / thanh). Khi tìm rỗng: thử lại với đoạn từ giả thuyết hạng 2–5, rồi tên gần âm nhất. Dùng cho cả địa danh (ý N4 trong plan) | Thay nhầm cũng mất điểm như bỏ sót → ngưỡng chặt, chỉ thay khi tên có trong danh sách | Cao |
| B3 | **Gợi ý một ứng viên gần âm nhất trong prompt** | Pusateri et al. | Một gợi ý tốt nhất; năm gợi ý thì tụt. Gợi ý **không qua train** gây tụt mạnh | Chỉ làm nếu đưa vào vòng train, có cả trường hợp gợi ý sai | Bộ khớp theo luật yếu hơn encoder đã train của paper | Trung bình–cao nếu train; thấp nếu chỉ thêm lúc infer |
| B4 | **Nhiễu nhắm vào thực thể, giữ nhãn sạch** | Lee et al. ([2409.06263](https://arxiv.org/html/2409.06263v1)); ISNI ([2410.15609](https://arxiv.org/html/2410.15609v1)); PiDA | Độ chính xác tên 45.8 → 51.1 (T5-base). Nhiễu từ ASR khác kém hơn nhiễu từ ASR lúc thi | Tinh chỉnh plan A1 hiện có. Cân nhắc lấy nhiễu từ giả thuyết hạng 2–5 của **chính ASR đang dùng** thay vì model gốc | Dồn nhiễu vào thực thể quá mức làm tăng thực thể bịa | Trung bình |
| B5 | **Ràng buộc nhất quán sạch–nhiễu** (KL giữa hai phân phối đáp án) | Kim et al., NAACL 2024 ([2405.15097](https://arxiv.org/abs/2405.15097), chỉ abstract) | Không có số; model cỡ BERT | Q1 là chọn một token trên logits đã che nên áp KL trực tiếp được | Chưa có bằng chứng cho LLM + LoRA; vế câu mất thật thì ép nhất quán là dạy đoán | Trung bình |
| B6 | **Chấm ứng viên bằng likelihood của chính Whisper** (ý I1 trong plan) | Ma et al., NAACL 2024 ([2311.09363](https://arxiv.org/abs/2311.09363), chỉ abstract) | Phân loại audio zero-shot, không phải phân biệt tên | Chỉ đáng thử cho tham số chữ tự do; `id`/`enum` chỉ tụt 0.98 → 0.96 nên ít chỗ để lấy | Likelihood thô thiên về chuỗi ngắn | Thấp–trung bình |

Tài nguyên tiếng Việt: [vPhon](https://github.com/kirbyj/vPhon) (chữ → IPA ba miền, GPL-3.0) làm nền cho khoảng cách âm tiết; [vietnamadminunits](https://github.com/tranngocminhhieu/vietnamadminunits) (danh mục tỉnh thành, MIT); [ViTextnormASR](https://github.com/vietbtx/ViTextnormASR) (chuẩn hoá số/ngày, Apache-2.0). Không có sẵn thước đo khoảng cách âm tiết tiếng Việt — phải tự viết theo luật.

### C. FC với app lạ, dừng sớm, hỏi lại

| # | Kỹ thuật | Paper | Paper báo cáo | Cách gắn vào hệ thống | Rủi ro | Độ tin |
|---|---|---|---|---|---|---|
| C1 | **Che tên hàm/tham số bằng chuỗi ngẫu nhiên**, giữ mô tả → model phải chọn theo mô tả | Hammer, ICLR 2025 ([2410.04587](https://arxiv.org/html/2410.04587v2)) | Qwen1.5-4B: F1 trung bình 42.3 → 71.4. Mẫu "không liên quan" tốt nhất ở ~10%; nhiều hơn thì độ chính xác gọi tool giảm | Data hiện có 25% đổi tên sang bí danh tiếng Việt — tên vẫn có nghĩa. Thêm biến thể tên vô nghĩa cho app tầng 1 | App tầng 2 chỉ hiện tên → tên vô nghĩa làm hỏng tầng này trừ khi thêm một dòng mô tả | Cao |
| C2 | **Cặp tương phản trên cùng câu hỏi**: thiếu tool đúng → không gọi; thêm tool đúng → gọi. Chỉ có tool gần đúng → dùng; thêm tool chuyên biệt → chuyển sang | GenTool ([2502.18990](https://arxiv.org/html/2502.18990v1)) | Chọn tool trên cụm tool giữ riêng: Llama-3.1-8B 90.2 (baseline 48–60); Phi-3.5-3B 87.4 | Trúng 14/41 lỗi app trùng chức năng. LLM ngoài viết cho mỗi app train một "app anh em" chuyên biệt hơn và một app chung hơn (không dùng tên app của test); xuất cả hai vế làm mẫu Q1 | App anh em tổng hợp có thể không giống cặp trùng thật ở test | Cao cho lỗi trùng app |
| C3 | **Môi trường tổng hợp chạy được + kiểm bằng trạng thái cuối** | AgentScaler ([2509.13311](https://arxiv.org/html/2509.13311v1)); APIGen-MT, NeurIPS 2025 D&B ([2504.03601](https://arxiv.org/html/2504.03601)) | AgentScaler trên Qwen3-4B: ACEBench-zh 43.9 → 65.6. APIGen-MT: vòng phản hồi nâng tỉ lệ bản thiết kế đạt 28% → 70% | LLM ngoài sinh 30–60 app mới: schema + mock Python trên dict, có tool tìm/liệt kê và tool ghi dùng id trả về. Chạy thử, chỉ giữ chuỗi chạy được và đổi trạng thái. Lấp đúng lỗ hổng: `synth/newapps` hiện chỉ dạy được bước chọn tool | Quy ước của mock lệch với `zaic_tools` thật; cách xử lý kết quả rỗng phải đọc từ 1.545 hội thoại train, không tự đặt | Cao về hướng |
| C4 | **Dựng chuỗi nhiều bước từ đồ thị hàm rồi dịch ngược ra câu hỏi** | MAGNET, ACL 2025 ([2503.07826](https://arxiv.org/html/2503.07826)); BUTTON, ICLR 2025 ([2410.12952](https://arxiv.org/html/2410.12952v2)) | MAGNET, Qwen2.5-Coder-7B trên BFCL-v3 nhiều lượt: 8.3 → 26.5 | Dùng cho C3: đi đường trên đồ thị x-ref trước, viết câu người dùng sau → nhãn đúng theo cách dựng. Phép "Split" sinh mẫu "tìm → không làm tiếp được → hỏi" | Câu tiếng Việt dịch ngược có thể gượng | Trung bình–cao |
| C5 | **Bộ ba hỏi / làm / từ chối từ cùng một ví dụ** | When2Call, NAACL 2025 ([2504.18851](https://arxiv.org/html/2504.18851v1)); DiaTool-DPO ([2504.02882](https://arxiv.org/html/2504.02882v2)); AskToAct ([2503.01940](https://arxiv.org/html/2503.01940v2)) | Model 4B: trộn đều 3 loại làm F1 quyết định 29.7 → 48.1 nhưng độ chính xác gọi tool 57.9 → 51.7 ("quá dè dặt"); bản nghiêng về "làm" giữ được 54.0 | Trúng mục plan đã bỏ vì sợ nhãn sai (cặp hỏi/làm). Cách của paper: bỏ một tham số bắt buộc → hỏi; thêm lớp thứ tư: câu kể không cần hành động → DỪNG (29 lần gọi thừa) | Quá liều làm nặng thêm lỗi DỪNG ở app lạ | Trung bình–cao |
| C6 | **Chọn theo chỉ số trên mô tả, xáo chỉ số mỗi ví dụ; diễn đạt lại mô tả** | D3ST ([2201.08904](https://ar5iv.labs.arxiv.org/html/2201.08904)); SGD-X, AAAI 2022 (chỉ snippet) | T5-Large trên SGD: có mô tả 80.0 vs chỉ tên 73.7 | Chủ yếu là kiểm tra: Q1 có xáo thứ tự mỗi ví dụ lúc train không (code có `rng`), mọi phương án có mô tả không. Thêm diễn đạt lại mô tả bên cạnh đổi tên | Thấp | Trung bình |
| C7 | **App nhiễu gần nghĩa trong catalog lúc train**, đôi khi bỏ app đúng | DEER ([2402.16696](https://arxiv.org/html/2402.16696v3)) | Llama2-7B trên 77 tool lạ: 84.0 / 85.2 | Tăng tỉ lệ so với 5% "đổi giao diện" hiện tại | Đổi phân bố kích thước catalog so với test | Trung bình |

Chẩn đoán miễn phí nên làm trước khi train: (1) đếm tỉ lệ nhãn DỪNG ở bước đầu trong data train; (2) hoán vị vị trí phương án DỪNG trong Q1 trên mẫu app lạ — nếu tỉ lệ chọn DỪNG đổi theo vị trí thì model có thiên lệch theo nhãn (PriDe, ICLR 2024, [2309.03882](https://arxiv.org/abs/2309.03882), chỉ abstract).

## 4. Đã xem và loại

| Hướng | Lý do |
|---|---|
| Bỏ phiếu giữa các giả thuyết ASR mà không train lại (ý I4 dạng thuần) | [2001.05284](https://ar5iv.labs.arxiv.org/html/2001.05284): bỏ phiếu chỉ giảm 1.6% lỗi, trong khi nối giả thuyết + train giảm 11.7% |
| Cho LLM sửa transcript tự do trước khi quyết định | Hại với ASR mạnh: Whisper large-v3 3.11 → 3.34 ([2407.21414](https://arxiv.org/html/2407.21414v1)); ngoài miền 3.89 → 4.56 ([2409.09554](https://arxiv.org/html/2409.09554)). Bản chỉ chọn trong N-best thì có lợi |
| Thay cả cascade bằng speech-LLM | Các so sánh gần đây chênh trong khoảng ±2 điểm; model có tiếng Việt và vừa ngân sách tham số chưa thấy |
| Ngưỡng tự tin, hỏi lại bằng góc nhìn khác (N8) | Nhóm đã đo: không tăng. Không paper nào cho thấy lượt thứ hai của cùng model cứu được nhãn có xác suất gần 0 |
| Giải mã ràng buộc bằng grammar | Đã che logits. [2608.13959](https://arxiv.org/abs/2608.13959) (chỉ abstract): ràng buộc sửa output hỏng chứ không cải thiện quyết định gọi/không gọi |
| Lượt tự kiểm "đã làm xong chưa", lập kế hoạch trước khi làm | Bằng chứng mỏng hoặc chỉ ở model lớn; thêm lượt LLM; CoT trong data không giúp ([2412.01130](https://arxiv.org/html/2412.01130v2)) |
| DPO / tối ưu theo cặp ưu tiên | Lợi nhỏ so với SFT, phải viết mã train mới; Q1 một token đã là loss phân biệt nên data theo cặp lấy được phần lớn lợi ích |
| TCPGen, lớp attention ngữ cảnh, bộ dò từ khoá cho Whisper | Cần module train thêm hoặc encoder ngữ cảnh → vượt tham số hoặc không kịp |
| Đổi tên tool theo sở thích của model (PA-Tool) | Sửa lỗi "bịa tên tool" — hệ thống chọn bằng nhãn chữ cái nên không có lỗi này |

Khoảng trống: không tìm thấy paper nào về câu điều kiện/nhiều vế dưới nhiễu ASR, về sửa `datetime`, hay về độ khớp của tham số chữ tự do (nội dung tin nhắn).

## 5. Gợi ý xếp vào 5 ngày còn lại

1. **Ưu tiên tuyệt đối: Docker đúng hướng dẫn BTC** (báo cáo 01, mục 3). Không cải thiện nào có giá trị nếu bản nộp không chạy được.
2. **Không cần train (1–2 ngày, đo bằng lượt nộp public vì M1 không thấy được):** B2 — khớp ngữ âm + thử lại bằng giả thuyết hạng 2–5 khi tìm rỗng (mở rộng N1 đã cho +0.72). Kèm hai chẩn đoán miễn phí ở cuối mục 3.C.
3. **Gộp vào vòng train FC v3 đang chuẩn bị:** C1 + C2 + C5, và C3/C4 nếu kịp dựng mock. Giữ riêng vài app tổng hợp làm thước đo app lạ sạch (thay cho `val_ood` đã mất). Cân nhắc B1 (N-best trong prompt) trong cùng vòng — đây là thay đổi định dạng nên phải quyết trước khi dựng data.
4. **ASR trên Kaggle (miễn phí):** A1; khi có model biết dùng prompt thì mới bật nghe lại với tên từ kết quả tool. Nếu không kịp train: thử A2 và đo WER tổng trước khi dùng.
5. Mỗi thay đổi qua cổng của nhóm: M2/M3b tăng, M1 tụt không quá 1 điểm.

## 6. Giới hạn của bản tổng hợp này

- Số liệu trích từ bản tóm tắt tự động của trang paper; hai lượt đọc cùng một paper (Sun et al.) cho số tuyệt đối khác nhau tuỳ model nền, nên mục B1 chỉ ghi khoảng chênh.
- Một số arXiv ID rất mới (2026) chưa qua bình duyệt.
- Phần "cách gắn vào hệ thống" là suy luận của người viết báo cáo, chưa có thử nghiệm nào.
