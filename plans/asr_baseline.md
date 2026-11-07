# Kế hoạch so sánh các baseline ASR trên train

Cập nhật: 07/11/2026, múi giờ Asia/Saigon. Mục tiêu là so chất lượng ASR, latency và chi phí trước khi ghép LLM/tool calling. Modal profile: `vyvy372004`. Giữ PhoWhisper-small làm mốc và test thêm 5 model.

## Trạng thái

- [x] Metrics ASR và CLI đã có; bộ metrics có 47 unit test qua ở lượt đầu.
- [x] PhoWhisper-small đã chạy smoke 20 mẫu và đủ 1.545 mẫu train.
- [x] Có runner cho 5 model mới, manifest chung và điều phối tối đa 5 worker; 8 test runner và 47 test metrics qua.
- [x] Smoke chung 100 ID cho đủ 5 model, không lỗi/timeout/truncation; đã lưu WER/CER và dự toán full.
- [x] Full train 5 model mới, tổng hợp kết quả và kiểm tra tái lập 2 model đầu bảng.

Repo chính: `D:\GitHub\do-nothing`. Runner mới: [asr/modal_suite.py](../asr/modal_suite.py). Smoke đã hoàn tất ở `20261107-baselines-v3`; full tiếp tục bằng [launcher nhiều tiến trình](../asr/launch_suite.py) tại `20261107-isolated` để tách các Modal client trên Windows. Các prediction thành công được giữ cùng số đo gốc và provenance.

## Phạm vi và đầu vào

- Dataset: `data/train+public_test/train/train.jsonl`, 1.545 cặp `audio`–`audio_text` của yêu cầu cuối.
- Chỉ đánh giá các WAV có transcript chuẩn; audio user trong history chưa có nhãn text nên không đưa vào WER/CER lần này.
- Không dùng public/private test để chỉnh model. Không đọc hoặc tải các thư mục `*_environment/` lên Modal.
- Mốc ban đầu: [vinai/PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small), BSD-3-Clause, đã công bố trước hạn 26/10/2026. Smoke test đo được 241.734.912 tham số; revision `a86b604c346caf7148c37512eafe783a16420adb`.
- Chỉ chuyển ID và bytes WAV tới GPU; giữ transcript chuẩn ở máy local để chấm sau khi model đã xuất dự đoán.

## Lượt PhoWhisper-small đã thực hiện

1. Bổ sung `metrics/asr.py` và CLI chấm ASR bằng WER/CER; kiểm tra các case rỗng, token lặp, tiếng Việt NFC, thiếu prediction, lỗi/timeout và định dạng số.
2. Cấu hình và activate đúng profile `vyvy372004`, lấy token từ `.env` vào bộ nhớ; không in hoặc ghi token vào mã nguồn/artifact.
3. Chạy smoke test 20 mẫu train, cố định danh sách mẫu. Dùng một GPU Modal T4 trước để tiết kiệm chi phí; ghi rõ đây không phải GPU A30 dùng để chấm thi.
4. Nếu smoke test hoạt động đúng, chạy đủ 1.545 audio train, model giữ trong một worker, batch size 1, không fine-tune và không retry mẫu thất bại trong cùng lượt.
5. Chấm transcript, lưu lỗi theo mẫu và các ví dụ sai nhiều nhất. Tất cả mẫu vẫn nằm trong mẫu số khi có lỗi hoặc timeout.

## Cấu hình suy luận chung

- PyTorch/Transformers với trọng số local trong image; download model khi build, không gọi model API bên ngoài.
- FP16 trên T4, greedy decoding (`do_sample=False`, `num_beams=1` khi backend hỗ trợ), ngôn ngữ tiếng Việt, task `transcribe`, seed 42 cho Python/NumPy/PyTorch và bật deterministic algorithms. Ánh xạ tham số ngôn ngữ/decoding theo đúng API từng model.
- Bắt đầu `max_new_tokens=256`, kiểm tra chạm trần/truncation trong smoke; nếu cần đổi thì cố định cấu hình mới và chạy lại smoke trước full. Không mặc định BF16 hoặc FlashAttention 2 trên T4; backend attention phải tương thích và được ghi rõ.
- Không dùng `audio_text` hoặc nhãn tool để gợi ý transcript.
- Đo riêng model load, warmup, audio decode/preprocess và thời gian transcribe từng mẫu; ghi GPU, CUDA, dependency versions và peak VRAM.
- Kết quả ASR-only chưa chứng minh toàn pipeline ASR + LLM + tool loop đạt 30 giây.

## Metrics và cách đọc

- WER corpus = tổng substitution/deletion/insertion của token chia tổng token tham chiếu.
- CER corpus = tổng substitution/deletion/insertion của ký tự chia tổng ký tự tham chiếu, bao gồm khoảng trắng sau chuẩn hóa.
- Báo cáo cả raw (NFC, gộp khoảng trắng) và normalized (thêm lowercase, bỏ dấu câu). Giữ dấu tiếng Việt, chữ số, dấu âm, dấu phân cách số và ký hiệu tiền/đơn vị; không tự đổi số thành chữ hoặc ngược lại.
- WER/CER có thể vượt 100% khi model chèn nhiều từ. Transcript rỗng không bị bỏ qua; prediction thiếu/lỗi/timeout được chấm như chuỗi rỗng và vẫn báo lỗi riêng.
- Thêm exact transcript match, số mẫu thành công/lỗi/timeout, latency mean/p50/p95/max, real-time factor và các mẫu lỗi cao nhất.
- Đây là đo trên train để lấy baseline, không phải kết quả generalization trên tập holdout.

## Artifact của baseline đầu

`data/results/asr/phowhisper-small/<run_id>/` gồm:

- `predictions.jsonl`: ID, transcript, trạng thái, lỗi, thời gian và độ dài audio.
- `metrics.json`: tổng hợp và điểm từng mẫu.
- `worst_samples.json`: transcript tham chiếu/dự đoán và lỗi để xem lại.
- `run_config.json`: revision model, tham số decode, phần cứng, dependency versions và thời điểm chạy.
- `run.log`: tiến độ, không chứa token hoặc thông tin từ app environments.

## Điều kiện hoàn tất một lượt full

- Model đã chạy trên đủ 1.545 ID train; mỗi ID có một prediction/status, kể cả khi thất bại.
- Có WER/CER và thời gian thực đo; không dùng kết quả cache làm latency inference.
- Các file kết quả được tải về máy local, không chứa credential.
- Ghi lại giới hạn quan sát được và giữ nguyên baseline để so với model ASR tiếp theo.

## Kết quả PhoWhisper-small

Đã hoàn tất ngày 06/11/2026: profile `vyvy372004` đã activate; smoke 20 mẫu và lượt đầy đủ 1.545 mẫu đều thành công, không lỗi/timeout/thiếu ID. WER chuẩn hóa **34,20%**, CER **19,49%**, trung bình **0,970 giây/mẫu** trên T4. Metrics có 47 unit test qua và đã chấm lại toàn tập bằng CLI `--require-all`.

Xem [báo cáo kết quả](../asr/RESULTS.md), [runner](../asr/modal_phowhisper.py) và [metrics ASR](../metrics/asr.py). Các điều kiện hoàn tất của lượt PhoWhisper-small phía trên đã được đáp ứng.

## Model bổ sung và điều kiện lựa chọn

Giữ kết quả full PhoWhisper-small và chạy thêm **5 model**, tạo bảng tổng cộng 6 baseline. Các số chưa đo là xấp xỉ. Cột FP16 chỉ tính tensor trọng số theo `2 × số tham số` bằng GB thập phân; không phải VRAM tổng hoặc kích thước file download.

| Model / ID Hugging Face | Tham số tham khảo | Trọng số FP16 tối thiểu | Công bố | License | Trạng thái |
| --- | ---: | ---: | --- | --- | --- |
| [vinai/PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) | 241.734.912 đã đo | 0,48 GB | 2024 | BSD-3-Clause | Có full train |
| [vinai/PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium) | 763.857.920 đã đo | 1,53 GB | 2024 | BSD-3-Clause | Full xong: WER 28,33%; CER 16,30% |
| [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) | 1.543.304.960 đã đo | 3,09 GB | 2024 | BSD-3-Clause | Full xong: WER 25,69%; CER 14,76% |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 808.878.080 đã đo | 1,62 GB | 01/10/2024 | MIT | Full xong: WER 32,07%; CER 19,40% |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 782.426.112 đã đo | 1,56 GB | 29/01/2026 | Apache-2.0 | Full xong: WER 33,88%; CER 19,29% |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 2.038.052.480 đã đo | 4,08 GB | 29/01/2026 | Apache-2.0 | Full xong: WER 27,58%; CER 16,00% |

Nguồn PhoWhisper: [trang chính thức VinAI](https://github.com/VinAIResearch/PhoWhisper). Whisper turbo có [thông báo phát hành](https://github.com/openai/whisper/discussions/2363), kích thước/license trong model card. Qwen ghi ngày phát hành tại [repo chính thức](https://github.com/QwenLM/Qwen3-ASR#news); model card từng bản ghi Apache-2.0 và tổng tensor tham số. Cả ba họ có trọng số và mã suy luận để đóng gói offline. Hiện mới có tốc độ thực đo trên dữ liệu này cho PhoWhisper-small.

Tên Qwen `0.6B`/`1.7B` không đủ để tính ngân sách toàn hệ thống. Số đã đo dùng `sum(p.numel() for p in model.parameters())`, tính Parameter chia sẻ một lần: 782.426.112 và 2.038.052.480. Metadata HF ghi tổng tensor 938.008.576 và 2.349.217.408; hai cách đếm được lưu riêng để đối chiếu. Phải cộng cả ASR, LLM và model khác; Qwen lớn ghép LLM đúng 4B vẫn vượt giới hạn **nhỏ hơn** 6B ngay cả theo số đã đo. Vòng baseline không thêm forced aligner/timestamp model.

Trước smoke, ghi SHA revision, thời điểm checkpoint, license và dung lượng snapshot thực tế; chọn checkpoint đã tồn tại trước 26/10/2026. Ngày công bố ở bảng là căn cứ chọn ứng viên, chưa thay thế kiểm tra checkpoint cụ thể. Đối chiếu [AGENT.md](../AGENT.md): tổng hệ thống **<6B**, LLM **<5B**, context LLM **≤16.384 token**, toàn pipeline **≤30 giây/mẫu** trên A30. Chưa chọn LLM ở vòng ASR nên chưa kết luận phương án ghép hợp lệ.


### Kết quả full và repeat (07/11/2026)

| Model | WER chuẩn hóa | CER chuẩn hóa | Mean latency |
| --- | ---: | ---: | ---: |
| PhoWhisper-large | 25,69% | 14,76% | 2,306 giây |
| Qwen3-ASR-1.7B | 27,58% | 16,00% | 1,880 giây |
| PhoWhisper-medium | 28,33% | 16,30% | 1,728 giây |
| Whisper-large-v3-turbo | 32,07% | 19,40% | 0,469 giây |
| Qwen3-ASR-0.6B | 33,88% | 19,29% | 1,939 giây |

Mỗi model có 1.545/1.545 transcript, không lỗi hoặc timeout. Repeat 100 mẫu cho PhoWhisper-large và Qwen3-ASR-1.7B cho kết quả giống hệt full ở cả 100 mẫu. Xem [báo cáo chi tiết](../data/results/asr/comparison/20261107-isolated/report.md) và [artifact so sánh](../data/results/asr/comparison/20261107-isolated/comparison.csv).

## Bước A — Chuẩn bị runner và dữ liệu chung

- [x] Tạo cấu hình theo model: ID/revision, image/dependency, GPU/RAM, dtype/backend, language/decode, seed, timeout và output slug.
- [x] Tham số hóa adapter Whisper/PhoWhisper cho medium, large và turbo; tạo adapter riêng cho Qwen. Giữ chung schema prediction và metrics.
- [x] Tách image Whisper/PhoWhisper và Qwen để tránh xung đột dependency. [requirements-qwen-asr.txt](../requirements-qwen-asr.txt) là điểm xuất phát, cần xác minh adapter thực tế khi build.
- [x] Tạo manifest `train-full`: ID, đường dẫn WAV, độ dài audio, checksum input JSONL/manifest/audio. Không chứa nội dung app environment.
- [x] Chọn manifest `smoke-100`: 20 WAV dài nhất, rồi random 80 ID còn lại bằng seed 42 từ danh sách đã sort ID; hòa độ dài thì sort ID. Lưu ID và hash để mọi model dùng cùng 100 mẫu, không chọn theo nhãn.
- [x] Thêm tùy chọn chạy danh sách ID; runner hiện có chỉ hỗ trợ `--limit` lấy các dòng đầu nên chưa đủ cho manifest này.
- [x] Ghi source hash/version của runner, metrics, jiwer và policy chuẩn hóa. Chỉ gửi ID + WAV lên GPU; labels/references để local.

## Bước B — Smoke 100 mẫu/model song song

- [x] Build image và xác minh load offline, revision, tổng tham số; chạy tối đa 5 worker theo quota thực tế.
- [x] Mỗi model bắt đầu với **1 T4, 2 CPU core, 8 GiB RAM, batch size 1**, một worker giữ model. Đây là cấu hình thử, chưa đảm bảo mọi model load được.
- [x] Warmup riêng trước đo; chạy cùng manifest 100 ID cho 5 model mới và ghi prediction liên tục.
- [x] Kiểm tra ID duy nhất/đủ, schema, lỗi/OOM/timeout, transcript rỗng, sai ngôn ngữ, lặp dài và truncation. Không sửa nhãn hoặc loại mẫu vì WER cao.
- [ ] Nếu thiếu RAM/VRAM, ghi lỗi và tạo cấu hình riêng: tăng RAM trên T4 hoặc thử L4 khi cần. Đổi GPU/backend/dtype/quantization phải có dòng benchmark và dự toán riêng; không trộn latency khác điều kiện với T4.
- [x] Chấm WER/CER raw và normalized, exact match, latency và RTF. Có thể chấm subset prediction PhoWhisper-small cũ theo đúng 100 ID; timing lấy từ full cũ phải được chú thích.
- [x] Ước lượng full bằng tổng duration audio × RTF smoke, đối chiếu `1.545 × mean latency`; cộng load/warmup/overhead và biên dự phòng 25%. Chụp billing, cập nhật dự toán tổng trước full.

Model hoạt động đúng vẫn được chạy full để bảng có đủ ứng viên, kể cả khi WER smoke cao. Model còn lỗi cấu hình hoặc vượt dự toán cần xử lý và cập nhật kế hoạch trước full; không tự tăng số GPU. Nếu không chạy được, ghi nguyên nhân trong bảng, không bỏ mất model.

## Bước C — Full train cho 5 model mới

- [x] Bộ điều phối khởi chạy các job độc lập, tối đa **5 worker GPU tổng, 1 worker/model**, dùng profile `vyvy372004`. Job lỗi không làm mất kết quả job khác; quota thấp hơn thì chạy theo đợt, ghi thời gian xếp hàng.
- [x] Chạy đủ 1.545 ID/model với cấu hình đã cố định, không retry âm thầm. Giới hạn inference từng WAV 30 giây; deadline toàn job tính sau smoke.
- [x] Lưu prediction/status và log theo mẫu, output riêng cho mỗi model/run; dừng worker sau job để tránh giữ GPU nhàn rỗi. Phân biệt latency inference với thời gian truyền dữ liệu, load/warmup, xếp hàng và wall time.
- [x] Chấm lại bằng `python -m metrics asr ... --require-all`; không ghi đè input hoặc artifact lượt cũ.
- [x] Giữ PhoWhisper-small full cũ làm mốc. Nếu đổi policy metrics, chấm lại mọi prediction bằng cùng phiên bản và ghi provenance; không cần suy luận lại chỉ để đổi cách chấm.

## Bước D — Tổng hợp và chọn ứng viên

- [x] Tạo bảng 6 model: số tham số thực, GPU/backend, raw/normalized WER/CER, exact match, success/error/timeout/missing, mean/p50/p95/max, RTF, memory, wall time và cost.
- [ ] Lưu các mẫu WER cao nhất và ví dụ lỗi số/giờ/tên riêng/từ phủ định để phân tích; không sửa ground truth theo dự đoán.
- [x] Ưu tiên WER chuẩn hóa thấp, đối chiếu CER và lỗi nội dung; sau đó xét latency, bộ nhớ, chi phí và khả năng ghép LLM trong giới hạn tổng tham số. Không gộp tùy ý thành điểm thi tool/state.
- [x] Chạy lại cùng 100 ID cho 2 ứng viên tốt nhất để kiểm tra transcript tái lập; ghi tỷ lệ text giống nhau, chênh lệch metrics, run ID và chi phí riêng.
- [ ] Khi tích hợp, đo thêm ASR của history và toàn pipeline LLM/tool loop, ưu tiên xác nhận trên A30. Nếu fine-tune sau này, chia train/validation trước và chấm trên validation chưa dùng huấn luyện.

## Thời gian và ngân sách Modal

Theo [giá Modal](https://modal.com/pricing) kiểm tra ngày 06/11/2026: T4 **0,000164 USD/giây**, CPU **0,0000131 USD/core/giây**, RAM **0,00000222 USD/GiB/giây**. Cấu hình 2 core + 8 GiB tương đương khoảng **0,7487 USD/giờ/worker** ở giá cơ bản. Starter công bố 10 GPU đồng thời; cần kiểm tra quota workspace và GPU còn trống khi chạy.

| Thời gian chạy mỗi worker | Tổng compute 5 worker | Wall time nếu cùng bắt đầu, không xếp hàng |
| --- | ---: | ---: |
| 30 phút | ~1,87 USD | ~30 phút |
| 1 giờ | ~3,74 USD | ~1 giờ |
| 2 giờ | ~7,49 USD | ~2 giờ |

Bảng chưa gồm build/load/warmup/idle và phụ phí khác. Tổng tiền cộng theo worker; thời gian hoàn tất gần job chậm nhất cộng overhead. Dự trù ban đầu cho 5 model mới: **Đã đo:** 3,081 USD compute ước tính cho full/repeat của launcher mới; billing workspace tăng 3,97 USD credit so với snapshot trước đợt ASR và billed 0,00 USD. Tổng cap cấu hình là 8,05 USD; đây không phải hard cap Modal. Các lượt smoke và app dừng trước đó không thuộc 3,081 USD.

- [x] Thêm theo dõi thời gian và spend ước tính ở bộ điều phối; lưu tiến độ rồi dừng/không khởi chạy thêm job khi chạm ngân sách cấu hình. Mẫu chưa chạy báo missing, lượt đó báo incomplete. Đây là chức năng cần triển khai, không phải bảo đảm hard cap của nền tảng.
- [x] Lưu usage/cost theo app/job nếu có và chụp billing trước/sau. Billing workspace có thể gồm workload khác hoặc cập nhật trễ; không gán toàn bộ chênh lệch cho ASR. Lượt PhoWhisper-small trước đó có snapshot riêng; không cộng chênh lệch workspace thành chi phí chính xác của app ASR.

## Artifact chung cho vòng so sánh

Mỗi lượt lưu tại `data/results/asr/<model_slug>/<run_id>/`: `predictions.jsonl`, `references.jsonl` local, `metrics.json`, `worst_samples.json`, `run.log`, thông tin usage và `run_config.json`. Config bổ sung tổng tham số thực, license/snapshot size, manifest/input/source hashes, backend/decode/seed, phần cứng, timestamp và phạm vi đo thời gian; không lưu credential.

Thư mục `data/results/asr/comparison/<run_id>/` gồm manifest full/smoke, danh sách job/trạng thái, `summary.json`, `summary.csv`, `budget_estimate.json` và `report.md`. Cập nhật bảng ngắn vào [asr/RESULTS.md](../asr/RESULTS.md), giữ kết quả PhoWhisper-small cũ và ghi rõ khác biệt điều kiện benchmark.

## Điều kiện hoàn tất vòng so sánh

- Có 6 dòng model: PhoWhisper-small full cũ; 5 model mới có full prediction/status hoặc lý do không chạy được.
- Mỗi lượt full hoàn thành có đúng 1.545 ID duy nhất, chấm `--require-all` thành công; lỗi inference vẫn có status. Lượt bị dừng giữa chừng ghi incomplete, không gọi là full hoàn thành.
- Cùng references/tập ID/policy chấm; latency chỉ so trực tiếp khi cùng điều kiện hoặc có chú thích khác biệt.
- Phân biệt rõ số đo và dự toán thời gian/chi phí; có metadata để tái lập.
- [x] Có kiểm tra tái lập cho 2 ứng viên được chọn: mỗi model trùng transcript 100/100 mẫu. Ngân sách tham số toàn pipeline và tích hợp A30/LLM vẫn là bước tiếp theo.
