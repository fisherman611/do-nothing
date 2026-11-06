# Kế hoạch test ASR: PhoWhisper-small trên train

Ngày lập: 06/11/2026. Mục tiêu là đo chất lượng nhận dạng tiếng Việt trước khi ghép LLM/tool calling. Modal profile: `vyvy372004`.

## Phạm vi và đầu vào

- Dataset: `data/train+public_test/train/train.jsonl`, 1.545 cặp `audio`–`audio_text` của yêu cầu cuối.
- Chỉ đánh giá các WAV có transcript chuẩn; audio user trong history chưa có nhãn text nên không đưa vào WER/CER lần này.
- Không dùng public/private test để chỉnh model. Không đọc hoặc tải các thư mục `*_environment/` lên Modal.
- Model: [vinai/PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small), BSD-3-Clause, đã công bố trước hạn 26/10/2026. Smoke test đo được 241.734.912 tham số; revision `a86b604c346caf7148c37512eafe783a16420adb`.
- Chỉ chuyển ID và bytes WAV tới GPU; giữ transcript chuẩn ở máy local để chấm sau khi model đã xuất dự đoán.

## Trình tự

1. Bổ sung `metrics/asr.py` và CLI chấm ASR bằng WER/CER; kiểm tra các case rỗng, token lặp, tiếng Việt NFC, thiếu prediction, lỗi/timeout và định dạng số.
2. Cấu hình và activate đúng profile `vyvy372004`, lấy token từ `.env` vào bộ nhớ; không in hoặc ghi token vào mã nguồn/artifact.
3. Chạy smoke test 20 mẫu train, cố định danh sách mẫu. Dùng một GPU Modal T4 trước để tiết kiệm chi phí; ghi rõ đây không phải GPU A30 dùng để chấm thi.
4. Nếu smoke test hoạt động đúng, chạy đủ 1.545 audio train, model giữ trong một worker, batch size 1, không fine-tune và không retry mẫu thất bại trong cùng lượt.
5. Chấm transcript, lưu lỗi theo mẫu và các ví dụ sai nhiều nhất. Tất cả mẫu vẫn nằm trong mẫu số khi có lỗi hoặc timeout.

## Cấu hình suy luận ban đầu

- PyTorch/Transformers với trọng số local trong image; download model khi build, không gọi model API bên ngoài.
- FP16 trên GPU, greedy decoding (`do_sample=False`, `num_beams=1`), ngôn ngữ tiếng Việt, task `transcribe`, seed 42.
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

## Artifact dự kiến

`data/results/asr/phowhisper-small/<run_id>/` gồm:

- `predictions.jsonl`: ID, transcript, trạng thái, lỗi, thời gian và độ dài audio.
- `metrics.json`: tổng hợp và điểm từng mẫu.
- `worst_samples.json`: transcript tham chiếu/dự đoán và lỗi để xem lại.
- `run_config.json`: revision model, tham số decode, phần cứng, dependency versions và thời điểm chạy.
- `run.log`: tiến độ, không chứa token hoặc thông tin từ app environments.

## Điều kiện hoàn tất

- Model đã chạy trên đủ 1.545 ID train; mỗi ID có một prediction/status, kể cả khi thất bại.
- Có WER/CER và thời gian thực đo; không dùng kết quả cache làm latency inference.
- Các file kết quả được tải về máy local, không chứa credential.
- Ghi lại giới hạn quan sát được và giữ nguyên baseline để so với model ASR tiếp theo.

## Kết quả thực hiện

Đã hoàn tất ngày 06/11/2026: profile `vyvy372004` đã activate; smoke 20 mẫu và lượt đầy đủ 1.545 mẫu đều thành công, không lỗi/timeout/thiếu ID. WER chuẩn hóa **34,20%**, CER **19,49%**, trung bình **0,970 giây/mẫu** trên T4. Metrics có 47 unit test qua và đã chấm lại toàn tập bằng CLI `--require-all`.

Xem [báo cáo kết quả](../asr/RESULTS.md), [runner](../asr/modal_phowhisper.py) và [metrics ASR](../metrics/asr.py). Các điều kiện hoàn tất phía trên đã được đáp ứng.
