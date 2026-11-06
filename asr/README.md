# Test PhoWhisper-small bằng Modal

Kế hoạch: [plans/asr_baseline.md](../plans/asr_baseline.md). Runner chỉ dùng audio cuối của train; nhãn `audio_text` được giữ local để chấm sau inference. Mã không truy cập app environment.

Kết quả lượt đầu: [PhoWhisper-small trên đủ 1.545 mẫu train](RESULTS.md).

Chuẩn bị ở local: Modal 1.6.3 và jiwer 4.0.0. Các thư viện Torch/Transformers và trọng số được cài trong image GPU, không cần tải model về máy local.

Chạy các lệnh từ thư mục gốc repo. Nếu PowerShell chặn activate virtualenv, thay `python` trong các lệnh dưới bằng `.\.venv\Scripts\python.exe`; không cần sửa Execution Policy.

```powershell
python -m pip install modal==1.6.3 jiwer==4.0.0
modal profile activate vyvy372004
```

Chạy 20 mẫu đầu để kiểm tra:

```powershell
python -X utf8 -m modal run --profile vyvy372004 -m asr.modal_phowhisper --limit 20 --output-dir data/results/asr/phowhisper-small/smoke-20261106
```

Chạy đủ train:

```powershell
python -X utf8 -m modal run --profile vyvy372004 -m asr.modal_phowhisper --output-dir data/results/asr/phowhisper-small/train-full-20261106
```

Mỗi lượt chọn thư mục output mới; runner không ghi đè prediction cũ. Có thể dùng `--train-dir` nếu gói dữ liệu nằm ở vị trí khác. `-X utf8` tránh lỗi terminal Windows khi Modal in ký hiệu tiến độ.

Model được pin revision, download khi build image; inference chặn network, chỉ load trọng số local. Một worker T4 giữ model, xử lý từng WAV bằng greedy decoding, seed 42 và batch size 1. Mọi call có `retries=0`; lỗi/timeout được ghi theo mẫu. Model load/warmup được báo riêng, latency mỗi mẫu đo decode WAV + preprocessing + generate + text decode, chưa gồm truyền dữ liệu local–Modal hoặc toàn pipeline agent.

Output gồm `predictions.jsonl`, `references.jsonl` (local), `metrics.json`, `worst_samples.json`, `run_config.json` và `run.log`. Raw/normalized WER/CER, thời gian, GPU và revision được lưu để tái lập. Report này đo trên train; không dùng để tuyên bố chất lượng holdout hay tốc độ trên A30.
