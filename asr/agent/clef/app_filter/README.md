# Lọc app: so BM25 và 3 model embedding

Báo cáo: `plans/REPORT_app_filter.md`. Thiết kế: `plans/PLAN_app_filter.md`.

| File | Việc |
|---|---|
| `kernel.py` | Kernel Kaggle (T4): BM25, e5-small, vietnamese-bi-encoder, paraphrase-multilingual-MiniLM-L12; zero-shot rồi fine-tune 3 epoch; đo recall theo top-k và độ trễ CPU |
| `kernel-metadata.json` | Cấu hình kernel (riêng tư, GPU T4, bật internet để tải model) |
| `bm25_baseline.py` | BM25 chạy ở máy local, không cần GPU |
| `results.json`, `summary.log` | Kết quả lượt chạy 2026-11-08 (kernel version 2) |

Dữ liệu vào là dataset Kaggle riêng tư gồm ba file: `train.jsonl`, `train_tools_catalog.json` (trong `data/train+public_test/train/`) và `agent/clef/splits.json`. Không có audio, không có `*_environment/`, không có public test. Trong `kernel-metadata.json`, `id` và `dataset_sources` đang trỏ tới tài khoản đã dùng; đổi sang tài khoản của bạn khi chạy lại.

```
# BM25 local
python asr/agent/clef/app_filter/bm25_baseline.py data/train+public_test asr/agent/clef/splits.json

# Kaggle (token đặt qua biến môi trường KAGGLE_API_TOKEN, không để trong repo)
kaggle datasets create -p <thư mục chứa 3 file + dataset-metadata.json>
kaggle kernels push -p asr/agent/clef/app_filter --accelerator NvidiaTeslaT4
kaggle kernels output <user>/zaic-app-filter-eval -p <thư mục>
```

Lưu ý: `kaggle ... -p` phải chạy từ trong thư mục đó với đường dẫn tương đối nếu dùng Git Bash trên Windows, nếu không CLI báo lỗi đường dẫn.
