# Bộ lọc app `pack_b` (đang dùng ở mọi bản nộp từ 11/11)

Gói = `catalog.json` (85 app: 46 train + 36 public + 28 newapps) + `keywords.json` (73 app) + `pack.json` (BM25 top-4 ∪ embedding top-5 ∪ luật history/x-ref) + `model/`.
`model/` KHÔNG để trong repo (540 MB): là `fisherman611/vibi-appfilter-v2-b` (fine-tune từ `bkai-foundation-models/vietnamese-bi-encoder`,
cấu hình B, epoch 2, 11 160 ví dụ, loại 5 app giữ riêng; `model.safetensors` sha256 bắt đầu `f522388fc72e1c7f`).

```bash
hf download fisherman611/vibi-appfilter-v2-b --revision 386ba3baf57876a556d06bb5a0eb3a05fa7944ec --local-dir agent/clef/app_filter/pack_b/model
# dùng: --app-filter agent/clef/app_filter/pack_b
```
