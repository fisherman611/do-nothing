# Nộp Docker — các bước

File image (trên a100): `/mnt/hungpv/zac_export/zac2026_TeamName.tar.gz`
MD5: `/mnt/hungpv/zac_export/zac2026_TeamName.md5`

Image `zac2026-lifemate:v1` đã chứa sẵn mọi model trong `/code/saved_models` (ASR, FC, embedding bộ lọc, tokenizer).
Lúc chạy không tải gì từ HuggingFace và không cần mạng.

## 1. Đổi tên file theo tên đội (không làm đổi MD5)

```bash
cd /mnt/hungpv/zac_export
mv zac2026_TeamName.tar.gz zac2026_<TenDoi>.tar.gz
cat zac2026_TeamName.md5          # MD5 không đổi khi đổi tên file
```

## 2. Upload lên Google Drive bằng `gdrive` (đã có trên a100)

```bash
gdrive account add                # làm một lần: mở link, đăng nhập Google, dán mã xác nhận
gdrive files upload /mnt/hungpv/zac_export/zac2026_<TenDoi>.tar.gz
# ghi lại FILE_ID in ra, rồi mở quyền "Anyone with the link":
gdrive permissions share --role reader --type anyone <FILE_ID>
```

Link nộp: `https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing`

Cách khác: tải file về máy (`scp a100:/mnt/hungpv/zac_export/zac2026_<TenDoi>.tar.gz .`), kéo vào drive.google.com,
chuột phải → Share → General access: **Anyone with the link** (Viewer).

## 3. Nộp trên portal Zalo AI Challenge

Form "Solution Submission": dán **link Google Drive** và **MD5**. Tối đa 2 lần nộp; BTC chỉ chấm lần cuối. Hạn 23:59 (GMT+7) 16/11.

## 4. BTC chạy kiểm tra như sau (để tự kiểm lại)

```bash
docker load -i zac2026_<TenDoi>.tar.gz
docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result zac2026-lifemate:v1 /bin/bash /code/predict.sh
docker run -it --gpus '"device=0"' -p 9777:9777 -v <data>:/data:ro -v <out>:/result zac2026-lifemate:v1 /bin/bash /code/start_jupyter.sh
```

`<data>` chứa thư mục `private_test/` (cùng cấu trúc `public_test`). Kết quả: `<out>/submission.jsonl`
(notebook: `<out>/jupyter_submission.jsonl`), mỗi dòng `id, tool_calls, state_hash, time`.
