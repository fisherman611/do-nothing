# Monitor job b8 (e46f tiếp tục fine-tune, chỉ dữ liệu thật) — lần chạy 2

- **Trạng thái:** đang chạy
- **Lần kiểm tra cuối:** 2026-11-12 13:11:37 (giờ server, UTC)
- **Tiến độ:** epoch 1/5, bước 88/725 (~7,9 s/bước, 145 bước/epoch)
- **ETA:** ~1 giờ 25 phút còn lại cho 637 bước (chưa tính eval mỗi epoch)
- **Phiên tmux `b8_asr`:** đang tồn tại (tạo 12:59:39)
- **GPU 1 (job b8):** 9686 MiB, 58%
- **Cấu hình:** --bs 4 --accum 8, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Lưu ý: lần chạy đầu (12:47–12:58) bị CUDA OOM ngay bước 0, log cũ đã chuyển sang `logs/b8_oom1.log`. Lần này là relaunch sau khi giảm batch.

## Bảng eval theo epoch

Chưa có kết quả eval nào.

| epoch | eval WER | CER | so với e46f ~6,2% | ghi chú |
|---|---|---|---|---|
| - | - | - | - | chưa tới |

## Train loss gần đây

| bước | epoch | loss | grad_norm | learning rate |
|---|---|---|---|---|
| ~80 | 0,55 | 2,7828 | 0,761 | 9,998e-5 |

## Nhật ký

- 2026-11-12 12:58:43: lần chạy đầu OOM ở bước 0/725, EXIT 1 (đã chuyển sang b8_oom1.log).
- 2026-11-12 13:01:30: relaunch đang chạy, bước 12/725, GPU 1 ~9,7 GB, không lỗi.
- 2026-11-12 13:11:37: bước 88/725 (epoch 1), loss 2,78 ở bước ~80, không lỗi, tốc độ ổn định.
