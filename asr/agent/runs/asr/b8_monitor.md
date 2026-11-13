# Monitor job B8 và B8c (e46f tiếp tục fine-tune, chỉ dữ liệu thật)

- **Lần kiểm tra cuối:** 2026-11-12 15:16:17 (giờ server, UTC)
- **Mốc so sánh:** e46f ~6,2%; B4b epoch 1 = 7,92% (exam 7,45%); B8 epoch 1 = 8,40%

---

## B8 (CTC 0,3, LR 1e-4, 5 epoch, 725 bước) — DỪNG DO LỖI (mất phiên, không có EXIT)

- **Trạng thái:** lỗi / dừng. Phiên tmux `b8_asr` không còn, không có tiến trình asr_train.py nào của B8, GPU 1 trống (0 MiB). Log không có EXIT và không có traceback hay CUDA OOM, nên tiến trình có vẻ bị kết thúc đột ngột.
- **Bước cuối:** 634/725 (epoch 4,4), log ghi lần cuối lúc 15:15:07 UTC.
- **Cấu hình:** --bs 4 --accum 8, expandable_segments
- **Lưu ý:** lần chạy đầu (12:47–12:58) bị CUDA OOM ngay bước 0, log cũ ở `logs/b8_oom1.log`.
- **Không có** FINAL, RESULT_JSON hay thư mục model đầu ra.

| epoch | eval WER | CER | so với e46f ~6,2% | ghi chú |
|---|---|---|---|---|
| 1 | 8,40% | 4,99% | xấu hơn (+2,2 điểm) | cao hơn B4b epoch 1 (7,92%) |
| 2 | 7,61% | 4,56% | xấu hơn (+1,4 điểm) | tốt hơn epoch 1 |
| 3 | 7,59% | 4,58% | xấu hơn (+1,4 điểm) | gần như không đổi |
| 4 | 7,44% | 4,50% | xấu hơn (+1,2 điểm) | cải thiện nhỏ, vẫn xấu hơn B4b epoch 1 |
| 5 | không có | - | - | chưa tới (job dừng ở bước 634) |
| FINAL | không có | - | - | không có |

Train loss gần đây: epoch 4,0 → 1,2484; epoch 4,28 → 1,2234 (grad_norm 0,513, lr 6,38e-6).

### 30 dòng log cuối (`logs/b8.log`, rút gọn)

```
 85%|████████▍ | 612/725 [2:12:14<15:29,  8.23s/it]
 85%|████████▍ | 613/725 [2:12:23<15:22,  8.23s/it]
 85%|████████▍ | 614/725 [2:12:31<15:13,  8.23s/it]
 85%|████████▍ | 615/725 [2:12:39<15:04,  8.22s/it]
 85%|████████▍ | 616/725 [2:12:47<14:56,  8.22s/it]
 85%|████████▌ | 617/725 [2:12:55<14:48,  8.22s/it]
 85%|████████▌ | 618/725 [2:13:04<14:40,  8.23s/it]
 85%|████████▌ | 619/725 [2:13:12<14:31,  8.22s/it]
 86%|████████▌ | 620/725 [2:13:20<14:23,  8.22s/it]
{'loss': 1.2234, 'grad_norm': 0.5129164457321167, 'learning_rate': 6.381086836051498e-06, 'epoch': 4.28}
 86%|████████▌ | 620/725 [2:13:20<14:23,  8.22s/it]
 86%|████████▌ | 621/725 [2:13:28<14:14,  8.22s/it]
 86%|████████▌ | 622/725 [2:13:36<14:06,  8.22s/it]
 86%|████████▌ | 623/725 [2:13:45<13:58,  8.22s/it]
 86%|████████▌ | 624/725 [2:13:53<13:50,  8.22s/it]
 86%|████████▌ | 625/725 [2:14:01<13:35,  8.16s/it]
 86%|████████▋ | 626/725 [2:14:09<13:29,  8.18s/it]
 87%|████████▋ | 627/725 [2:14:17<13:23,  8.20s/it]
 87%|████████▋ | 628/725 [2:14:26<13:16,  8.21s/it]
 87%|████████▋ | 629/725 [2:14:34<13:07,  8.21s/it]
 87%|████████▋ | 630/725 [2:14:42<12:58,  8.20s/it]
 87%|████████▋ | 631/725 [2:14:50<12:51,  8.21s/it]
 87%|████████▋ | 632/725 [2:14:58<12:43,  8.21s/it]
 87%|████████▋ | 633/725 [2:15:07<12:34,  8.20s/it]
 87%|████████▋ | 634/725 [2:15:15<12:25,  8.19s/it]
```

---

## B8c (CTC tắt, LR 3e-5, 2 epoch, 290 bước) — XONG (EXIT 0)

- **Trạng thái:** xong. tmux `b8c_asr` đã kết thúc, GPU 0 trống.
- **Log:** `/mnt/hungpv/projects/do-nothing/asr_v3_code/logs/b8c.log`

| epoch | eval WER | CER | so với e46f ~6,2% | ghi chú |
|---|---|---|---|---|
| 1 | 6,68% | 3,94% | xấu hơn (+0,5 điểm) | tốt hơn B8 epoch 1 và B4b epoch 1 (7,92%) |
| 2 | 6,56% | 3,95% | xấu hơn (+0,4 điểm) | tốt nhất trong các epoch |
| FINAL | **6,27%** (CER 3,92%) | - | xấu hơn +0,07 điểm, gần e46f nhất | RESULT_JSON {"wer": 0.0627, "cer": 0.0392}, EXIT 0 |

Ghi chú: log chỉ in một dòng FINAL (WER 6,27%), không tách oracle-prompt / no-prompt / exam-condition. Đường dẫn thư mục model đầu ra không được in.

---

## Nhật ký

- 2026-11-12 12:58:43: B8 lần đầu OOM ở bước 0/725, EXIT 1 (chuyển sang b8_oom1.log).
- 2026-11-12 13:01:30: B8 relaunch (bs 4, accum 8), bước 12/725, không lỗi.
- 2026-11-12 13:32:09: B8 eval epoch 1 xong: WER 8,40%, CER 4,99%.
- 2026-11-12 13:39:58: B8c khởi động trên GPU 0.
- 2026-11-12 14:03:05: B8 eval epoch 2 xong: WER 7,61%, CER 4,56%.
- 2026-11-12 14:13:23: B8c eval epoch 1 xong: WER 6,68%, CER 3,94%.
- 2026-11-12 14:44:41: B8 eval epoch 3 xong: WER 7,59%, CER 4,58%. B8c eval epoch 2 xong: WER 6,56%.
- 2026-11-12 15:05:41: B8c FINAL WER 6,27%, CER 3,92%, EXIT 0.
- 2026-11-12 15:05–15:15: B8 eval epoch 4 xong: WER 7,44%, CER 4,50%. Train tiếp đến bước 634/725.
- 2026-11-12 15:16:17: B8 đã mất phiên tmux, không có EXIT, log dừng ở 15:15:07 (bước 634). Ghi nhận là LỖI và dừng monitor theo quy định.
