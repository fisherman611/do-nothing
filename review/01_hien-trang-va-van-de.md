# LifeMate (ZAC 2026) — Hiện trạng và vấn đề

**Ngày lập:** 11/11/2026 · **Code:** `do-nothing/` (clone `fisherman611/do-nothing`, commit `c8cf39f`)
**Nguồn:** đọc code + `plans/PLAN_72h_v3.md`, `plans/PLAN_v3_improve.md`, các `REPORT_*`, `asr/docker/`, và tài liệu BTC trong `docs/`.
**Hạn:** nộp Docker **16/11 23:59** (tối đa 2 lần, BTC chấm lần cuối) · leaderboard public tới 16/11 (5 lượt/ngày) · private 20–23/11.

---

## 1. Tóm tắt

- Điểm public đang là **62.76**. Bản nộp: ASR PhoWhisper-large (`e46f`) → bộ lọc app `pack_b` → FC Qwen3.5-4B + LoRA `fmtA_48e` → bật N1 (tìm lại khi tìm kiếm rỗng).
- **Rủi ro lớn nhất không nằm ở điểm mà ở đóng gói:** Docker hiện tại (`asr/docker/`) được viết theo *giả định*, trong khi hướng dẫn chính thức của BTC đã có sẵn ở `docs/Docker_Guide_ZAC2026_LifeMate.pdf` và yêu cầu khác ở nhiều điểm (mục 3). Docker cũng chưa chạy trọn lần nào.
- Về chất lượng, điểm mất chủ yếu ở ba chỗ: ASR nghe sai **câu cuối**, **app chưa từng thấy**, và FC **làm không trọn quy trình**.
- Còn 5 ngày. Plan 72h của nhóm cần ~$45–52 Modal trong khi còn khoảng $20.

## 2. Hệ thống và số đo

| Thành phần | Hiện tại | Tham số |
|---|---|---|
| ASR | PhoWhisper-large fine-tune (`fisherman611/pw-tmp-e46f`), beam 5, fp16, prompt = transcript lượt trước | 1.54B |
| Bộ lọc app | BM25 top-4 ∪ `vietnamese-bi-encoder` fine-tune top-5 ∪ luật → catalog hai tầng | 0.135B |
| FC | Qwen3.5-4B chỉ text + LoRA `fmtA_48e` gộp (`fisherman611/q-f48e`), vLLM, định dạng Clef-token (Q1/Q2 một token) | 4.206B |
| **Tổng** | | **5.88B** (giới hạn 6B, còn dư ~0.11B) |

| Số đo | Giá trị |
|---|---|
| Public: ASR v2 medium → ASR v1 large → + N1 | 59 → 62.04 → **62.76** |
| FC trên val_in (128 mẫu), câu sạch (M1) | 76.5 |
| FC trên val_in, có lỗi ASR giống điều kiện thi (M2) | 63.5 (sai số ±3.2) |
| Chọn đúng tool bước đầu trên public: app cũ / app mới | 97.0% / 84.5% |
| Thước đo app lạ M3b (12 app giữ riêng, có app dễ nhầm) | 56% (mục tiêu ≥ 65%) |
| ASR trên 405 đoạn giữ riêng | WER 6.23%, sai thực thể có nhãn 9.3% |
| Bộ lọc app: recall val_in / val_ood | 97.2% / 98.1%, giữ ~6.5–6.9 app |
| Độ trễ FC trên val_in (chạy nhiều luồng) | p50 13.8 s, p90 33.3 s |

## 3. Vấn đề mức P0 — có thể làm hỏng bản nộp Docker

### 3.1 Docker chưa theo hướng dẫn chính thức của BTC

> **Cập nhật 11/11 tối — đã viết lại trong bản clone `do-nothing/asr/docker/` (chưa commit, chưa build image thật):**
> `predict.py` + `predict.sh` (lần lượt từng mẫu, có trường `time`), `predict_notebook.ipynb`, `start_jupyter.sh`, `selfcheck.py`,
> `save.sh`, Dockerfile mới (`/code`, `ENTRYPOINT []`, `jupyterlab`, `pyvi`), `asr_stage.py --serve`, README cho `/code` và `training_code/`.
> Thử trên máy không GPU bằng ASR giả + vLLM giả: 634 mẫu public qua bộ kiểm của BTC (0 lỗi schema); notebook Run All khớp
> `submission.jsonl` 634/634; ASR treo/chết và vLLM không trả lời đều vẫn ra đủ dòng kết quả.
> **Còn phải làm trên máy có GPU:** build thật, đo bộ nhớ khi ASR + vLLM cùng nạp (`GPU_UTIL=0.70`), đo p50/p95 một luồng,
> nộp thử một lượt public để xác nhận lối chạy tuần tự không lệch điểm, điền các mục **[ĐỘI ĐIỀN]** trong `training_README.md`.
> Chi tiết: `do-nothing/asr/docker/README.md`. Bảng dưới là tình trạng **trước khi sửa**.

`asr/docker/README.md` ghi "tài liệu đóng gói của BTC sẽ có riêng" và liệt kê các giả định chưa xác nhận. Tài liệu đó đã có: `docs/Docker_Guide_ZAC2026_LifeMate.pdf` (19 trang). Đối chiếu:

| Yêu cầu của BTC | Docker hiện tại | Hệ quả |
|---|---|---|
| BTC chạy `docker run … <image> /bin/bash /code/predict.sh`; code ở `/code` | Code ở `/app`, không có `/code/predict.sh`; image đặt `ENTRYPOINT ["bash","/app/docker/run.sh"]` | Lệnh của BTC trở thành tham số của `run.sh`. Lệnh chạy notebook (`/bin/bash /code/start_jupyter.sh`) cũng bị entrypoint nuốt → không mở được Jupyter |
| Mỗi dòng kết quả có thêm trường **`time`** (ms của mẫu, tính từ trước khi đọc audio) | `submit.py` ghi đúng 3 trường `id, tool_calls, state_hash` | Thiếu trường BTC dùng để chấm thời gian |
| Phải có **`predict_notebook.ipynb`**: đặt seed → nạp model → đọc test → chạy **lần lượt từng mẫu** (ASR → LLM → tool) → ghi `/result/jupyter_submission.jsonl` | Chưa có notebook, chưa có `start_jupyter.sh`, chưa cài `jupyterlab` | BTC "chỉ mở notebook và run all cells". Thiếu là không chấm được thời gian |
| Điểm của `jupyter_submission.jsonl` **phải giống** `submission.jsonl` thì BTC mới dùng `time` | Hai lần chạy cùng cấu hình đang lệch ~6/634 mẫu (do batch khác nhau khi chạy 16 luồng) | Có thể bị coi là không tái lập; khi lệch BTC chạy 5 lần lấy trung bình |
| Notebook đo từng mẫu nối tiếp, model nạp sẵn một lần | Kiến trúc hiện tại: ASR chạy hết mọi mẫu → **thoát để nhả GPU** → mới bật vLLM → 16 luồng song song | Notebook cần ASR và vLLM **cùng nằm trên GPU**. vLLM đang đặt `--gpu-memory-utilization 0.85` → phải hạ để chừa chỗ cho PhoWhisper; chưa đo |
| In thời gian nạp model và thời gian predict ra stdout | `run.sh` chỉ in mốc giờ từng giai đoạn | Thiếu nhỏ, dễ thêm |
| Thư mục `training_code/` + `README.md` mô tả ý tưởng (model, số tham số, context length, vòng agent, dữ liệu tổng hợp) | Chưa có trong image | Thiếu thành phần bắt buộc |
| BTC **chạy lại training** (có mạng) để kiểm tính nhất quán; data + model để trên HF, **không được đổi sau hạn nộp**; base model không đưa vào Docker | Training rải ở Modal / Kaggle (3 tài khoản) / myserver; nhiều thứ không có trong git (mục 3.5) | Cần gom script + data + URL cố định trước 16/11 |
| Base image CUDA ≤ 13.2 (driver 595.58.03) | `vllm/vllm-openai:v0.31.0` dùng CUDA **13.0.2** (đã kiểm metadata image) | Đạt. Lưu ý một câu khác trong tài liệu ghi "bắt buộc CUDA 13.2" — nên hỏi BTC |
| Nộp: `docker save … \| gzip` + MD5, đặt Google Drive công khai | Chưa làm | Image vLLM + 2 model rất nặng → cần thời gian nén và upload |

**Tài liệu BTC tự mâu thuẫn về phần cứng:** phần chữ ghi GPU **RTX 5090 32 GB**, CPU Xeon Gold 6459C; ảnh chụp `nvidia-smi` trong cùng tài liệu và `docs/rule.md` lại ghi **A30 24 GB**, Xeon Gold 6442Y. Nên thiết kế cho 24 GB và hỏi BTC.

### 3.2 Image thiếu `pyvi` — **đã sửa trong bản clone**

- Bộ lọc app dùng `vietnamese-bi-encoder` (gốc PhoBERT), được train với câu đã tách từ bằng `pyvi` (`train_v2.py:357`).
- Image không cài `pyvi`. Khi thiếu, `filter.py:110` lặng lẽ bỏ tách từ **và** bỏ luôn giới hạn 256 token. Hệ quả: điểm embedding khác lúc đo; theo ghi chú trong code, tài liệu app dài còn gây `IndexError` → cả vòng agent của mẫu lỗi.
- Đã sửa `asr/docker/Dockerfile`: cài `pyvi==0.1.1` và thêm bước kiểm ngay lúc build (import + tách từ một câu). `docker build --check` qua. **Chưa commit, chưa push, chưa build image thật.**
- Chưa xác minh: lượt đo 62.76 chạy trên máy có `pyvi` hay không. `requirements.txt` không liệt kê `pyvi`, `sentence-transformers`, `rapidfuzz`.

### 3.3 Docker chưa chạy trọn lần nào

- README của Docker ghi "chưa build, chưa chạy thử". Plan ghi đang chạy thử 20 mẫu trên Modal A10; chưa có số p50/p95 và bộ nhớ GPU.
- Image vLLM quản lý Python bằng `uv`; lệnh `pip install` trong Dockerfile chưa được xác nhận chạy được trong image này.

### 3.4 Giới hạn 30 giây mỗi mẫu

- p90 = 33.3 s trên val_in (đo khi chạy nhiều luồng nên bị đội lên; chạy một luồng sẽ thấp hơn, nhưng chưa đo trên A30).
- Ngân sách thời gian chỉ được kiểm ở **đầu mỗi bước** (`clef_agent.py:251`), chừa 2 s. Một bước gồm nhiều request (Q1, các Q2, sinh tham số) nên vẫn có thể vượt.
- Mỗi request tới vLLM có `timeout=600` giây và thử lại 8 lần, chờ tới 30 s mỗi lần (`request_json`). Nếu vLLM treo hoặc chết giữa chừng, một mẫu có thể chờ nhiều phút, không bị ngân sách 30 s chặn. *(Cập nhật: lối chạy `predict.py` hạ xuống 2 lần thử × 30 s và tự bật lại vLLM nếu tiến trình chết; lối theo lô `run.sh` giữ mặc định cũ.)*
- Mẫu bị cắt giữa chừng nộp các call đã có → thường sai state → 0 điểm mẫu đó.

### 3.5 Không dựng lại được từ git

- `pack_b` (bộ lọc app) và `agent/clef/data/fmtA_48e_20k/train.jsonl.meta.json` (định dạng prompt) là đầu vào bắt buộc của `docker/ctx.sh` nhưng không nằm trong repo.
- `sentence-transformers`, `soundfile`, `huggingface_hub` không ghim phiên bản.
- Dữ liệu tổng hợp (`synth/…`), bảng nhiễu ASR, adapter LoRA nằm trên HF/Modal/myserver, chưa có bảng kê URL + commit.

## 4. Vấn đề mức P1 — luật thi

| Điểm | Chi tiết | Đề nghị |
|---|---|---|
| `get_state()` trên **public_test** | `asr/qwen_cloud_label.py:76` và `asr/label_remaining.py:523` gọi `ToolEnv.get_state()` của public_test để lấy hotword khi gán nhãn ASR. `docs/rule.md`: "get_state chỉ dành cho quá trình huấn luyện" và cấm gán nhãn test để huấn luyện/tinh chỉnh | Nhóm ghi là "chỉ để đánh giá". Vẫn là vùng xám: không để nhãn này ảnh hưởng tới lựa chọn model nào, không đưa script này vào `training_code/`, hoặc hỏi BTC |
| Không fit theo test | Nhóm tự đặt luật không dùng catalog/tên tool của public test để sinh data, không tune theo public | Đang giữ đúng; tiếp tục |
| Tổng tham số | 5.88B < 6B, còn dư 0.11B | Không còn chỗ cho model thứ hai |

## 5. Vấn đề mức P2 — chất lượng (theo báo cáo của nhóm)

### 5.1 ASR nghe sai câu cuối — nguyên nhân lớn nhất

| Nhóm mẫu (val_in) | Sạch | Có lỗi ASR | Tụt |
|---|---|---|---|
| Chỉ history có lỗi, câu cuối sạch (38) | 76.1 | 76.1 | 0.0 |
| Câu cuối có lỗi (90) | 76.7 | 58.1 | **18.6** |
| Câu điều kiện "nếu… thì…" (33) | 75.4 | 50.2 | **25.2** |

- Kiểu lỗi chính: động từ hành động hoặc từ nối bị nghe sai ("huỷ" → "huyết", "họp" → "họ") → model **bỏ luôn hành động của vế đó** (call thiếu 20 → 40).
- Tham số: `id`/`enum` gần như không bị ảnh hưởng; `datetime` 0.95 → 0.81; nội dung tự do 0.76 → 0.60 (model chép nguyên từ nghe sai).
- ASR còn yếu ở tên chưa gặp: tựa bài hát/sự kiện/món ăn sai 13.4%, số sai 10%.
- FC đã được train với nhiễu ASR (40% mẫu, bảng 441 cụm) mà vẫn tụt 13 điểm: bảng nhiễu quá hẹp, gần như không đụng tới động từ.

### 5.2 App chưa từng thấy (44% mẫu public)

- Chọn tool bước đầu kém app cũ 12.5 điểm. 41 lỗi: 20 chọn DỪNG, 14 sang app trùng chức năng (`public_transit`↔`maps`, `parking`↔`ride_hailing`, `grocery`↔`ecommerce`), 6 sai tool cùng app.
- Khi sai, xác suất của tool đúng rất thấp (trung vị 0.078) → hỏi lại/ngưỡng không cứu được. Đã đo N8 và I5: **không tăng điểm**, đang tắt.
- Dữ liệu app lạ hiện có (`synth/newapps`) chỉ dạy được bước chọn tool, không có tham số và không có môi trường chạy → không dạy được chuỗi "tìm → đặt".

### 5.3 FC làm không trọn quy trình

- Tìm kiếm rỗng thì dừng: 25 mẫu public (N1 cứu một phần, +0.72).
- Tìm thấy rồi vẫn dừng: 18 mẫu, 13 ở app mới.
- Hỏi lại/làm sai chỗ: 9/41 lỗi trên câu sạch; gọi thừa khi người dùng chỉ kể chuyện: 29 lần (9 ở ecommerce).
- Nội dung tin nhắn/ghi chú chỉ khớp 76% ngay cả với câu sạch (diễn đạt khác nhãn).

## 6. Vấn đề mức P3 — đo lường và quy trình

- `fmtA_48e` đã train trên `val_ood` → mất thước đo app lạ gốc; M3b thay thế chỉ đo bước chọn tool.
- M2 sai số ±3.2 trên 128 mẫu → chỉ thấy thay đổi ≥ 4–6 điểm.
- M1 không đo được lợi ích của các luật như N1/N4 (val_in gần như không có tìm kiếm rỗng) → chỉ biết qua lượt nộp public.
- `PROJECT.md`, `TEST_READY.md`, `TEST_INFRA.md` mô tả việc phụ (gán nhãn ASR public), dễ gây hiểu nhầm là kiến trúc chính.
- Chưa có test nào cho đường chạy chính (`clef_agent.py`, `submit.py`, `asr_stage.py`) chạy được mà không cần GPU.

## 7. Nguồn lực và các quyết định đang treo

- Tiền: plan cần ~$45–52 Modal, còn khoảng $20. Kaggle miễn phí (3 tài khoản, 30 h/tuần mỗi tài khoản).
- Việc đang chạy nền theo plan (trạng thái 11/11 19:47, có thể đã đổi): B0 (lỗi ASR thật trên audio train) → dựng data FC v3; Qwen3-ASR 6 epoch; ASR L2; D1 thử Docker.
- Chủ repo cần quyết: ngân sách, dùng myserver hay không, train FC tiếp từ adapter cũ hay từ đầu.

## 8. Đã kiểm chứng gì trong phiên này

| Việc | Kết quả |
|---|---|
| `metrics/tests` (chấm điểm, phát lại tool call, WER) | **47/47 đạt** (cần `jiwer`, cài trong venv tạm) |
| `asr/tests` | **31/31 đạt** (cần `tqdm`, `rapidfuzz`) |
| `tests/` (bộ E2E gán nhãn ASR public) | 6/35 đạt; 29 lỗi vì thiếu file `data/results/asr/qwen-cloud/…` không có trong repo — không liên quan pipeline chính |
| `docker build --check` cho Dockerfile đã sửa | Qua, không cảnh báo |
| CUDA của base image vLLM | 13.0.2 (đọc metadata, không tải image) |
| Dữ liệu | `do-nothing/data/train+public_test` và `do-nothing/asr/data/train+public_test` là symlink tới `public_data/` |

**Chưa kiểm được:** build image thật, chạy `clef_agent` (cần GPU + vLLM + `pack_b`), các số đo trong báo cáo của nhóm (lấy nguyên từ plan), `test_fmt.py`/`test_loop.py` (cần `transformers` + tokenizer).

## 9. Việc nên làm, theo thứ tự

1. ~~Đưa Docker về đúng hướng dẫn BTC~~ — **mã đã viết xong và thử bằng mock** (xem cập nhật ở mục 3.1). Việc còn lại là mục 2 bên dưới và điền `training_README.md`.
2. **Build và chạy thật** 20 mẫu rồi 634 mẫu, tắt mạng, trên GPU 24 GB: đo p50/p95 một luồng, bộ nhớ GPU khi hai model cùng nạp, kiểm `pyvi`.
3. **Tất định:** chạy notebook và `predict.sh` cho ra cùng kết quả (thử một luồng, hoặc `VLLM_BATCH_INVARIANT=1`); đặt deadline cho từng request theo ngân sách còn lại của mẫu.
4. **Đóng băng tái lập:** bảng kê URL + commit của data/model trên HF, gom script train vào `training_code/`, đưa `pack_b` và file meta định dạng vào nơi lấy lại được.
5. Hỏi BTC ba điểm: phần cứng chấm (A30 hay 5090), CUDA "≤ 13.2" hay "đúng 13.2", và `get_state` trên public test cho mục đích đánh giá.
6. Song song, cải thiện chất lượng theo đề xuất ở `02_de-xuat-cai-thien-tu-paper.md`.
