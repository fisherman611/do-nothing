# LifeMate — Quy tắc cuộc thi và tiêu chí chọn model

Tài liệu này ghi lại các quy định cuộc thi LifeMate mà mọi thiết kế, đề xuất model và thay đổi pipeline trong repository cần tuân thủ. Khi đề xuất ASR, LLM hoặc thành phần học máy khác, hãy kiểm tra các điều kiện dưới đây trước khi so sánh chất lượng hay tốc độ.

## Các vòng thi

- **Vòng 1 — Public Test:** chạy giải pháp trên `public_test` và nộp kết quả lên leaderboard công khai. Điểm leaderboard chỉ để tham khảo trong quá trình phát triển.
- **Nộp giải pháp cuối:** đóng gói giải pháp thành Docker image theo hướng dẫn riêng của ban tổ chức. Khi chấm, ban tổ chức mount thư mục private test vào container tại `/data`, chạy script suy luận và lấy kết quả tại `result/submission.jsonl`. Container phải chạy hoàn toàn offline, không có kết nối mạng.
- **Vòng 2 — Private Test:** sau khi các đội nộp và khóa giải pháp, ban tổ chức cung cấp `private_test` cùng môi trường, có cùng cấu trúc với `public_test`. Giải pháp đã khóa chạy với phiên bản `zaic_tools` được cung cấp; ban tổ chức chạy lại Docker image để xác minh. Xếp hạng cuối dựa trên điểm private test.

## Ràng buộc khi chọn model

- Tổng số tham số của **toàn bộ hệ thống** phải **dưới 6B**, tính cả ASR, LLM và các thành phần model khác.
- LLM phải **dưới 5B tham số** và context length tối đa **16.384 token**.
- Chỉ cân nhắc dữ liệu công khai và model mã nguồn mở được phát hành **trước ngày 26/10/2026**, với license cho phép sử dụng trong cuộc thi và phân phối trong Docker image.
- Mọi model, tokenizer, thư viện và tài nguyên cần ở sẵn trong image hoặc ổ đĩa khi chạy. Không phụ thuộc vào tải model, API, dịch vụ hoặc tài nguyên mạng ở inference.
- Ưu tiên benchmark end-to-end trên phần cứng chấm: **1 GPU NVIDIA A30, CPU Intel Xeon Gold 6442Y 16 lõi và RAM 64 GB**. Mỗi mẫu phải hoàn tất trong **30 giây**; timeout hoặc lỗi thực thi khiến mẫu đó được **0 điểm**.
- Đánh giá ASR trên tiếng Việt nói tự nhiên và đánh giá LLM trên chất lượng chọn tool, tham số, xử lý lịch sử và chuỗi thao tác nhiều bước. Không xem riêng chỉ số của một thành phần là đủ; pipeline cuối phải đạt giới hạn thời gian và bộ nhớ.

Trước khi đưa một ứng viên model vào đề xuất, hãy kiểm tra và nêu rõ: ngày phát hành, license, số tham số, context length của LLM, dung lượng/tài nguyên cần thiết, khả năng chạy offline và tốc độ thực đo nếu đã benchmark. Dẫn nguồn chính thức như model card, trang phát hành và license. Nếu chưa xác minh được một điều kiện bắt buộc, không khẳng định model đó hợp lệ.

## Dữ liệu và quy tắc chống rò rỉ

- Không đọc, sao chép hoặc parse bất kỳ file nào dưới `<split>_environment/`, bao gồm `state.json`. Mọi truy cập dữ liệu ứng dụng phải thông qua `zaic_tools`.
- Không gán nhãn `public_test` hoặc `private_test`, bằng tay hay qua dịch vụ/model bên ngoài, để huấn luyện hoặc tinh chỉnh giải pháp.
- Có thể tạo dữ liệu tổng hợp trong giai đoạn phát triển bằng dịch vụ hoặc model khác; không sử dụng các dịch vụ/model đó ở inference. Inference vẫn phải hoàn toàn offline.
- `public_test` được dùng để chạy và đánh giá trong vòng public theo quy định; không biến nhãn hoặc kết quả riêng của test thành dữ liệu huấn luyện/tuning.

## Quy tắc dùng `zaic_tools` lúc inference

- Tạo **chính xác một `ToolEnv` cho mỗi mẫu** và dùng môi trường đó xuyên suốt mẫu.
- Chỉ dùng các phương thức `validate`, `call`, `get_tool_calls` và `get_state_hash` ở inference. `get_state` chỉ dành cho train; không dùng `states`, `Executor`, `Simulator`, không clone môi trường và không khởi tạo lại môi trường để thử lại.
- Mọi tool call đã thực hiện trên môi trường phải xuất hiện trong `tool_calls` của bài nộp, kể cả call bị từ chối hoặc lỗi schema.
- `state_hash` phải lấy từ `env.get_state_hash()`. Ban tổ chức phát lại các call theo đúng thứ tự bằng phiên bản `zaic_tools` tương ứng; kết quả của ban tổ chức là kết quả cuối cùng nếu có khác biệt.
- Định dạng đầu ra khi chấm Docker: `result/submission.jsonl`, một dòng cho mỗi mẫu, gồm `id`, `tool_calls` và `state_hash`.

## Tính tái lập và xếp hạng

- Cố định seed của Python, NumPy, PyTorch và các thành phần liên quan.
- Dùng giải mã xác định cho ASR và LLM, chẳng hạn greedy decoding hoặc `temperature = 0`.
- Nếu lần chạy lại khác kết quả đã nộp, ban tổ chức chạy Docker image 5 lần và lấy điểm trung bình làm điểm chính thức.
- Khi hai đội hòa điểm cuối, ban tổ chức tính lại F1 của `tool_calls` mà không áp dụng điều kiện khớp `state_hash`. Nếu vẫn hòa, đội có thời gian xử lý trung bình mỗi mẫu thấp hơn được xếp trên.

Xem [README.md](README.md) để biết cấu trúc dữ liệu, định dạng mẫu và phần mô tả cách tính điểm.
