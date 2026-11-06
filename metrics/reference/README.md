# Nguồn công thức

`official_formula.png` được lấy từ ảnh nhúng trong mục **Evaluation** trên trang LifeMate, đối chiếu ngày 06/11/2026.

- Trang công khai: https://challenge.zalo.ai/portal/lifemate
- API nội dung mà trang sử dụng: `https://api-challenge.zalo.ai/fe/api/config-detail?name=evaluationMetric&title=1` (kèm các tham số UTM mặc định và Origin/Referer của trang).
- Trường `updated_time` của nội dung: `1792937455`.

Với gold không rỗng, công thức trong ảnh cho điểm mẫu bằng 0 khi hash khác hoặc `M = 0`, ngược lại dùng `2PR/(P+R)`, với `P=M/(M+U)`, `R=M/|G|`. Điểm cuối là **100/N × tổng điểm mẫu**. Trường hợp gold rỗng được mô tả riêng trong văn bản của trang.

Văn bản quy định chữ thường và Unicode NFC cho fuzzy token-F1, nhưng không mô tả chi tiết tách token/dấu câu. Implementation local dùng khoảng trắng, giữ dấu câu và token lặp; không khẳng định đã đối chiếu được tokenizer nội bộ của BTC.
