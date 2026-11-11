# Phân tích lỗi ASR v2 medium trên 405 đoạn giữ riêng (beam 5, có prompt)

```
285 cụm lỗi ( 270 không tính định dạng số ); 394 từ lỗi
loại                      cụm  % cụm từ nội dung  có trong ngữ cảnh
từ thường khác            140   49.1          67                  4
tên riêng                  37   13.0          22                  3
từ đệm/khẩu ngữ            28    9.8           0                  0
mất từ                     27    9.5           2                  1
thừa từ                    18    6.3           0                  0
sai dấu                    16    5.6           8                  1
định dạng số (chữ↔số)      15    5.3           4                  0
từ mượn/tiếng Anh           2    0.7           2                  0
sai số/tiền/giờ             2    0.7           1                  0
TRẦN biasing: chặt 9/106 | theo từ (từ hiếm) 18/106 | theo từ, không lọc độ hiếm 51/201 | từ đúng đã nằm trong prompt v1 (conv) mà vẫn sai: 25

lỗi chứa từ nội dung: 106; trong đó có sẵn trong ngữ cảnh: 9 (8%)
theo nguồn: [('results', 5), ('conv', 4), ('catalog', 2), ('profile', 1)] | chỉ 1 nguồn: [('catalog', 2), ('results', 2), ('conv', 1), ('profile', 1)]
   [hâu₅ cân₂] → [hô₅ con]  ['conv', 'results']  (từ thường khác)
   [dưng₂ hăn₃] → [điêu₂ hoa₂]  ['conv']  (từ thường khác)
   [đai₂] → []  ['catalog']  (mất từ)
   [mot₅] → [đoc₅]  ['conv', 'results']  (tên riêng)
   [vưa₅] → [vưa₂]  ['profile']  (sai dấu)
   [be₁ bông₁] → [ve₁ bong₁]  ['results']  (tên riêng)
   [toa₅] → [tao₅]  ['results']  (từ thường khác)
   [đen] → [đi ăn]  ['conv', 'results']  (tên riêng)
   [dich₅] → [lich₅]  ['catalog']  (từ thường khác)
```
