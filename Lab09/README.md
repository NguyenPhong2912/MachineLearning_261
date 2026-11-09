# Bài thực hành 9: Mạng nơ-ron nhiều lớp (MLP)

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 9 — Mạng nơ-ron nhiều lớp (MLP) |
| **Tuần** | 9 |
| **Môi trường** | Google Colab / Python 3.12 (numpy / pandas / matplotlib / scikit-learn 1.9) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab09_MLP.ipynb` | Notebook: phần bài giảng m1–m5 (kèm hình 1–7) và sáu bài tập (thêm hình 8–10), có nhận xét dưới từng bài |
| `code/m1_lan_truyen_tien.py` … `code/m5_chu_so.py` | Năm tệp mã của buổi thực hành (bảng 1 trong đề) |
| `baitap09/bai1.py` … `baitap09/bai6.py` | Mã nguồn sáu bài tập |
| `toc_do_hoc.png` | Bốn đường mất mát do bài tập 4 lưu ra |
| `data/xor.csv` | 400 điểm XOR mô phỏng, cột `x1`, `x2`, `nhan` |
| `data/tao_xor.py` | Mã sinh lại `xor.csv` (đề không phát kèm tệp) |
| `data/hai_vang_trang.csv`, `data/sinh_vien_mo_rong.csv` | Dùng lại từ bài 4 |
| `data/gia_nha.csv` | Dùng lại từ bài 1: 60 căn nhà |
| `Lab09_MLP.pdf` | Đề bài |

Bộ chữ số viết tay lấy bằng `load_digits` của scikit-learn, không cần tệp.

## 2. Cách chạy

Chạy từ thư mục `Lab09/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python code/m1_lan_truyen_tien.py
...
python code/m5_chu_so.py
python baitap09/bai1.py
...
python baitap09/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả bốn tệp `xor.csv`, `hai_vang_trang.csv`, `sinh_vien_mo_rong.csv`, `gia_nha.csv` để tải lên. Notebook tự đặt các tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab09/`, chọn `Restart and run all` (khoảng 3 phút).
- **Máy nhiều nhân:** mạng (100, 100) của `m3_do_lon_mang.py` học trên vài trăm điểm, nên BLAS chạy nhiều luồng lại chậm hơn hẳn. Nên đặt `OMP_NUM_THREADS=1` trước khi chạy (các kết quả bên dưới chạy theo cách này).
- `xor.csv` sinh lại bằng `python data/tao_xor.py`: 400 điểm đều trên [−1, 1]², nhãn 1 khi x1·x2 > 0, đảo nhãn khoảng 5% (`default_rng(0)`), làm tròn 4 chữ số.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| Một nơ-ron: z và sigmoid(z) | 1.80 và 0.8581 |
| Mạng tự đặt trọng số tính XOR | 0, 1, 1, 0 |
| XOR, hồi quy logistic | 0.4500 |
| XOR, MLP 1 / 2 / 4 / 8 nơ-ron ẩn, tập kiểm tra | 0.6900 / 0.6900 / 0.9300 / 0.9100 |
| Hai vầng trăng, mạng 2 nơ-ron | 0.8444 và 0.8533 |
| Hai vầng trăng, mạng to không điều chuẩn | 1.0000 và 0.7867 |
| Hai vầng trăng, mạng to với alpha bằng 1 | 0.8800 và 0.9067 |
| alpha tốt nhất, kiểm định chéo | 1.0, 0.8578 |
| Sinh viên mở rộng, không và có chuẩn hoá | 0.4000 và 0.9000 |
| Chữ số, số tham số và số vòng học | 7510 và 144 |
| Chữ số, accuracy trên tập kiểm tra | 0.9844, sai 7 trên 450 |
| Chữ số, dừng sớm | 44 vòng, 0.9578 |
| Chữ số, logistic / KNN / SVM | 0.9711 / 0.9867 / 0.9911 |

Gần như mọi con số khớp bảng 4 trong đề. Các chỗ lệch:
- `xor.csv` do tôi sinh lại (đề chỉ mô tả bộ dữ liệu). Bảy con số của mục 3 khớp sáu; riêng tập học của mạng 8 nơ-ron được 0.9133 (đề 0.9167, lệch 1 điểm trên 300).
- Mạng (100, 100) với alpha 1e-5: tập kiểm tra 0.7867 (đề 0.7333). Kiểm định chéo với alpha 0.001 / 0.1 / 3.0 được 0.7600 / 0.7733 / 0.8444 (đề 0.7733 / 0.7644 / 0.8400). Các mạng điều chuẩn yếu này có rất nhiều nghiệm học thuộc tập học, nên lbfgs ở phiên bản scikit-learn / scipy khác dễ dừng ở một nghiệm khác. alpha tốt nhất và kết quả cuối (1.0, 0.9067) vẫn khớp.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | Lan truyền tiến tự tính (ReLU rồi sigmoid) trùng `predict_proba[:, 1]` ở cả 5 điểm, chênh lệch 0.00e+00 |
| 2 | Mạng h1 = max(0, x1 − x2), h2 = max(0, x2 − x1), y = h1 + h2 cho 0, 1, 1, 0; y = \|x1 − x2\| |
| 3 | logistic 0.8400 / 0.8533, tanh 0.8711 / 0.9067, relu 0.8800 / 0.9067; relu tốt nhất (kiểm định chéo 0.8578) |
| 4 | Tốc độ 0.0001 / 0.001 / 0.01 / 0.1: 582 / 144 / 38 / 30 vòng, accuracy 0.9889 / 0.9844 / 0.9778 / 0.9533; lưu `toc_do_hoc.png` |
| 5 | Tốt nhất `(50,)`, alpha = 1 (kiểm định chéo 0.9785), tập kiểm tra 0.9822; không vượt SVM (0.9911) |
| 6 | MAE: MLP 0.5008, đường thẳng 0.2578; căn 250 m² MLP đoán 28.18 tỷ, đường thẳng 19.91 tỷ |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** Mạng chỉ làm ba phép tính: `relu(X @ W1 + b1)`, rồi `@ W2 + b2`, rồi sigmoid. Lớp ra là một hồi quy logistic chạy trên 8 đặc trưng mới do lớp ẩn tạo ra; mỗi điểm bật một nhóm nơ-ron ẩn khác nhau tuỳ góc nó nằm.

**Bài 2.** Hai nơ-ron đo chênh lệch theo hai chiều, mỗi lúc nhiều nhất một nơ-ron bật, nên y = |x1 − x2|: bằng 1 khi hai bit khác nhau, đúng XOR. Bỏ ReLU thì hai nhánh triệt tiêu nhau và y luôn bằng 0.

**Bài 3.** Với alpha = 1, sigmoid bão hoà và có độ dốc nhỏ, nên ranh giới gần như thẳng. Tanh và ReLU uốn theo hai vầng trăng; ReLU nhỉnh hơn trên kiểm định chéo.

**Bài 4.** Tốc độ học quá nhỏ học rất chậm, quá lớn thì mất mát giật lên xuống quanh đáy và dừng ở chỗ tệ. Mặc định 0.001 là điểm khởi đầu an toàn.

**Bài 5.** alpha = 1 tốt nhất ở mọi kích thước mạng; từ 50 nơ-ron trở lên các mạng gần như ngang nhau, thêm lớp ẩn không giúp gì. SVM mặc định vẫn hơn 4 ảnh.

**Bài 6.** Khác cây và rừng, MLP với ReLU có ngoại suy: ngoài vùng dữ liệu nó thành một đường thẳng. Nhưng độ dốc là của đoạn gấp khúc cuối (0.137 tỷ/m², gần gấp đôi 0.078 của đường thẳng), nên càng ra xa càng sai.

## 5. Checklist tuần 9

- [x] Đã nhận đề và ghi chủ đề vào README này và README gốc
- [x] Notebook chạy lại từ đầu không lỗi (`Restart and run all`)
- [x] Hoàn thành phần nội dung bài giảng
- [x] Hoàn thành toàn bộ bài tập
- [x] Dữ liệu đặt trong `data/`
- [ ] Ảnh chụp kết quả đặt trong `anh_chup/`
- [x] Điền kết quả chính (mục 3)
- [x] Viết nhận xét cho từng bài (trong notebook)
- [x] Cập nhật bảng tiến độ ở README gốc
- [x] Đã commit và push lên GitHub
