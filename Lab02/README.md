# Bài thực hành 2: Hồi quy logistic

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 2 — Hồi quy logistic |
| **Tuần** | 2 |
| **Môi trường** | Google Colab (Python 3, numpy / pandas / matplotlib / scikit-learn) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab02_Hoi_quy_logistic.ipynb` | Notebook: phần bài giảng c1–c6 (kèm hình 1, 3–7) và sáu bài tập |
| `sigmoid.png` | Đồ thị hàm sigmoid của bài tập 2 |
| `data/sinh_vien.csv` | Dữ liệu phát kèm bài: 120 sinh viên, 3 cột |
| `Lab02_Hoi_quy_logistic.pdf` | Đề bài |

## 2. Cách chạy

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn `sinh_vien.csv` để tải lên. Notebook tự đặt tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab02/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| Số sinh viên qua / rớt | 71 / 49 (tỷ lệ qua 0.5917) |
| Giờ ôn trung bình nhóm rớt / qua | 8.29 / 19.89 giờ |
| Mô hình một biến | w = 0.392819, b = −5.064890 |
| Mốc xác suất 0.5 | 12.89 giờ |
| Tập học / tập kiểm tra | 90 / 30 |
| TN / FP / FN / TP | 9 / 3 / 1 / 17 |
| Accuracy / Precision / Recall / F1 | 0.8667 / 0.8500 / 0.9444 / 0.8947 |
| Accuracy mô hình hai biến | 0.9000 (w₁ = +0.3035, w₂ = +0.5885, b = −6.9811) |

Mọi con số khớp với bảng 8 trong đề.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | 31 bạn có điểm giữa kỳ ≥ 7, tỷ lệ qua 0.9032; nhóm còn lại 0.4831 |
| 2 | `sigmoid.png` |
| 3 | 3 → 0.0201; 8 → 0.1276; 12.89 → 0.4996 (nhãn 0); 18 → 0.8814; 26 → 0.9942 |
| 4 | Tự tính 0.8667 / 0.8500 / 0.9444 / 0.8947, khớp thư viện |
| 5 | Ngưỡng tốt nhất theo F1 là 0.60 (F1 = 0.9412), so với 0.8947 ở ngưỡng 0.5 |
| 6 | Lớp rớt môn: precision 0.9000, recall 0.7500 |

## 4. Nhận xét

Nhận xét chi tiết từng bài nằm ngay dưới ô mã tương ứng trong notebook.

## 5. Checklist tuần 2

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
