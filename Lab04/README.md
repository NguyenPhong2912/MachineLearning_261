# Bài thực hành 4: K láng giềng gần nhất

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 4 — K láng giềng gần nhất (KNN) |
| **Tuần** | 4 |
| **Môi trường** | Google Colab / Python 3.9 (numpy / pandas / matplotlib / scikit-learn 1.6) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab04_KNN.ipynb` | Notebook: phần bài giảng e1–e5 (kèm hình 1–7) và sáu bài tập (thêm hình 8), có nhận xét dưới từng bài |
| `code/e1_khoang_cach.py` … `code/e5_knn_hoi_quy.py` | Năm tệp mã của buổi thực hành (bảng 1 trong đề) |
| `baitap04/bai1.py` … `baitap04/bai6.py` | Mã nguồn sáu bài tập |
| `data/sinh_vien_mo_rong.csv` | 120 sinh viên, 4 cột `gio_on`, `diem_giua_ky`, `chi_tieu`, `qua_mon` |
| `data/hai_vang_trang.csv` | 300 điểm, 3 cột `x1`, `x2`, `nhan` |
| `data/gia_nha.csv` | Dùng lại từ bài 1: 60 căn nhà |
| `Lab04_KNN.pdf` | Đề bài |

## 2. Cách chạy

Chạy từ thư mục `Lab04/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python code/e1_khoang_cach.py
...
python code/e5_knn_hoi_quy.py
python baitap04/bai1.py
...
python baitap04/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả ba tệp csv để tải lên. Notebook tự đặt chúng vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab04/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| d(A, B) và d(A, C) khi chưa chuẩn hoá | 102.2 và 3000.0 |
| d(A, B) và d(A, C) khi đã chuẩn hoá | 3.85 và 1.79 |
| Năm láng giềng của bạn mới (12 giờ, 6.5 điểm) | 4 qua, 1 rớt, đoán qua môn |
| Ba cột, không chuẩn hoá / có chuẩn hoá / bỏ cột chi tiêu | 0.5333 / 0.8667 / 0.9000 |
| Mức đoán bừa trên tập kiểm tra | 0.6000 |
| k tốt nhất theo kiểm định chéo | 13, điểm 0.8667 |
| Hai vầng trăng, KNN với k = 13 | 0.9200 |
| Hai vầng trăng, logistic và Naive Bayes | cùng 0.8533 |
| Giá nhà, MAE của KNN và của đường thẳng | 0.4030 và 0.2578 |
| Căn 150 m² và 250 m² theo KNN | cùng 9.32 tỷ |

Mọi con số khớp bảng 3 trong đề. Hình 6 (lời nguyền số chiều): tỷ số gần nhất / xa nhất đo được 0.016 với 2 chiều và 0.861 với 500 chiều; đề ghi 0.01 và 0.86 nhưng không nêu cách đo.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | KNN tự viết đạt 0.9200, bằng `KNeighborsClassifier`; 0/75 mẫu khác nhau, kiểm thêm với k = 1, 4, 5, 13, 31, 101 đều khớp |
| 2 | Manhattan: k tốt nhất 27 (điểm 0.8667), tập kiểm tra 0.9067; Euclid: k = 13, 0.9200 |
| 3 | Chi tiêu sang triệu đồng, không chuẩn hoá: 0.5333 lên 0.8667 (12/30 mẫu đổi dự đoán); có chuẩn hoá: cả hai đơn vị đều 0.8667, 0/30 đổi |
| 4 | `weights="distance"`: k = 1 0.8000 (không đổi), k = 13 0.9333 (từ 0.9200), k = 101 0.9067 (từ 0.8533) |
| 5 | Kiểm định chéo chọn k = 2 (MAE 0.3498); tập kiểm tra KNN k = 2 0.4867, k = 5 0.4030, đường thẳng 0.2578 |
| 6 | `brute`: fit khoảng 0.2–2 ms gần như không đổi, predict 2 → 7 → 50 ms theo 1 000 → 10 000 → 100 000 mẫu |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** KNN chỉ có ba bước: tính khoảng cách tới mọi mẫu học, lấy k mẫu gần nhất bằng `np.argsort`, bỏ phiếu. Muốn khớp thư viện tuyệt đối thì phải xử lý chỗ hoà giống thư viện: khoảng cách bằng nhau thì lấy mẫu đứng trước (`kind="stable"`), hoà phiếu thì lấy lớp nhỏ hơn.

**Bài 2.** Trên dữ liệu 2 chiều có thang đo gần nhau, đổi Euclid sang Manhattan chỉ lệch 1 mẫu kiểm tra. Chọn k và chuẩn hoá quan trọng hơn chọn công thức khoảng cách.

**Bài 3.** Không chuẩn hoá thì khoảng cách bị cột có số to nhất chi phối, nên chỉ đổi đơn vị là đổi cả mô hình. Kết quả tốt lên ở đây là may: lần này cột lấn át là giờ ôn, vốn có ích. Có chuẩn hoá thì đổi đơn vị không ảnh hưởng gì.

**Bài 4.** Trọng số theo khoảng cách làm k = 101 bớt cẩu thả (sửa 4 mẫu, không làm hỏng mẫu nào), vì láng giềng gần nặng gấp khoảng 15 lần láng giềng xa. Cái giá là accuracy tập học luôn bằng 1.0, vô nghĩa giống k = 1.

**Bài 5.** `neg_mean_absolute_error` mang dấu âm, phải đổi dấu trước khi chọn. Với 48 căn học, điểm kiểm định chéo rất nhiễu: k tốt nhất nhảy giữa 2, 3, 4 khi xáo lại các phần. Dù chọn k nào, đường thẳng vẫn thắng vì quan hệ thật gần như thẳng.

**Bài 6.** Huấn luyện gần như không tốn gì, mọi chi phí dồn sang lúc dự đoán và tăng theo cỡ dữ liệu, nên KNN là mô hình học lười. Với mặc định `algorithm="auto"`, scikit-learn dựng cây k-d lúc fit nên fit chậm dần còn predict gần như đứng yên.

## 5. Checklist tuần 4

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
