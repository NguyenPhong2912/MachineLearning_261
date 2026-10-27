# Bài thực hành 5: Cây quyết định và Rừng ngẫu nhiên

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 5 — Cây quyết định và Rừng ngẫu nhiên |
| **Tuần** | 5 |
| **Môi trường** | Google Colab / Python 3.9 (numpy / pandas / matplotlib / scikit-learn 1.6) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab05_CayQuyetDinh_RungNgauNhien.ipynb` | Notebook: phần bài giảng f1–f7 (kèm hình 1–8) và sáu bài tập (thêm hình 9), có nhận xét dưới từng bài |
| `code/f1_gini.py` … `code/f7_cay_hoi_quy.py` | Bảy tệp mã của buổi thực hành (bảng 2 trong đề) |
| `baitap05/bai1.py` … `baitap05/bai6.py` | Mã nguồn sáu bài tập |
| `cay_nam_la.png`, `cay_sau_2.png` | Hai hình cây do bài tập 1 lưu ra |
| `data/khach_vay.csv` | 600 hồ sơ khách vay, 5 cột `thu_nhap`, `ty_le_no`, `co_viec_on_dinh`, `so_nguoi_phu_thuoc`, `tra_dung_han` |
| `data/gia_nha.csv` | Dùng lại từ bài 1: 60 căn nhà |
| `Lab05_CayQuyetDinh_RungNgauNhien.pdf` | Đề bài |

## 2. Cách chạy

Chạy từ thư mục `Lab05/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python code/f1_gini.py
...
python code/f7_cay_hoi_quy.py
python baitap05/bai1.py
...
python baitap05/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả `khach_vay.csv` và `gia_nha.csv` để tải lên. Notebook tự đặt hai tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab05/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| Gini nút gốc của mười khách | 0.48 |
| Ngưỡng tốt nhất và mức Gini giảm | 12.0 và 0.1633 |
| Mức Gini giảm của câu hỏi về việc làm | 0.0610 |
| Cây năm lá, tập học và tập kiểm tra | 0.7978 và 0.7467 |
| Cây tự do, độ sâu và số lá | 13 và 92 |
| Cây tự do, tập kiểm tra | 0.7667 |
| Độ sâu tốt nhất và điểm kiểm định chéo | 5 và 0.8333 |
| Cây sâu 5, tập kiểm tra (gốc / chuẩn hoá / nghìn đồng) | cùng 0.8533, 0 khách đổi dự đoán |
| Rừng 200 cây, tập kiểm tra | 0.9000 |
| Độ quan trọng của thu nhập và tỷ lệ nợ | 0.479 và 0.339 |
| Độ quan trọng của cột ngẫu nhiên | 0.150 |
| Giá nhà, MAE của cây sâu 3 và của đường thẳng | 0.5039 và 0.2578 |

Mọi con số khớp bảng 7 trong đề. Hai chỗ lệch nhỏ ngoài bảng 7, do khác phiên bản scikit-learn (cách rừng rút ngẫu nhiên đã đổi giữa các bản): rừng 50 cây đạt 0.8867 (đề 0.8733), và khi thêm cột ngẫu nhiên, độ quan trọng của tỷ lệ nợ và việc làm là 0.269 và 0.078 (đề 0.270 và 0.077).

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | Lưu `cay_nam_la.png` (giống hình 2 của đề) và `cay_sau_2.png`; hai lá trái của cây sâu 2 cùng là "trễ hạn" ([68, 53] và [114, 14]) nhưng phép chia làm Gini giảm 0.054 |
| 2 | `tim_cau_hoi` tìm ra `thu_nhap <= 15.050`, Gini giảm 0.1221, trùng câu hỏi gốc của scikit-learn |
| 3 | Entropy: độ sâu tốt nhất 5 (23 lá), tập kiểm tra 0.8467; Gini: độ sâu 5 (27 lá), 0.8533; khác 7/150 dự đoán |
| 4 | `min_samples_leaf` tốt nhất 10 (hoà 20, điểm 0.8089), tập kiểm tra 0.8467; kém cây sâu 5 (0.8333 / 0.8533) |
| 5 | Xáo trộn: cột ngẫu nhiên xếp thứ 5/5 (giảm 0.0003 ± 0.0111); theo `feature_importances_` nó xếp thứ 3 (0.150) |
| 6 | MAE: rừng 0.4559, cây sâu 3 0.5039, đường thẳng 0.2578; căn 250 m² rừng đoán 9.20 tỷ, bằng căn 150 m² |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** Cây tối ưu độ thuần (Gini) chứ không tối ưu nhãn, nên nó vẫn chia khi hai con cùng nhãn: xác suất đúng hạn 0.44 và 0.11 khác hẳn nhau, và nhóm lẫn lộn được tách riêng để chia tiếp ở tầng sau.

**Bài 2.** Câu hỏi gốc là kết quả của việc thử hết mọi cột và mọi ngưỡng ở giữa hai giá trị liền nhau rồi lấy mức giảm Gini lớn nhất. Thu nhập thắng với mức giảm gấp đôi cột đứng sau.

**Bài 3.** Gini và entropy có hình dạng rất giống nhau nên chọn cùng câu hỏi lớn và gần như cùng kết quả; khác biệt chỉ ở vài nút sâu.

**Bài 4.** Trên bộ này, giới hạn độ sâu tốt hơn giới hạn cỡ lá. Hai cây có số lá tương đương, nhưng giới hạn cỡ lá vẫn cho nhánh đông khách mọc sâu tới 8 tầng và học thêm nhiễu.

**Bài 5.** `feature_importances_` đo trên tập học, thưởng cho cột có nhiều giá trị khác nhau, kể cả cột vô nghĩa. Xáo trộn trên tập kiểm tra đo đúng cột nào có ích cho dữ liệu mới: cột ngẫu nhiên về 0, và việc làm ổn định vượt lên thứ 3.

**Bài 6.** Rừng mịn hơn một cây nhưng vẫn không ngoại suy được: mỗi cây trả về giá của lá ngoài cùng, nên dự đoán không bao giờ vượt giá cao nhất tập học (9.74 tỷ).

## 5. Checklist tuần 5

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
