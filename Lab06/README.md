# Bài thực hành 6: Máy vectơ hỗ trợ (SVM)

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 6 — Máy vectơ hỗ trợ (SVM) |
| **Tuần** | 6 |
| **Môi trường** | Google Colab / Python 3.12 (numpy / pandas / matplotlib / scikit-learn 1.9) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab06_SVM.ipynb` | Notebook: phần bài giảng g1–g5 (kèm hình 1–8) và sáu bài tập (thêm hình 9–11), có nhận xét dưới từng bài |
| `code/g1_le_rong_nhat.py` … `code/g5_chu_so.py` | Năm tệp mã của buổi thực hành (bảng 1 trong đề) |
| `baitap06/bai1.py` … `baitap06/bai6.py` | Mã nguồn sáu bài tập |
| `anh_doan_sai/` | Bốn ảnh chữ số SVM đoán sai, do bài tập 5 lưu ra |
| `data/sinh_vien.csv` | Dùng lại từ bài 2: cột `gio_on`, `diem_giua_ky`, `qua_mon` |
| `data/sinh_vien_mo_rong.csv` | Dùng lại từ bài 4: thêm cột `chi_tieu` (nghìn đồng) |
| `data/hai_vang_trang.csv` | Dùng lại từ bài 4: 300 điểm, cột `x1`, `x2`, `nhan` |
| `Lab06_SVM.pdf` | Đề bài |

Không có dữ liệu nào phải dựng lại: ba tệp CSV là đúng tệp của bài 2 và bài 4 mà đề dùng lại. 24 điểm của mục 2 sinh bằng `make_blobs(..., random_state=0)`, bộ chữ số lấy bằng `load_digits`, dữ liệu bài tập 6 sinh bằng `make_classification`, đều có sẵn trong scikit-learn.

## 2. Cách chạy

Chạy từ thư mục `Lab06/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python code/g1_le_rong_nhat.py
...
python code/g5_chu_so.py
python baitap06/bai1.py
...
python baitap06/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả `sinh_vien.csv`, `sinh_vien_mo_rong.csv` và `hai_vang_trang.csv` để tải lên. Notebook tự đặt ba tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab06/`, chọn `Restart and run all`. Trên máy nhiều lõi nên đặt `OMP_NUM_THREADS=1` để bài tập 6 đo thời gian ổn định.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| 24 điểm tách được: $w_1$, $w_2$, $b$ | −0.831, −0.355, 4.820 |
| 24 điểm tách được: độ rộng lề, số vectơ hỗ trợ | 2.2126 và 3 |
| Học lại chỉ với 3 vectơ hỗ trợ | $w$ giống hệt, $b$ = 4.821 |
| Sinh viên, số vectơ hỗ trợ khi C = 0.01 và C = 100 | 74 và 27 (lề 3.84 và 1.05) |
| Ba cột, không chuẩn hoá và có chuẩn hoá | 0.6000 và 0.9000 |
| Một ngưỡng trên trục số / thêm cột $x^2$ | 0.7143 / 1.0000 |
| Hai vầng trăng, kernel tuyến tính, đa thức bậc 3, RBF | 0.8667, 0.8667, 0.9200 (89 vectơ hỗ trợ / 225) |
| Gamma bằng 100, tập học và tập kiểm tra | 0.9422 và 0.8133 (217 vectơ hỗ trợ) |
| Dò lưới, cặp tốt nhất | C = 10, gamma = 1 |
| Dò lưới, kiểm định chéo và tập kiểm tra | 0.8711 và 0.9200 |
| Chữ số, SVM kernel RBF | 0.9911, sai 4 trên 450, 626 vectơ hỗ trợ |
| Chữ số, SVM chuẩn hoá, KNN, logistic | 0.9778, 0.9867, 0.9711 |

Mọi con số khớp bảng 3 và các khung kết quả trong đề, không có chỗ lệch.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | $f(3, 3) = +1.262$ (cách ranh giới 1.397), $f(5, 4.5) = -0.932$ (cách 1.032, nằm trong lề); ba vectơ hỗ trợ cách 1.106 ± 0.001 = nửa độ rộng lề $1/\lVert w \rVert$ |
| 2 | Thêm (−1, −1): $f$ cũ = +6.007, đường không đổi; thay bằng (4.5, 4): $f$ cũ = −0.340, đường thành $w$ = (−2.603, −0.807), $b$ = 15.941, lề co từ 2.21 còn 0.73, còn 2 vectơ hỗ trợ |
| 3 | `coef0=0`: bậc 2 / 3 / 4 được 0.5067 / 0.8267 / 0.4800 trên tập kiểm tra (bậc chẵn đối xứng tâm, hỏng hẳn); `coef0=1`: 0.8533 / 0.9333 / 0.9200; theo kiểm định chéo không bậc nào vượt RBF (0.8711) |
| 4 | Lưới mịn chọn C = 30, gamma = 1, kiểm định chéo 0.8756 (hơn 0.0045, nhỏ hơn độ lệch giữa các phần); tập kiểm tra vẫn 0.9200, 0/75 dự đoán đổi |
| 5 | Bảng nhầm lẫn: 446/450 trên đường chéo, bốn lỗi 4→8, 7→9, 8→1, 9→3, mỗi cặp một lần; lưu bốn ảnh vào `anh_doan_sai/` |
| 6 | n = 2000 → 20000: SVC từ ~0.05 s lên ~8 s, LinearSVC từ ~0.003 s lên ~0.03 s; gấp đôi dữ liệu thì LinearSVC ×2, SVC ×3 (bước 5000 → 10000 ×12–14 vì số vectơ hỗ trợ nhảy từ 913 lên 4555) |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** Vectơ hỗ trợ là điểm có $|f(x)| = 1$, nên cách ranh giới $1/\lVert w \rVert$, đúng nửa độ rộng lề $2/\lVert w \rVert$. Phần lệch ở chữ số thứ tư là do làm tròn hệ số và sai số cho phép của bộ giải.

**Bài 2.** Điểm nằm ngoài lề, đúng phía không phải vectơ hỗ trợ nên không đổi được ranh giới. Điểm rơi vào trong lề (ở đây còn sai phía) trở thành vectơ hỗ trợ và với lề cứng C = 1000 nó bóp lề hẹp đi 3 lần: lý do cần lề mềm.

**Bài 3.** Kernel đa thức mặc định `coef0=0` với bậc chẵn cho $f(-x) = f(x)$, còn hai vầng trăng đối xứng tâm theo kiểu đổi lớp, nên bậc 2 và 4 chỉ ngang đoán bừa. Bậc 3 tốt nhất; thêm `coef0=1` thì mọi bậc đều ổn nhưng vẫn không vượt RBF theo kiểm định chéo.

**Bài 4.** Lưới thưa đã tìm đúng vùng tốt; quanh đó điểm kiểm định chéo gần như phẳng, nên dò mịn chỉ nhích trong phạm vi nhiễu và không đổi kết quả trên tập kiểm tra.

**Bài 5.** Không có cặp chữ số nào bị nhầm lặp lại. Cả bốn ảnh sai đều viết khó đọc (vòng hở thành vòng kín hay ngược lại ở độ phân giải 8 × 8); với ảnh 161, lớp 8 và lớp 9 hoà số phiếu và chữ số thật chỉ đứng thứ tư.

**Bài 6.** LinearSVC tăng gần tuyến tính theo n; SVC tăng nhanh hơn hẳn và còn phụ thuộc số vectơ hỗ trợ, ở 20000 mẫu chậm hơn gần 300 lần. Dữ liệu lớn thì thử LinearSVC trước.

## 5. Checklist tuần 6

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
