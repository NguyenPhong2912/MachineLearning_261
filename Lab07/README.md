# Bài thực hành 7: Đánh giá mô hình

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 7 — Đánh giá mô hình |
| **Tuần** | 7 |
| **Môi trường** | Google Colab / Python 3 (numpy / pandas / matplotlib / scikit-learn 1.9) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab07_DanhGiaMoHinh.ipynb` | Notebook: phần bài giảng h1–h6 (kèm hình 1–5) và sáu bài tập, có nhận xét dưới từng bài |
| `code/h1_do_bua.py` … `code/h6_can_bang.py` | Sáu tệp mã của buổi thực hành (bảng 2 trong đề) |
| `baitap07/bai1.py` … `baitap07/bai6.py` | Mã nguồn sáu bài tập |
| `roc.png` | Hình đường ROC do bài tập 4 lưu ra |
| `data/sinh_vien_bo_hoc.csv` | 1200 sinh viên, 5 cột `diem_tb`, `so_buoi_vang`, `tin_chi_no`, `gio_lam_them`, `bo_hoc` (119 bạn bỏ học) |
| `data/tao_du_lieu.py` | Mã dựng lại `sinh_vien_bo_hoc.csv` (xem lưu ý bên dưới) |
| `data/gia_nha.csv` | Dùng lại từ bài 1: 60 căn nhà (bài tập 6) |
| `Lab07_DanhGiaMoHinh.pdf` | Đề bài |

> **Lưu ý về dữ liệu.** Đề không phát kèm `sinh_vien_bo_hoc.csv` và không có mã sinh dữ liệu. Tệp trong `data/` được dựng lại bằng `data/tao_du_lieu.py` theo mô tả của đề: 1200 dòng, đúng 119 bạn bỏ học, năm dòng đầu giống hệt bảng `df.head()` trong đề, hạt giống được chọn để ma trận nhầm lẫn của hồi quy logistic khớp đề (TN 267, FP 3, FN 13, TP 17). Đây **không phải tệp gốc** của giảng viên, nên mục 2–4 của đề khớp từng con số còn các mục sau chỉ gần đúng (ghi rõ ở mục 3).

## 2. Cách chạy

Chạy từ thư mục `Lab07/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python data/tao_du_lieu.py        # chỉ cần nếu muốn sinh lại tệp dữ liệu
python code/h1_do_bua.py
...
python code/h6_can_bang.py
python baitap07/bai1.py
...
python baitap07/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả `sinh_vien_bo_hoc.csv` và `gia_nha.csv` để tải lên. Notebook tự đặt hai tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab07/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng (bảng 5 của đề) | Kết quả | Đề |
|---|---|---|
| Tỷ lệ bỏ học | 0.099 | 0.099 |
| Accuracy của mô hình lười và của logistic | 0.9000 và 0.9467 | 0.9000 và 0.9467 |
| TP, FP, FN, TN của logistic | 17, 3, 13, 267 | 17, 3, 13, 267 |
| Precision, recall, F1 ở ngưỡng 0.5 | 0.8500, 0.5667, 0.6800 | 0.8500, 0.5667, 0.6800 |
| Ngưỡng chọn bằng kiểm định chéo | 0.25 | 0.25 |
| Precision, recall, F1 ở ngưỡng 0.25 | 0.6452, 0.6667, 0.6557 | 0.6571, 0.7667, 0.7077 |
| AUC của logistic | 0.9491 | 0.9449 |
| Recall kiểm định chéo trên tập học | 0.4843 | 0.4810 |
| Precision và recall khi cân bằng lớp | 0.4727 và 0.8667 | 0.4483 và 0.8667 |

Năm dòng đầu của bảng 5 và toàn bộ `classification_report` khớp đề, vì hạt giống dữ liệu được chọn theo ma trận nhầm lẫn. Các dòng còn lại lệch do dữ liệu dựng lại (không do phiên bản thư viện). Ngoài bảng 5: đường ROC có 36 điểm (đề 28); mười cách chia không phân tầng cho 18–40 bạn bỏ học (đề 20–35); trong bảng so sáu mô hình, Naive Bayes lệch nhiều nhất (recall 0.6333, đề 0.3667), nhưng kết luận vẫn như đề: logistic có AUC và F1 cao nhất, KNN k = 15 có AUC khá mà recall ở ngưỡng 0.5 rất thấp (0.2667). scikit-learn 1.9 in thêm `FutureWarning` cho `SVC(probability=True)`.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | `danh_gia` tự viết khớp scikit-learn ở cả hai mô hình; mô hình lười có TP + FP = 0 nên precision (và F1) là 0/0, hàm trả về 0 kèm cảnh báo, giống `zero_division=0` |
| 2 | SVM RBF trên chữ số: accuracy 0.9911 (4/450 ảnh sai); recall và precision thấp nhất đều là chữ số 8 (0.9767); macro avg = weighted avg = 0.9911 vì mỗi chữ số có 43–46 ảnh |
| 3 | Ngưỡng lớn nhất cho recall ≥ 0.8 trên tập học: 0.19 (recall 0.8202); tập kiểm tra báo 43 bạn, recall 0.7667 (**không đạt** 0.8, thiếu 1 bạn), precision 0.5349 |
| 4 | Lưu `roc.png`; AUC logistic 0.9491, rừng 0.9372; logistic cao hơn trên phần lớn trục, nhưng hai đường cắt nhau (rừng cao hơn ở FPR 0.037–0.081 và 0.248–0.299) |
| 5 | Cân bằng lớp: SVM thay đổi nhiều nhất (recall 0.5000 → 0.9000, báo 18 → 52 bạn); rừng thay đổi ít nhất (recall 0.5667 → 0.6667) |
| 6 | Đường thẳng giá nhà: MAE 0.2578 → 0.6745, RMSE 0.3730 → 1.5505 (gấp 4.16 lần), R² 0.9622 → 0.3475 khi một căn bị đoán lệch thêm 5 tỷ |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** Với mô hình không báo ai, precision không xác định (không có lời báo nào để chấm). Hàm phải kiểm tra mẫu số và nói rõ con số 0 là quy ước; accuracy 0.90 của mô hình lười không có ý nghĩa, recall 0 mới là con số thật.

**Bài 2.** Bộ chữ số gần như cân bằng lớp, nên trung bình theo số mẫu và trung bình cộng gần bằng nhau. Hai dòng avg lệch nhau (như 0.8255 và 0.9418 ở bài bỏ học) là dấu hiệu dữ liệu lệch lớp.

**Bài 3.** Ngưỡng "lớn nhất còn đạt" luôn nằm sát mép: trên tập học chỉ dư 1 bạn, nên trên tập kiểm tra 30 bạn bỏ học nó trượt xuống 0.7667. Khi yêu cầu là mức sàn, phải chừa biên an toàn khi chọn ngưỡng trên tập học, không được chỉnh lại sau khi nhìn tập kiểm tra.

**Bài 4.** AUC cao hơn không có nghĩa là tốt hơn ở mọi ngưỡng; hai đường cắt nhau, nên khi chỉ dùng một vùng ngưỡng cụ thể vẫn phải nhìn đường ROC.

**Bài 5.** Trọng số lớp nhân hằng số phạt của SVM nên dời biên rất mạnh; rừng mọc tự do có lá gần thuần nên trọng số gần như không đổi được kết luận. Cân bằng lớp chỉ dời điểm đánh đổi, giống hạ ngưỡng.

**Bài 6.** RMSE bình phương sai số nên một căn lệch 5 tỷ chiếm 94.4% tổng bình phương sai số (chỉ 64.5% tổng |sai số|). RMSE hợp khi lỗi lớn rất đắt, MAE hợp khi muốn con số ít bị vài điểm bất thường chi phối.

## 5. Checklist tuần 7

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
