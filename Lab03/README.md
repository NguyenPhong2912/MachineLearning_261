# Bài thực hành 3: Naive Bayes

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 3 — Naive Bayes |
| **Tuần** | 3 |
| **Môi trường** | Google Colab / Python 3.9 (numpy / pandas / matplotlib / scikit-learn 1.6) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab03_Naive_Bayes.ipynb` | Notebook: phần bài giảng d1–d5 (kèm hình 1–6) và sáu bài tập, có nhận xét dưới từng bài |
| `code/d1_bayes_tay.py` … `code/d5_gaussian.py` | Năm tệp mã của buổi thực hành (bảng 1 trong đề) |
| `baitap03/bai1.py` … `baitap03/bai6.py` | Mã nguồn sáu bài tập |
| `data/tin_nhan.csv` | 300 tin nhắn, 2 cột `noi_dung`, `la_rac` |
| `data/sinh_vien.csv` | Dùng lại từ bài 2: 120 sinh viên, 3 cột |
| `Lab03_Naive_Bayes.pdf` | Đề bài |

## 2. Cách chạy

Chạy từ thư mục `Lab03/`, đúng như đề yêu cầu (trong mã vẫn viết `data/tin_nhan.csv`):

```
python code/d1_bayes_tay.py
python baitap03/bai1.py
...
python baitap03/bai6.py
```

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả `tin_nhan.csv` và `sinh_vien.csv` để tải lên. Notebook tự đặt hai tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab03/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| Hậu nghiệm của ví dụ xét nghiệm | 0.5000 |
| Ví dụ hai dây chuyền, khả năng từ dây B | 0.5556 |
| Kho tin nhắn | 300 tin, 120 rác và 180 thường |
| Số từ trong từ điển | 40 |
| Chia dữ liệu | 225 để học, 75 để kiểm tra |
| Ma trận nhầm lẫn của bộ lọc | TP = 26, TN = 45, FP = 0, FN = 4 |
| Accuracy / Precision / Recall | 0.9467 / 1.0000 / 0.8667 |
| Kho tí hon, α = 0 / α = 1 | cả hai điểm bằng 0 / rác 0.0000073212 > thường 0.0000021656 |
| GaussianNB và hồi quy logistic trên dữ liệu sinh viên | cùng 0.9000 |
| Hình 6, với 10 tin huấn luyện | Naive Bayes 0.9131, logistic 0.8712 (trung bình 50 lần bốc) |
| Hình 6, với đủ 225 tin | Naive Bayes 0.9467, logistic 0.9200 |

Mọi con số khớp bảng 8 trong đề, trừ hàng "với 10 tin huấn luyện": đề không nêu cách bốc mẫu nên khoảng cách đo được nhỏ hơn đề (0.9072 / 0.7808), nhưng Naive Bayes vẫn dẫn ở mọi cỡ mẫu.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | P(qua) = 0.745; hậu nghiệm học đều 0.6376, học dồn 0.2819, ít học 0.0805 (tổng 1.0000) |
| 2 | Rác: `mãi` 3.20, `tay` 2.91, `ngay` 2.66, `miễn` 2.48, `trúng` 2.37. Thường: `lớp` −2.61, `em` −2.56, `bài` −2.29, `báo` −2.27, `phòng` −2.27 |
| 3 | α = 0.001 … 100 đều cho 0.9467 / 1.0000 / 0.8667. Đẩy tiếp: α = 500 recall 0.3667, α ≥ 700 đoán mọi tin là thường (accuracy 0.6000) |
| 4 | Tự viết 0.9467 = MultinomialNB 0.9467; 0/75 nhãn khác, điểm loga lệch tối đa 1.42e-14 |
| 5 | MultinomialNB 0.8000 so với GaussianNB 0.9000, không báo lỗi; đổi điểm sang thang 100 thì còn 0.7333; dữ liệu âm thì `ValueError` |
| 6 | 180 thường + 20 rác: accuracy 0.9800, recall rác 0.8000 (4/5); luôn đoán thường đã được 0.9000; trên 100 tin rác bị bỏ ra recall 0.85 |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** Biết bạn đó qua môn thì nhóm nào có tỷ lệ qua cao hơn mức chung 0.745 sẽ tăng xác suất (học đều 0.50 lên 0.64), nhóm thấp hơn sẽ giảm (ít học 0.20 còn 0.08).

**Bài 2.** Thứ quyết định là tỷ số giữa hai lớp chứ không phải số lần xuất hiện: `mãi` chỉ xuất hiện 32 lần ở tin rác nhưng đứng đầu vì gần như không có trong tin thường.

**Bài 3.** Alpha quá lớn thì mọi P(từ | lớp) bị kéo về cùng giá trị 1/V ở cả hai lớp, từ ngữ không còn phân biệt được hai lớp, chỉ còn tiên nghiệm quyết định nên mô hình đoán mọi tin là lớp đông hơn. Năm giá trị đề cho chưa đủ lớn để làm đổi nhãn, nhưng xác suất đã mất dần độ chắc chắn.

**Bài 4.** Muốn khớp thư viện thì cách chia, từ điển và công thức phải giống nhau. Các lỗi cố tình cài vào (quên tiên nghiệm, quên αV, bỏ làm trơn) không đổi accuracy, chỉ lộ ra khi so xác suất. Vì vậy accuracy bằng nhau chưa đủ để chứng minh cài đúng.

**Bài 5.** MultinomialNB coi giờ ôn và điểm là số lần đếm, chỉ học tỷ trọng giữa hai cột. Kết quả phụ thuộc vào đơn vị đo và chỉ chạy được vì dữ liệu tình cờ không âm. Đây là một lỗi im lặng.

**Bài 6.** Accuracy 0.98 không nói lên điều gì khi chỉ đoán bừa là thường đã được 0.90. Cần xem recall của lớp rác, và với chỉ 5 tin rác kiểm tra thì phải kiểm thêm trên dữ liệu khác mới tin được.

## 5. Checklist tuần 3

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
