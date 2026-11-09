# Bài thực hành 8: Phân cụm K-Means

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 8 — Phân cụm K-Means |
| **Tuần** | 8 |
| **Môi trường** | Google Colab / Python 3 (numpy / pandas / matplotlib / scikit-learn 1.9) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `Lab08_KMeans.ipynb` | Notebook: phần bài giảng k1–k5 (kèm hình 1–8) và sáu bài tập (thêm hình 9–11), có nhận xét dưới từng bài |
| `code/k1_tu_viet.py` … `code/k5_nen_anh.py` | Năm tệp mã của buổi thực hành (bảng 2 trong đề) |
| `baitap08/bai1.py` … `baitap08/bai6.py` | Mã nguồn sáu bài tập |
| `anh_goc.png`, `anh_16_mau.png`, `anh_rieng.png`, `anh_rieng_16_mau.png` | Bốn ảnh do bài tập 4 lưu ra |
| `data/khach_hang.csv` | 210 khách hàng, 3 cột `tuoi`, `thu_nhap`, `diem_chi_tieu`, không có nhãn (dựng lại, xem ghi chú bên dưới) |
| `data/tao_khach_hang.py` | Mã dựng lại `khach_hang.csv` |
| `data/hai_vang_trang.csv` | Dùng lại từ bài 4: 300 điểm, cột `x1`, `x2`, `nhan` |
| `Lab08_KMeans.pdf` | Đề bài |

**Ghi chú về dữ liệu.** Đề không phát kèm `khach_hang.csv`, cũng không có mã sinh dữ liệu. Tệp được dựng lại bằng `data/tao_khach_hang.py` (số liệu mô phỏng): năm nhóm khách sinh theo phân phối chuẩn quanh đúng năm tâm, số khách và tuổi trung bình đề cho, co giãn độ lệch trong cụm để inertia gần 23.0, rồi dò hạt giống và thứ tự dòng sao cho bảng `describe`, bảng chọn K, thứ tự số cụm và bẫy khởi tạo gần với đề nhất. Các con số chính trùng đề; các ô còn lại lệch nhẹ (ghi rõ ở mục 3). Ảnh `china.jpg`, `flower.jpg` và bộ `load_digits` có sẵn trong scikit-learn.

## 2. Cách chạy

Chạy từ thư mục `Lab08/`, đúng như đề yêu cầu (trong mã vẫn viết `data/...`):

```
python code/k1_tu_viet.py
...
python code/k5_nen_anh.py
python baitap08/bai1.py
...
python baitap08/bai6.py
```

Muốn sinh lại tệp dữ liệu: `python data/tao_khach_hang.py` (ra đúng tệp đang có).

- **Colab:** mở notebook, chạy ô đầu tiên rồi chọn cả `khach_hang.csv` và `hai_vang_trang.csv` để tải lên. Notebook tự đặt hai tệp vào `data/`.
- **Máy cá nhân:** mở notebook ngay trong thư mục `Lab08/`, chọn `Restart and run all`.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị | Đề (bảng 5) |
|---|---|---|
| Bảy điểm, số vòng lặp tới khi dừng | 3 | 3 |
| Bảy điểm, inertia sau cùng (tự tính và scikit-learn) | 8.5250 | 8.5250 |
| Khách hàng, inertia với K = 5 | 23.0 (22.98) | 23.0 (23.02) |
| Silhouette với K = 4, 5, 6 | 0.624, 0.723, 0.661 | 0.626, 0.723, 0.651 |
| Khởi tạo ngẫu nhiên một lần, số lần tệ trên 20 | 10 | 10 |
| Hai vầng trăng, tỷ lệ khớp nhãn thật | 0.7500 | 0.7500 |
| Ảnh 16 màu, nhỏ hơn bao nhiêu lần | 6.0 | 6.0 |
| Ảnh 16 màu, sai lệch trung bình | 7.97 mức sáng | 7.97 mức sáng |

Phần bảy điểm và phần nén ảnh khớp đề từng chữ số. Năm cụm khách hàng khớp đề cả số thứ tự, số khách (35, 35, 60, 40, 40) và cách xếp hai khách mới (cụm 4 và cụm 0); tâm và tuổi trung bình lệch tối đa 0.1. Những chỗ lệch nhẹ, đều do dữ liệu dựng lại không trùng từng khách với tệp của đề:

- `describe`: độ lệch chuẩn của tuổi 10.4 (đề 10.3), trung vị và phân vị 75% của tuổi 38 và 45 (đề 39 và 46), phân vị 25% của thu nhập 8.9 (đề 8.8), phân vị 75% của điểm chi tiêu 76.8 (đề 77).
- Bảng chọn K: mức giảm từ K = 4 lên 5 là 52.2 (đề 50.7), từ 5 lên 6 là 2.8 (đề 2.1); silhouette ở K = 8, 9, 10 là 0.491, 0.497, 0.357 (đề 0.454, 0.471, 0.425). K = 5 vẫn là lựa chọn rõ ràng của cả hai công cụ.
- Bẫy 1 (không chuẩn hoá): năm tâm có điểm chi tiêu từ 45.7 tới 54.0 (đề 41.9 tới 52.9); kết luận giống hệt.
- Bẫy 2: mười lần tệ có inertia từ 73.2 tới 120.0 (đề 72.4 tới 118.2).

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | Silhouette tự tính 0.723440, trùng `silhouette_score` (lệch 1.1e-16); khách s nhỏ nhất (26.5 triệu, chi 64, cụm 4, s = 0.273) nằm ở rìa dưới bên trái cụm 4, nhìn về cụm giữa; không khách nào có s âm |
| 2 | Thêm cột tuổi, silhouette chọn K = 6 (0.522, sát K = 5 với 0.511); nhóm thu nhập thấp chi ít tách đôi theo tuổi (52 và 36 tuổi), 7 khách trẻ nhất nhóm giữa chuyển sang nhóm trẻ chi nhiều |
| 3 | `init="random"` đạt 23.0 trong 24/50 lần (tệ nhất 157.5), `k-means++` 49/50 lần (tệ nhất 72.3), hội tụ 2.5 vòng so với 5.8 |
| 4 | `anh_goc.png` 538.4 KB, `anh_16_mau.png` 151.8 KB (nhỏ hơn 3.5 lần); ảnh PNG đọc bằng `plt.imread` có dạng (427, 640, 4), bỏ kênh trong suốt rồi nén 16 màu |
| 5 | DBSCAN với eps 0.2 / 0.3 / 0.4: 1 cụm với 40 / 10 / 3 điểm nhiễu, đều kém K-Means (0.7500); trên vầng trăng ít nhiễu, eps 0.2 và 0.3 tách đúng 100% |
| 6 | K = 10 trên `load_digits`: 0, 6, 7, 4 được gom gọn (tỷ lệ 0.91 tới 0.99); 1 và 8 chung một cụm (100 và 100), 9, 8, 5 chung một cụm; gán cụm theo đa số đúng 0.7919 |

## 4. Nhận xét

Nhận xét chi tiết từng mục nằm ngay dưới ô mã tương ứng trong notebook. Tóm tắt:

**Bài 1.** s đo một điểm nằm sâu trong cụm của mình (gần 1) hay ở ranh giới (gần 0). Những khách s nhỏ là khách ở rìa cụm, nằm trong khoảng trống giữa một cụm góc và cụm giữa. Muốn trùng scikit-learn phải bỏ chính điểm i khi tính a và tính b là trung bình tới từng cụm khác, không phải tới tâm.

**Bài 2.** Tuổi trải đều trong mỗi nhóm, không có cấu trúc cụm riêng, nên thêm cột tuổi làm mọi khoảng cách trong cụm lớn lên và silhouette tụt từ 0.72 xuống khoảng 0.5. K-Means bắt đầu chia theo tuổi ở chỗ thu nhập và chi tiêu không còn khác nhau. Cột tuổi trong dữ liệu dựng lại chỉ được khớp theo trung bình, nên bài này nhạy với phần dựng lại nhất.

**Bài 3.** Khởi tạo ngẫu nhiên hay đặt hai tâm vào cùng một nhóm, rồi kẹt ở cực tiểu cục bộ. k-means++ chọn tâm mới với xác suất tỷ lệ bình phương khoảng cách tới tâm gần nhất, nên năm tâm gần như luôn rơi vào năm nhóm khác nhau.

**Bài 4.** Tệp PNG nhỏ hơn 3.5 lần chứ không phải 6 lần, vì `plt.imsave` vẫn ghi đủ 4 kênh 8 bit; phần nhỏ đi là nhờ bộ nén của PNG gặp ít màu. Ảnh PNG đọc bằng `plt.imread` có kênh thứ tư (độ trong suốt), phải bỏ trước khi phân cụm. Phần "ảnh của chính em" dùng `flower.jpg` của scikit-learn thay cho ảnh cá nhân.

**Bài 5.** Bộ hai vầng trăng của bài 4 nhiễu nhiều, hai vầng chạm nhau, không có khe hở mật độ thấp nên DBSCAN nối chúng thành một cụm với mọi eps của đề. Trên dữ liệu ít nhiễu, DBSCAN đi theo được hình cong mà K-Means không làm được. eps phải lớn hơn khoảng cách trong cụm nhưng nhỏ hơn khe hở giữa các cụm.

**Bài 6.** K-Means đo khoảng cách theo từng điểm ảnh, nên các chữ số có hình dạng riêng (0, 6, 7, 4) được gom gọn, còn 1 và 8 hay 9, 8, 5 bị gom chung. Số 8 không làm đa số ở cụm nào.

## 5. Checklist tuần 8

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
