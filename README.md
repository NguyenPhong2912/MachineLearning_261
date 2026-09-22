# Bài thực hành 1: Hồi quy tuyến tính

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Bài** | Bài 1 — Hồi quy tuyến tính |
| **Môi trường** | Google Colab (Python 3, numpy / pandas / matplotlib / scikit-learn) |

---

## 1. Nội dung nộp

| Tệp | Nội dung |
|---|---|
| `bai1.py` | Lọc nhóm căn hộ có diện tích lớn hơn 100 m² |
| `bai2.py`, `bai2.png` | Biểu đồ phân tán giá theo số phòng ngủ |
| `bai3.py` | Hồi quy giá theo tuổi nhà bằng công thức bình phương tối thiểu |
| `bai4.py` | Mô hình hai biến: diện tích và số phòng |
| `bai5.py` | So sánh ba tốc độ học của gradient descent |
| `bai6.py` | Hàm dự đoán giá kèm cảnh báo ngoại suy |
| `anh_chup/` | Ảnh chụp màn hình kết quả của từng bài |

Dữ liệu dùng chung cho cả sáu bài là `data/gia_nha.csv`, gồm 60 căn hộ và 4 cột,
đúng bộ dữ liệu được phát kèm bài thực hành. Dữ liệu không bị sửa đổi.

## 2. Cách chạy

Mọi lệnh chạy từ thư mục gốc của bài thực hành, không `cd` vào `baitap01`:

```
cd C:\MayHoc\bai01_hoi_quy
python baitap01/bai1.py
```

Trong mã, đường dẫn dữ liệu viết là `data/gia_nha.csv`, tính tương đối theo thư
mục đang đứng lúc gõ lệnh.

Bài làm được thực hiện trên Google Colab nên bốn thư viện đã có sẵn, không cần
tạo môi trường ảo. Khi chạy lại bằng `python` trên cửa sổ lệnh Windows, các câu
`print` đều viết không dấu để tránh lỗi mã hóa.

## 3. Kết quả chính

### Phần nội dung bài giảng

| Đại lượng | Giá trị |
|---|---|
| Hệ số góc w (một biến, 60 căn) | 0.078367 |
| Hệ số chặn b (một biến, 60 căn) | 0.401752 |
| MSE trên toàn bộ 60 căn | 0.1790 |
| Dự đoán căn 80 m² | 6.671 tỷ đồng |
| Dự đoán căn 100 m² | 8.238 tỷ đồng |
| Số căn tập học / tập kiểm tra | 48 / 12 |
| MAE trên tập kiểm tra | 0.2578 tỷ đồng |
| RMSE trên tập kiểm tra | 0.3730 tỷ đồng |
| R² tập kiểm tra, mô hình một biến | 0.9622 |
| R² tập kiểm tra, mô hình ba biến | 0.9761 |

Ba cách tìm w và b đều cho cùng một kết quả: công thức bình phương tối thiểu
tính tay, thư viện scikit-learn, và gradient descent sau khi đổi về thang mét
vuông. Cả ba khớp nhau tới chữ số thứ sáu sau dấu chấm.

### Phần bài tập

| Bài | Kết quả |
|---|---|
| 1 | 10 căn có diện tích trên 100 m², giá trung bình 8.9620 tỷ đồng |
| 2 | Giá trung bình theo số phòng: 4.58 / 5.57 / 7.45 / 8.92 tỷ đồng |
| 3 | w = −0.037858, b = 6.983593 |
| 4 | R² mô hình hai biến = 0.9698 |
| 5 | MSE sau 200 vòng: 20.2175 (lr = 0.001), 0.1790 (lr = 0.1), 290391782.0192 (lr = 1.02) |
| 6 | 60 m² → 5.104 tỷ; 80 m² → 6.671 tỷ; 200 m² → 16.075 tỷ kèm cảnh báo |

## 4. Nhận xét

**Bài 1.** Nhóm căn trên 100 m² chỉ có 10 căn trong tổng số 60, nhưng giá trung
bình 8.9620 tỷ cao hơn hẳn mức 6.4933 tỷ của cả bộ dữ liệu. Điều này khớp với xu
hướng đi lên đã thấy ở biểu đồ phân tán ban đầu. Tuy vậy 10 căn là một cỡ mẫu
nhỏ, nên chưa đủ để kết luận chắc chắn điều gì riêng về phân khúc căn hộ lớn.

**Bài 2.** Giá trung bình tăng đều theo số phòng ngủ, nhưng vì `so_phong` chỉ
nhận bốn giá trị nguyên nên các điểm chồng thành bốn cột dọc thay vì trải đều
như biểu đồ diện tích. Quan trọng hơn, các cột chồng lấn nhau: nhóm 2 phòng có
căn tới 7.6 tỷ trong khi nhóm 3 phòng có căn chỉ 5.14 tỷ. Biết số phòng thì chỉ
đoán được giá một cách rất thô.

**Bài 3.** Hệ số góc mang dấu âm, nghĩa là nhà càng cũ thì giá càng thấp, đúng
với trực giác thông thường. Nhưng biểu đồ phân tán cho thấy các điểm tản gần như
đều khắp chứ không bám quanh đường thẳng: có căn 18 năm tuổi giá 3 tỷ và căn 18
năm tuổi khác giá 9.2 tỷ. Vậy một hệ số có dấu hợp lý chưa đủ để kết luận hai
đại lượng có quan hệ chặt. Công thức bình phương tối thiểu luôn trả về một con số
w kể cả khi dữ liệu không có xu hướng rõ ràng, nên phải vẽ hình ra xem trước khi
tin vào con số đó.

**Bài 4.** Thêm `so_phong` làm R² tăng từ 0.9622 lên 0.9698, và thêm tiếp
`tuoi_nha` lên 0.9761. Mức tăng nhỏ vì diện tích đã giải thích gần hết biến động
của giá. Đáng chú ý là hệ số của `dien_tich` giảm khi `so_phong` được đưa vào, do
căn rộng thường có nhiều phòng ngủ hơn nên ở mô hình một biến hệ số diện tích
gánh luôn phần đóng góp của số phòng.

**Bài 5.** Tốc độ học 0.001 bước quá ngắn nên sau 200 vòng sai số vẫn còn
20.2175, chưa tới đáy. Tốc độ 0.1 về đúng 0.1790 và ổn định từ khoảng vòng 50.
Tốc độ 1.02 bước quá dài nên từ sườn bên này nhảy vọt qua đáy sang sườn bên kia,
sai số tăng theo cấp số nhân và phình lên 290 triệu, tức gấp khoảng 6.5 triệu lần
điểm xuất phát. Một tham số chỉ nhỉnh hơn 1 một chút đã biến thuật toán từ hoạt
động tốt thành vô dụng hoàn toàn.

**Bài 6.** Mô hình trả về 16.075 tỷ cho căn 200 m² mà không báo lỗi gì, nhưng căn
lớn nhất trong dữ liệu mới chỉ 117.5 m² nên không có căn nào để đối chiếu con số
đó. Mô hình không học được gì về loại nhà đó, nó chỉ kéo dài đường thẳng đã học
ra xa thêm. Vì vậy hàm dự đoán cần in cảnh báo khi diện tích nằm ngoài khoảng
35.5 tới 117.5 m².
