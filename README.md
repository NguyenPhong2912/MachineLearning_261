# Học máy ứng dụng — Bài thực hành

| | |
|---|---|
| **Sinh viên** | Nguyễn Thành Phong |
| **Học phần** | Học máy ứng dụng |
| **Giảng viên** | ThS. Nguyễn Thái Anh |
| **Môi trường** | Google Colab (Python 3, numpy / pandas / matplotlib / scikit-learn) |

---

## 1. Cấu trúc repo

```
MachineLearning_261/
├── README.md          # Tệp này: thông tin chung + tiến độ theo tuần
├── Lab01/             # Bài 1 — Hồi quy tuyến tính
├── Lab02/             # Bài 2 — Hồi quy logistic
├── Lab03/
├── Lab04/
├── Lab05/
├── Lab06/
├── Lab07/
├── Lab08/
├── Lab09/
└── Lab10/
```

Mỗi thư mục `LabXX/` theo cùng một bố cục:

```
LabXX/
├── README.md          # Đề bài, cách chạy, kết quả, nhận xét, checklist
├── *.ipynb            # Notebook làm trên Colab
├── baiN.py            # Mã từng bài tập (nếu đề yêu cầu)
├── data/              # Dữ liệu được phát kèm bài
└── anh_chup/          # Ảnh chụp màn hình kết quả
```

## 2. Tiến độ theo tuần

Đánh dấu `[x]` khi xong. Cột **Nộp** chỉ đánh dấu sau khi đã `git push` lên GitHub.

| Tuần | Lab | Chủ đề | Notebook | Bài tập | README | Ảnh kết quả | Nộp |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | [Lab01](Lab01/) | Hồi quy tuyến tính | [x] | [x] | [x] | [ ] | [x] |
| 2 | [Lab02](Lab02/) | Hồi quy logistic | [x] | [x] | [x] | [ ] | [x] |
| 3 | [Lab03](Lab03/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 4 | [Lab04](Lab04/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 5 | [Lab05](Lab05/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 6 | [Lab06](Lab06/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 7 | [Lab07](Lab07/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 8 | [Lab08](Lab08/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 9 | [Lab09](Lab09/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |
| 10 | [Lab10](Lab10/) | _(cập nhật khi có đề)_ | [ ] | [ ] | [ ] | [ ] | [ ] |

Checklist chi tiết của từng tuần nằm ở cuối `README.md` trong thư mục lab tương ứng.

## 3. Quy trình nộp mỗi tuần

1. Làm bài trên Google Colab, chạy lại toàn bộ notebook từ đầu (`Runtime → Restart and run all`) để chắc chắn mọi ô đều có kết quả.
2. Tải notebook về (`File → Download → .ipynb`) và đặt vào `LabXX/`.
3. Chép dữ liệu vào `LabXX/data/`, ảnh chụp kết quả vào `LabXX/anh_chup/`.
4. Điền `LabXX/README.md`: kết quả chính và nhận xét từng bài.
5. Đánh dấu checklist trong `LabXX/README.md` và bảng tiến độ ở trên.
6. Đẩy lên GitHub:

   ```
   git add LabXX README.md
   git commit -m "update"
   git push
   ```
