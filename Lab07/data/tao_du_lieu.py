# -*- coding: utf-8 -*-
"""Dựng lại tệp data/sinh_vien_bo_hoc.csv (số liệu mô phỏng).

Đề bài 7 không phát kèm tệp dữ liệu và cũng không cho mã sinh dữ liệu, nên
tệp này được dựng lại theo mô tả trong đề:
  - 1200 sinh viên, 4 cột đặc trưng và nhãn bo_hoc, đúng 119 bạn bỏ học;
  - năm dòng đầu giống hệt bảng df.head() in trong đề;
  - nguy cơ bỏ học tăng khi điểm thấp, vắng nhiều, nợ nhiều tín chỉ,
    làm thêm nhiều (quan hệ gần thẳng trên thang logit).
Hạt giống 2850 được chọn bằng cách thử nhiều hạt giống, lấy hạt giống cho
đúng ma trận nhầm lẫn của hồi quy logistic trong đề (TN 267, FP 3, FN 13,
TP 17). Các con số khác chỉ gần đúng với đề, xem README.

Chạy từ thư mục Lab07/: python data/tao_du_lieu.py
"""

import numpy as np
import pandas as pd

HAT_GIONG = 2850
N = 1200
# Năm dòng đầu in trong đề: diem_tb, so_buoi_vang, tin_chi_no, gio_lam_them, bo_hoc
NAM_DONG_DAU = [(8.0, 3, 0, 0, 0), (8.9, 5, 1, 7, 0), (3.3, 3, 1, 4, 0),
                (6.4, 3, 0, 7, 0), (7.9, 3, 5, 23, 0)]

rng = np.random.default_rng(HAT_GIONG)
# Điểm trung bình thang 10, làm tròn một chữ số
diem = np.clip(rng.normal(6.6, 1.5, N), 0, 10).round(1)
# Điểm càng thấp thì càng hay vắng và càng nợ nhiều tín chỉ
vang = rng.poisson(2.5 + 0.7 * np.clip(7 - diem, 0, None))
no = rng.poisson(0.6 + 0.5 * np.clip(6.5 - diem, 0, None))
# Khoảng 30% không làm thêm, còn lại từ 1 tới 30 giờ mỗi tuần
gio = np.where(rng.random(N) < 0.3, 0, rng.integers(1, 31, N))

# Nguy cơ bỏ học theo thang logit
z = -4.25 + 1.3 * (-1.1 * (diem - 6.5) + 0.35 * (vang - 4)
                   + 0.35 * no + 0.05 * (gio - 10))
xac_suat = 1 / (1 + np.exp(-z))
bo_hoc = (rng.random(N) < xac_suat).astype(int)

df = pd.DataFrame({"diem_tb": diem, "so_buoi_vang": vang, "tin_chi_no": no,
                   "gio_lam_them": gio, "bo_hoc": bo_hoc})
df.iloc[:5] = NAM_DONG_DAU
df = df.astype({"so_buoi_vang": int, "tin_chi_no": int,
                "gio_lam_them": int, "bo_hoc": int})
df.to_csv("data/sinh_vien_bo_hoc.csv", index=False)
print("Da ghi data/sinh_vien_bo_hoc.csv:", df.shape,
      ", so ban bo hoc:", int(df["bo_hoc"].sum()))
