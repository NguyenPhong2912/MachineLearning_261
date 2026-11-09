# -*- coding: utf-8 -*-
"""Dựng lại tệp data/khach_hang.csv (đề không phát kèm tệp này).

Đề chỉ cho bảng describe, tâm, số khách và tuổi trung bình của năm cụm. Tệp này
sinh năm nhóm khách theo phân phối chuẩn quanh đúng năm tâm đó, co giãn độ lệch
trong cụm để inertia (đã chuẩn hoá) gần 23.0, rồi làm tròn như đề. Hạt giống 12993
và thứ tự xáo trộn 3200 được chọn bằng cách dò sao cho bảng describe, bảng chọn K,
thứ tự số cụm và bẫy khởi tạo ngẫu nhiên gần với đề nhất. Số liệu mô phỏng.

Chạy từ thư mục Lab08/: python data/tao_khach_hang.py
"""

import numpy as np
import pandas as pd

# (thu nhập, điểm chi tiêu, số khách, tuổi trung bình) của năm cụm, theo thứ tự trong đề
CUM = [(7.5, 80.2, 35, 24.9), (30.1, 20.0, 35, 42.9), (17.4, 49.3, 60, 40.4),
       (8.0, 18.9, 40, 44.9), (30.2, 81.5, 40, 32.6)]
DL_THU_NHAP = [2.4, 2.4, 2.0, 2.4, 2.4]       # độ lệch chuẩn ban đầu trong từng cụm
DL_CHI_TIEU = [7.0, 6.5, 5.5, 6.5, 7.0]
DL_TUOI = [4.5, 8.5, 9.0, 9.0, 6.0]
INERTIA_MUC_TIEU = 23.02

rng = np.random.default_rng(12993)
lech = []
for j, (_, _, n, _) in enumerate(CUM):
    a = rng.normal(0, DL_THU_NHAP[j], n)
    b = rng.normal(0, DL_CHI_TIEU[j], n)
    lech.append((a - a.mean(), b - b.mean()))

# Co giãn độ lệch trong cụm để tổng bình phương (trên thang chuẩn hoá) bằng mục tiêu
tam = np.array([[c[0], c[1]] for c in CUM])
for _ in range(3):
    toan_bo = np.vstack([np.column_stack([d[0] + tam[j, 0], d[1] + tam[j, 1]])
                         for j, d in enumerate(lech)])
    dlc = toan_bo.std(axis=0)
    w = sum(((d[0] / dlc[0]) ** 2 + (d[1] / dlc[1]) ** 2).sum() for d in lech)
    he_so = np.sqrt(INERTIA_MUC_TIEU / w)
    lech = [(d[0] * he_so, d[1] * he_so) for d in lech]

dong = []
for j, (tn, ct, n, tuoi_tb) in enumerate(CUM):
    thu_nhap = np.clip(lech[j][0] + tn, 3.0, 35.5).round(1)
    chi_tieu = np.clip(lech[j][1] + ct, 7, 99).round().astype(int)
    tuoi = rng.normal(0, DL_TUOI[j], n)
    tuoi = np.clip(tuoi - tuoi.mean() + tuoi_tb, 18, 62).round().astype(int)
    dong += list(zip(tuoi, thu_nhap, chi_tieu))

df = pd.DataFrame(dong, columns=["tuoi", "thu_nhap", "diem_chi_tieu"])
df = df.sample(frac=1, random_state=3200).reset_index(drop=True)
df.to_csv("data/khach_hang.csv", index=False)
print("Da ghi data/khach_hang.csv:", df.shape)
