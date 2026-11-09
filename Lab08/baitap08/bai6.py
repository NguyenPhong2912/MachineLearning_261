# -*- coding: utf-8 -*-
"""Bài tập 6: phân cụm chữ số viết tay bằng K-Means với K = 10.

Học không dùng nhãn; xong mới dùng nhãn thật lập bảng chéo nhãn thật và cụm.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

so = load_digits()
X, nhan_that = so.data, so.target
print("Kich thuoc:", X.shape, "(moi anh 8 x 8 = 64 diem anh, gia tri 0..16)")
print()

# Các cột cùng thang 0..16 nên không cần chuẩn hoá
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
cum = km.labels_

bang = pd.crosstab(nhan_that, cum, rownames=["chu_so"], colnames=["cum"])
print(bang.to_string())
print()

# Mỗi cụm gọi theo chữ số chiếm đa số trong cụm
print("Cum   so mau   chu so chiem da so   ty le")
ten_cum = {}
for c in range(10):
    cot = bang[c]
    ten_cum[c] = cot.idxmax()
    print(f"  {c}     {cot.sum():4d}          {cot.idxmax()}            "
          f"{cot.max() / cot.sum():.2f}")
print()

# Với mỗi chữ số: tỷ lệ nằm trong cụm lớn nhất của nó
print("Chu so   cum chua nhieu nhat   ty le gom gon")
for d in range(10):
    hang = bang.loc[d]
    print(f"   {d}            {hang.idxmax()}               {hang.max() / hang.sum():.2f}")
print()

du_doan = np.array([ten_cum[c] for c in cum])
print(f"Gan moi cum bang chu so da so: dung {(du_doan == nhan_that).mean():.4f}")
thieu = sorted(set(range(10)) - set(ten_cum.values()))
print("Chu so khong lam da so o cum nao:", thieu)
