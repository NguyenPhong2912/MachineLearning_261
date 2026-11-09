# -*- coding: utf-8 -*-
"""Bài tập 3: so khởi tạo ngẫu nhiên và k-means++.

Chạy K-Means 50 lần với n_init=1, random_state từ 0 tới 49, cho mỗi cách khởi
tạo; đếm số lần đạt inertia 23.0.
"""

from collections import Counter

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

tot_nhat = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs).inertia_
print(f"Inertia tot nhat (n_init=10): {tot_nhat:.2f}")
print()

ty_le = {}
for cach in ["random", "k-means++"]:
    ket = []
    vong = []
    for rs in range(50):
        km = KMeans(n_clusters=5, init=cach, n_init=1, random_state=rs).fit(Xs)
        ket.append(round(km.inertia_, 1))
        vong.append(km.n_iter_)
    dat = sum(abs(x - round(tot_nhat, 1)) < 0.05 for x in ket)
    ty_le[cach] = dat / 50
    print(f"init='{cach}': {dat}/50 lan dat inertia {tot_nhat:.1f}")
    print(f"  inertia tre nhat {max(ket):.1f}, trung binh {sum(ket) / 50:.1f},"
          f" so vong lap trung binh {sum(vong) / 50:.1f}")
    print("  cac muc inertia (gia tri: so lan):",
          dict(sorted(Counter(ket).items())))
    print()

# Xác suất có ít nhất một lần tốt khi chạy n_init lần độc lập, ước từ tỷ lệ trên
for cach, p in ty_le.items():
    print(f"init='{cach}': p = {p:.2f}, n_init=10 -> P(it nhat 1 lan tot) ="
          f" {1 - (1 - p) ** 10:.4f}")
