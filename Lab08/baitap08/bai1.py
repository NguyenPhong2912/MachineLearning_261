# -*- coding: utf-8 -*-
"""Bài tập 1: tự tính silhouette bằng numpy.

Hàm silhouette_mot_diem(X, nhan, i) tính a, b, s của điểm thứ i đúng như định
nghĩa ở mục 4; chạy cho mọi khách với K = 5 rồi so với silhouette_score.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
X = df[["thu_nhap", "diem_chi_tieu"]]
sc = StandardScaler().fit(X)
Xs = sc.transform(X)
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
nhan = km.labels_


def silhouette_mot_diem(X, nhan, i):
    """Tra ve (a, b, s) cua diem thu i."""
    kc = np.sqrt(((X - X[i]) ** 2).sum(axis=1))      # khoảng cách từ i tới mọi điểm
    cung = (nhan == nhan[i])
    cung[i] = False                                  # không tính khoảng cách tới chính nó
    if cung.sum() == 0:                              # cụm chỉ có một điểm: quy ước s = 0
        return 0.0, 0.0, 0.0
    a = kc[cung].mean()
    # b: khoảng cách trung bình tới cụm khác gần nhất
    b = min(kc[nhan == k].mean() for k in np.unique(nhan) if k != nhan[i])
    s = (b - a) / max(a, b)
    return a, b, s


ket = np.array([silhouette_mot_diem(Xs, nhan, i) for i in range(len(Xs))])
s = ket[:, 2]
print(f"Silhouette tu tinh  : {s.mean():.6f}")
print(f"silhouette_score    : {silhouette_score(Xs, nhan):.6f}")
print(f"Chenh lech tuyet doi: {abs(s.mean() - silhouette_score(Xs, nhan)):.2e}")
print()

# Trung bình s theo từng cụm, kèm tâm cụm theo đơn vị gốc
tam = sc.inverse_transform(km.cluster_centers_)
print("Cum   tam (thu nhap, chi tieu)   so khach   s trung binh   s nho nhat")
for k in range(5):
    sk = s[nhan == k]
    print(f"  {k}   ({tam[k, 0]:5.1f}, {tam[k, 1]:5.1f})"
          f"            {len(sk):3d}       {sk.mean():.3f}        {sk.min():6.3f}")
print()

# Năm khách có s nhỏ nhất
print("Nam khach co s nho nhat:")
print("  dong   thu nhap   chi tieu   cum       a       b        s   cum gan nhat")
for i in np.argsort(s)[:5]:
    a, b, si = ket[i]
    # cụm khác gần nhất là cụm cho ra b
    gan = min((k for k in range(5) if k != nhan[i]),
              key=lambda k: np.sqrt(((Xs[nhan == k] - Xs[i]) ** 2).sum(axis=1)).mean())
    print(f"  {i:4d}   {X.iloc[i, 0]:8.1f}   {X.iloc[i, 1]:8d}   {nhan[i]:3d}"
          f"   {a:5.3f}   {b:5.3f}   {si:6.3f}   {gan:5d}")
print()
print("So khach co s am:", int((s < 0).sum()))
