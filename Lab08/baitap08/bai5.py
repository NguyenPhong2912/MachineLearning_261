# -*- coding: utf-8 -*-
"""Bài tập 5: DBSCAN trên bộ hai vầng trăng.

Với eps bằng 0.2, 0.3, 0.4 (min_samples=5), đếm số cụm và số điểm nhiễu (nhãn
-1), rồi so tỷ lệ khớp nhãn thật với K-Means. Thêm hai phần: dò eps nhỏ hơn,
và chạy lại trên hai vầng trăng ít nhiễu hơn (make_moons) để thấy vai trò của eps.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons

tr = pd.read_csv("data/hai_vang_trang.csv")
X = tr[["x1", "x2"]].to_numpy()
y = tr["nhan"].to_numpy()


def ty_le_khop(nhan_cum, y=y):
    """Moi cum doan theo nhan that chiem da so; diem nhieu (-1) tinh la sai."""
    dung = 0
    for c in np.unique(nhan_cum):
        if c == -1:
            continue
        dung += np.bincount(y[nhan_cum == c]).max()
    return dung / len(y)


km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
print(f"K-Means K = 2: ty le khop {ty_le_khop(km):.4f}")
print()

def bang_dbscan(X, y, cac_eps):
    print(" eps    so cum   so diem nhieu   ty le khop   co cau cum")
    for e in cac_eps:
        nhan = DBSCAN(eps=e, min_samples=5).fit_predict(X)
        so_cum = len(set(nhan)) - (1 if -1 in nhan else 0)
        nhieu = int((nhan == -1).sum())
        co_cau = np.bincount(nhan[nhan >= 0]).tolist() if so_cum else []
        print(f" {e:.2f}   {so_cum:6d}   {nhieu:13d}   {ty_le_khop(nhan, y):10.4f}   {co_cau}")
    print()


print("Bo hai vang trang cua bai 4, ba gia tri eps cua de:")
bang_dbscan(X, y, [0.2, 0.3, 0.4])
print("Do them eps nho hon:")
bang_dbscan(X, y, [0.10, 0.12, 0.15, 0.18])

# Khoảng cách tới láng giềng thứ 5 của mỗi điểm: gợi ý chọn eps
kc = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
kc5 = np.sort(kc, axis=1)[:, 5]                  # cột 0 là chính nó
print("Khoang cach toi lang gieng thu 5:")
print(f"  trung vi {np.median(kc5):.3f}, phan vi 90% {np.quantile(kc5, 0.9):.3f},"
      f" lon nhat {kc5.max():.3f}")
print()

# Cùng thí nghiệm trên hai vầng trăng ít nhiễu: giữa hai vầng có khe hở rõ
X_it, y_it = make_moons(n_samples=300, noise=0.05, random_state=0)
km_it = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X_it)
print(f"Hai vang trang it nhieu (noise=0.05): K-Means khop {ty_le_khop(km_it, y_it):.4f}")
bang_dbscan(X_it, y_it, [0.2, 0.3, 0.4])
