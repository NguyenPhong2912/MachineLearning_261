# -*- coding: utf-8 -*-
"""Bài tập 6: đo thời gian .fit và .predict của KNN theo cỡ dữ liệu.

Thời gian tuyệt đối tuỳ máy, chỉ so tương đối giữa các cỡ.
Đặt algorithm="brute" để KNN không dựng cây tìm kiếm lúc fit, đúng với mô tả
"chỉ cất dữ liệu vào bộ nhớ". Dòng cuối chạy thêm mặc định "auto" để so.
"""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
X_moi = rng.random((1000, 2))           # 1000 mẫu mới cần dự đoán
SO_LAN = 7                              # đo nhiều lần, lấy lần nhanh nhất


def do_thoi_gian(ham):
    """Tra ve thoi gian nhanh nhat (giay) qua SO_LAN lan chay."""
    ham()                               # chạy nháp một lần: lần đầu luôn chậm
    tot = float("inf")
    for _ in range(SO_LAN):
        bd = time.perf_counter()
        ham()
        tot = min(tot, time.perf_counter() - bd)
    return tot


# Chạy nháp cả hai cách một lượt cho máy "nóng" lên trước khi đo thật
for algo in ["brute", "auto"]:
    X = rng.random((20000, 2))
    mh = KNeighborsClassifier(n_neighbors=5, algorithm=algo).fit(X, X[:, 0] > 0.5)
    for _ in range(20):
        mh.predict(X_moi)

for algo in ["brute", "auto"]:
    print(f"algorithm = '{algo}', k = 5, du doan 1000 mau moi")
    print("     so mau      fit (ms)   predict (ms)   predict / fit")
    for n in [1000, 10000, 100000]:
        X = rng.random((n, 2))
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        mh = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
        t_fit = do_thoi_gian(lambda: mh.fit(X, y))
        t_pre = do_thoi_gian(lambda: mh.predict(X_moi))
        print(f"  {n:9d}   {t_fit * 1000:10.2f}   {t_pre * 1000:12.2f}"
              f"   {t_pre / t_fit:12.0f}")
    print()
