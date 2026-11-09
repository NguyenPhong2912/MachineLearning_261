# -*- coding: utf-8 -*-
"""Bài tập 2: thêm một điểm vào 24 điểm của mục 2 rồi học lại.

Lần 1 thêm điểm lớp 1 tại (-1, -1); lần 2 bỏ điểm đó, thay bằng điểm lớp 1
tại (4.5, 4). Xem lần nào đường ranh giới thay đổi.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y


def hoc_va_in(ten, X, y):
    svm = SVC(kernel="linear", C=1000).fit(X, y)
    w, b = svm.coef_[0], svm.intercept_[0]
    print(ten)
    print(f"  w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f},"
          f" do rong le = {2 / np.linalg.norm(w):.4f}")
    print("  vecto ho tro:", ", ".join(
        f"diem {i} ({X[i, 0]:.2f}, {X[i, 1]:.2f}) lop {y[i]}" for i in svm.support_))
    return svm


goc = hoc_va_in("24 diem ban dau", X, y)
print()

# Trước khi học lại, xem điểm mới nằm đâu so với đường ranh giới cũ
for diem in [(-1.0, -1.0), (4.5, 4.0)]:
    f = goc.decision_function([diem])[0]
    print(f"Diem moi {diem}, lop 1: f(x) theo duong cu = {f:+.3f}")
print()

# Lần 1: điểm lớp 1 tại (-1, -1), nằm sâu trong phía lớp 1
X_a = np.vstack([X, [[-1.0, -1.0]]])
y_a = np.append(y, 1)
hoc_va_in("Them diem lop 1 tai (-1, -1)  (diem moi la diem 24)", X_a, y_a)
print()

# Lần 2: bỏ điểm đó, thay bằng điểm lớp 1 tại (4.5, 4), nằm trong lề cũ
X_b = np.vstack([X, [[4.5, 4.0]]])
y_b = np.append(y, 1)
hoc_va_in("Thay bang diem lop 1 tai (4.5, 4)  (diem moi la diem 24)", X_b, y_b)
