# -*- coding: utf-8 -*-
"""Bài tập 1: tự viết KNN không dùng thư viện.

Dựa vào e1_khoang_cach.py: tính khoảng cách Euclid từ mẫu mới tới mọi mẫu
của tập học, lấy k mẫu gần nhất bằng np.argsort rồi cho chúng bỏ phiếu.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Tra ve nhan du doan cho mot mau moi.

    X_hoc : mang (n, d) cac mau da biet nhan
    y_hoc : mang (n,) nhan 0 / 1 cua tung mau
    x_moi : mang (d,) mau can doan
    k     : so lang gieng duoc hoi
    """
    khoang = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    # kind="stable": hai mẫu cách đều nhau thì mẫu đứng trước được chọn trước,
    # giống cách scikit-learn xử lý chỗ hoà
    gan = np.argsort(khoang, kind="stable")[:k]
    phieu = np.bincount(y_hoc[gan], minlength=2)
    return int(np.argmax(phieu))    # hoà phiếu thì argmax lấy lớp nhỏ hơn


df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Cùng cách chia với e3_chon_k.py để so được với KNeighborsClassifier
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
X_hoc, y_hoc = X_tr.to_numpy(), y_tr.to_numpy()
X_kt = X_te.to_numpy()

K = 13
du_tay = np.array([knn_du_doan(X_hoc, y_hoc, x, K) for x in X_kt])
du_tv = KNeighborsClassifier(n_neighbors=K).fit(X_tr, y_tr).predict(X_te)

print(f"KNN tu viet, k = {K}")
print(f"  Accuracy tu viet              : {accuracy_score(y_te, du_tay):.4f}")
print(f"  Accuracy KNeighborsClassifier: {accuracy_score(y_te, du_tv):.4f}")
print(f"  So mau doan khac nhau         : {(du_tay != du_tv).sum()} / {len(du_tay)}")
print()

# Đi chậm qua một mẫu kiểm tra để thấy ba bước của thuật toán
x0 = X_kt[0]
khoang0 = np.sqrt(((X_hoc - x0) ** 2).sum(axis=1))
gan0 = np.argsort(khoang0, kind="stable")[:K]
print(f"Mau kiem tra dau tien: x1 = {x0[0]:.4f}, x2 = {x0[1]:.4f}, "
      f"nhan that = {y_te.iloc[0]}")
print(f"  {K} lang gieng gan nhat co nhan: {y_hoc[gan0].tolist()}")
print(f"  Bo phieu: {(y_hoc[gan0] == 0).sum()} phieu lop 0, "
      f"{(y_hoc[gan0] == 1).sum()} phieu lop 1 -> doan {du_tay[0]}")
print()

# Kiểm thêm vài giá trị k khác, kể cả k chẵn có thể hoà phiếu
print("  k   tu viet   thu vien   so mau khac")
for k in [1, 4, 5, 13, 31, 101]:
    tay = np.array([knn_du_doan(X_hoc, y_hoc, x, k) for x in X_kt])
    tv = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)
    print(f"  {k:3d}  {accuracy_score(y_te, tay):.4f}    "
          f"{accuracy_score(y_te, tv):.4f}     {(tay != tv).sum()}")
