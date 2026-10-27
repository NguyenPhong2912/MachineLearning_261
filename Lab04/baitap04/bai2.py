# -*- coding: utf-8 -*-
"""Bài tập 2: chạy lại e3_chon_k.py với khoảng cách Manhattan.

Khoảng cách Manhattan cộng trị tuyệt đối chênh lệch từng cột,
|x1 - z1| + |x2 - z2|, thay vì căn của tổng bình phương như Euclid.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def chon_k(metric):
    """Chon k bang kiem dinh cheo 5 phan tren tap hoc, roi cham tap kiem tra."""
    diem = {}
    for k in range(1, 52, 2):
        mh = KNeighborsClassifier(n_neighbors=k, metric=metric)
        diem[k] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    k_tot = max(diem, key=diem.get)
    mh = KNeighborsClassifier(n_neighbors=k_tot, metric=metric).fit(X_tr, y_tr)
    return diem, k_tot, accuracy_score(y_te, mh.predict(X_te))


diem_e, k_e, acc_e = chon_k("euclidean")
diem_m, k_m, acc_m = chon_k("manhattan")

print("Accuracy tren tap hoc va tap kiem tra theo k, khoang cach Manhattan")
print("    k    tap hoc   tap kiem tra")
for k in [1, 5, 13, 31, 101]:
    mh = KNeighborsClassifier(n_neighbors=k, metric="manhattan").fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
    a_kt = accuracy_score(y_te, mh.predict(X_te))
    print(f"  {k:3d}   {a_hoc:.4f}     {a_kt:.4f}")
print()

print("Kiem dinh cheo 5 phan tren tap hoc:")
print("   k    Euclid   Manhattan")
for k in [1, 5, 9, 13, 17, 25, 51]:
    print(f"  {k:2d}    {diem_e[k]:.4f}   {diem_m[k]:.4f}")
print()

print("Tong ket          k tot nhat   diem KDC   tap kiem tra")
print(f"  Euclid        {k_e:6d}       {diem_e[k_e]:.4f}     {acc_e:.4f}")
print(f"  Manhattan     {k_m:6d}       {diem_m[k_m]:.4f}     {acc_m:.4f}")
