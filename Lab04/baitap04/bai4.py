# -*- coding: utf-8 -*-
"""Bài tập 4: láng giềng có trọng số theo khoảng cách.

weights="distance" cho mỗi láng giềng một phiếu nặng 1 / khoảng cách,
nên láng giềng càng gần thì phiếu càng nặng.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Accuracy tren tap hoc va tap kiem tra")
print("    k   trong so     tap hoc   tap kiem tra")
for k in [1, 13, 101]:
    for w in ["uniform", "distance"]:
        mh = KNeighborsClassifier(n_neighbors=k, weights=w).fit(X_tr, y_tr)
        a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
        a_kt = accuracy_score(y_te, mh.predict(X_te))
        print(f"  {k:3d}   {w:9s}    {a_hoc:.4f}    {a_kt:.4f}")
print()

# Với k = 101, xem phiếu nặng dồn vào những láng giềng nào
mh = KNeighborsClassifier(n_neighbors=101, weights="distance").fit(X_tr, y_tr)
khoang, _ = mh.kneighbors(X_te)
trong_so = 1 / khoang
ty_phan = trong_so / trong_so.sum(axis=1, keepdims=True)
print("k = 101, weights='distance': phan phieu cua cac lang gieng gan nhat")
print(f"  13 lang gieng gan nhat chiem trung binh {ty_phan[:, :13].sum(axis=1).mean():.1%}"
      " tong phieu")
print(f"  lang gieng gan nhat nang gap trung binh "
      f"{np.median(trong_so[:, 0] / trong_so[:, -1]):.1f} lan lang gieng thu 101 (trung vi)")
print()

# Đếm số mẫu kiểm tra được cứu và bị làm hỏng khi bật trọng số ở k = 101
deu = KNeighborsClassifier(n_neighbors=101).fit(X_tr, y_tr).predict(X_te)
nang = mh.predict(X_te)
dung = y_te.to_numpy()
print("k = 101, so voi bo phieu deu tren 75 mau kiem tra")
print(f"  sai -> dung: {((deu != dung) & (nang == dung)).sum()}")
print(f"  dung -> sai: {((deu == dung) & (nang != dung)).sum()}")
