# -*- coding: utf-8 -*-
"""Bài tập 2: tự tìm câu hỏi gốc trên dữ liệu thật.

Mở rộng f1_gini.py: thử mọi cột, mọi ngưỡng nằm giữa hai giá trị liền nhau,
rồi lấy câu hỏi làm Gini giảm nhiều nhất.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def gini(nhan):
    """Chi so Gini cua mot nut co hai lop."""
    if len(nhan) == 0:
        return 0.0
    p1 = nhan.mean()
    return 1 - p1 ** 2 - (1 - p1) ** 2


def gini_sau_chia(y, di_trai):
    """Gini trung binh co trong so cua hai nhanh."""
    trai, phai = y[di_trai], y[~di_trai]
    n = len(y)
    return len(trai) / n * gini(trai) + len(phai) / n * gini(phai)


def tim_cau_hoi(X, y):
    """Thu moi cot va moi nguong, tra ve (cot, nguong, Gini giam, bang tot nhat moi cot).

    X : DataFrame cac cot so
    y : nhan 0 / 1
    """
    y = np.asarray(y)
    g_goc = gini(y)
    tot = (None, None, -1.0)
    tung_cot = {}
    for cot in X.columns:
        gia_tri = X[cot].to_numpy()
        muc = np.unique(gia_tri)                     # đã sắp xếp tăng dần
        tot_cot = (None, -1.0)
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (a + b) / 2
            giam = g_goc - gini_sau_chia(y, gia_tri <= nguong)
            if giam > tot_cot[1]:
                tot_cot = (nguong, giam)
        tung_cot[cot] = (len(muc) - 1, *tot_cot)
        if tot_cot[1] > tot[2]:
            tot = (cot, tot_cot[0], tot_cot[1])
    return (*tot, tung_cot)


df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cot, nguong, giam, tung_cot = tim_cau_hoi(X_tr, y_tr)

print(f"Tap hoc {len(X_tr)} khach, Gini nut goc {gini(y_tr.to_numpy()):.4f}")
print()
print("Cau hoi tot nhat cua tung cot")
print("  cot                  so nguong thu   nguong tot   Gini giam")
for c, (so_nguong, ng, gm) in tung_cot.items():
    print(f"  {c:20s} {so_nguong:13d}   {ng:10.3f}   {gm:9.4f}")
print()
print(f"Cau hoi goc tu tim: {cot} <= {nguong:.3f}, Gini giam {giam:.4f}")

# So với câu hỏi gốc của cây scikit-learn
cay = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr)
cot_tv = X.columns[cay.tree_.feature[0]]
print(f"Cau hoi goc scikit-learn: {cot_tv} <= {cay.tree_.threshold[0]:.3f}")
print("Trung khop:", cot == cot_tv and np.isclose(nguong, cay.tree_.threshold[0]))
