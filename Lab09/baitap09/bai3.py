# -*- coding: utf-8 -*-
"""Bài tập 3: so các hàm kích hoạt trên bộ hai vầng trăng.

Mạng hai lớp ẩn 100 nơ-ron, alpha=1.0, như m3_do_lon_mang.py; lần lượt đặt
activation bằng "logistic", "tanh" và "relu".
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def mang(ham):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1.0,
                      activation=ham, solver="lbfgs", max_iter=20000,
                      random_state=0),
    )


print("Ham kich hoat   tap hoc   tap kiem tra   kiem dinh cheo   so vong")
for ham in ["logistic", "tanh", "relu"]:
    m = mang(ham).fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, m.predict(X_tr))
    a_kt = accuracy_score(y_te, m.predict(X_te))
    kdc = cross_val_score(mang(ham), X_tr, y_tr, cv=5).mean()
    print(f"  {ham:12s}  {a_hoc:.4f}    {a_kt:.4f}         {kdc:.4f}"
          f"          {m[-1].n_iter_}")
