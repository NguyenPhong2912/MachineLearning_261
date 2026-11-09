# -*- coding: utf-8 -*-
"""Bài tập 1: tự viết hàm đánh giá chỉ bằng numpy.

Chạy cho mô hình lười và hồi quy logistic của mục 2, so với scikit-learn,
và xem precision của mô hình lười có tính được không.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chia(tu, mau):
    """Phep chia an toan: mau bang 0 thi tra ve 0.0 va bao lai la co."""
    if mau == 0:
        return 0.0, True
    return tu / mau, False


def danh_gia(y_that, y_doan):
    """Tra ve accuracy, precision, recall, F1 cua lop 1, chi dung numpy.

    Khi mau so bang 0 (vi du mo hinh khong bao mau nao la lop 1),
    gia tri do duoc dat bang 0, giong zero_division=0 cua scikit-learn,
    va ten thuoc do duoc ghi vao danh sach canh_bao.
    """
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)
    # Bốn ô của ma trận nhầm lẫn
    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))

    canh_bao = []
    acc = (tp + tn) / len(y_that)
    p, loi = chia(tp, tp + fp)
    if loi:
        canh_bao.append("precision (TP + FP = 0)")
    r, loi = chia(tp, tp + fn)
    if loi:
        canh_bao.append("recall (TP + FN = 0)")
    f1, loi = chia(2 * p * r, p + r)
    if loi:
        canh_bao.append("F1 (precision + recall = 0)")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "o": (int(tp), int(fp), int(fn), int(tn)), "canh_bao": canh_bao}


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

luoi = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)

for ten, mh in [("Mo hinh luoi", luoi), ("Hoi quy logistic", lr)]:
    du = mh.predict(X_te)
    kq = danh_gia(y_te, du)
    tp, fp, fn, tn = kq["o"]
    print(f"{ten}: TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    # Hàm của thư viện, tắt cảnh báo để so riêng con số
    with warnings.catch_warnings(record=True) as cb_tv:
        warnings.simplefilter("always")
        tv = {
            "accuracy": accuracy_score(y_te, du),
            "precision": precision_score(y_te, du),
            "recall": recall_score(y_te, du),
            "f1": f1_score(y_te, du),
        }
    print("  thuoc do     ham tu viet   scikit-learn")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k:10s}   {kq[k]:.4f}        {tv[k]:.4f}")
    if kq["canh_bao"]:
        print("  Ham tu viet: mau so bang 0 o", ", ".join(kq["canh_bao"]),
              "-> dat bang 0")
    for w in cb_tv:
        print("  scikit-learn canh bao:", w.category.__name__)
    print()

# Thư viện cũng cho chọn cách xử lý chia cho 0
du_luoi = luoi.predict(X_te)
print("precision_score cua mo hinh luoi voi zero_division khac nhau:")
for zd in [0, 1, np.nan]:
    print(f"  zero_division = {zd}: {precision_score(y_te, du_luoi, zero_division=zd)}")
