# -*- coding: utf-8 -*-
"""Bài tập 3: đổi tham số làm trơn alpha của bộ lọc thư rác."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Cùng cách chia với d3_loc_thu_rac.py để so được với alpha = 1.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cot_tay = vec.vocabulary_["tay"]      # từ tố cáo rác mạnh, dùng để soi xác suất
la_rac_that = (y_test == 1).to_numpy()


def thu_alpha(alpha):
    mo_hinh = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    p_tay_rac, p_tay_thuong = np.exp(mo_hinh.feature_log_prob_[[1, 0], cot_tay])
    # Xác suất rác trung bình mô hình gán cho tin rác thật và cho tin thường thật
    p = mo_hinh.predict_proba(X_test)[:, 1]
    print(f"{alpha:8.3f}  {acc:.4f}   {pre:.4f}   {rec:.4f}  "
          f"{tn:3d}{fp:3d}{fn:3d}{tp:3d}   {p_tay_rac / p_tay_thuong:6.2f}"
          f"       {p[la_rac_that].mean():.3f}       {p[~la_rac_that].mean():.3f}")


TIEU_DE = ("   alpha  Accuracy Precision Recall   TN FP FN TP  P(tay|rac)/   "
           "P(rac) tb   P(rac) tb\n"
           "                                                  P(tay|thuong)  "
           "tin rac     tin thuong")

print("Nam gia tri alpha de bai yeu cau")
print(TIEU_DE)
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    thu_alpha(alpha)
print()

# Năm giá trị trên cho cùng một nhãn, nên đẩy alpha xa hơn để thấy chỗ gãy
print("Day alpha xa hon de thay cho mo hinh gay")
print(TIEU_DE)
for alpha in [200.0, 300.0, 500.0, 700.0, 1000.0]:
    thu_alpha(alpha)
print()

# Với alpha rất lớn, mọi P(từ | lớp) tiến về 1/V, bằng nhau ở hai lớp
V = X.shape[1]
print(f"So tu trong tu dien V = {V}, nen 1/V = {1 / V:.4f}")
print("Khi alpha rat lon, P(tu | lop) = (dem + alpha) / (tong + alpha*V) -> 1/V")
print("o ca hai lop, ty so gan 1, chi con tien nghiem quyet dinh.")
print(f"Tien nghiem tap hoc: P(thuong) = {1 - y_train.mean():.4f}, "
      f"P(rac) = {y_train.mean():.4f}")
