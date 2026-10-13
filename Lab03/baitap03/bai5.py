# -*- coding: utf-8 -*-
"""Bài tập 5: chọn sai biến thể Naive Bayes.

Đưa số giờ ôn và điểm giữa kỳ, vốn là số liệu liên tục, vào MultinomialNB
vốn dành cho số đếm, rồi so với GaussianNB của mục 6.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Cung du lieu, cung cach chia nhu d5_gaussian.py")
for ten, mo_hinh in [("GaussianNB   ", GaussianNB()),
                     ("MultinomialNB", MultinomialNB())]:
    mo_hinh.fit(X_train, y_train)          # không có lỗi nào được báo
    acc = accuracy_score(y_test, mo_hinh.predict(X_test))
    print(f"  {ten}: accuracy = {acc:.4f}")
print()

# MultinomialNB coi mỗi bạn là một "túi" gồm gio_on lần chữ "giờ" và
# diem_giua_ky lần chữ "điểm". Nó chỉ học tỷ trọng của hai "chữ" đó trong mỗi lớp.
mnb = MultinomialNB().fit(X_train, y_train)
ty_trong = np.exp(mnb.feature_log_prob_)
print("Nhung gi MultinomialNB hoc duoc: ty trong cua moi cot trong 'tui' cua lop")
print(f"  {'':12s} {'gio_on':>8s} {'diem_giua_ky':>13s}")
for i, ten_lop in enumerate(["Lop 0 (rot)", "Lop 1 (qua)"]):
    print(f"  {ten_lop:12s} {ty_trong[i, 0]:8.4f} {ty_trong[i, 1]:13.4f}")
print()

# Nó không biết thang đo: đổi điểm sang thang 100 thì mô hình đổi theo
print("Doi diem giua ky tu thang 10 sang thang 100 (nhan 10), du lieu khong doi y nghia:")
X_train_100 = X_train.assign(diem_giua_ky=X_train["diem_giua_ky"] * 10)
X_test_100 = X_test.assign(diem_giua_ky=X_test["diem_giua_ky"] * 10)
for ten, mo_hinh in [("GaussianNB   ", GaussianNB()),
                     ("MultinomialNB", MultinomialNB())]:
    mo_hinh.fit(X_train_100, y_train)
    acc = accuracy_score(y_test, mo_hinh.predict(X_test_100))
    print(f"  {ten}: accuracy = {acc:.4f}")
print()

# Và nó chỉ chạy được vì dữ liệu tình cờ không âm
print("Tru di trung binh (chuan hoa) de du lieu co so am:")
X_train_lech = X_train - X_train.mean()
for ten, mo_hinh in [("GaussianNB   ", GaussianNB()),
                     ("MultinomialNB", MultinomialNB())]:
    try:
        mo_hinh.fit(X_train_lech, y_train)
        acc = accuracy_score(y_test, mo_hinh.predict(X_test - X_train.mean()))
        print(f"  {ten}: accuracy = {acc:.4f}")
    except ValueError as loi:
        print(f"  {ten}: ValueError: {loi}")
