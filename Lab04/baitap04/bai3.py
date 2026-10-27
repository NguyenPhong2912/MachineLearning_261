# -*- coding: utf-8 -*-
"""Bài tập 3: đổi đơn vị cột chi_tieu từ nghìn đồng sang triệu đồng.

Giữ nguyên cách chia của e2_chuan_hoa.py, chỉ chia cột chi_tieu cho 1000.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_mo_rong.csv")
y = df["qua_mon"]
ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]

X_tr, X_te, y_tr, y_te = train_test_split(
    df[ba_cot], y, test_size=0.25, random_state=17, stratify=y
)

# Bản sao đổi chi_tieu sang triệu đồng
X_tr_tr, X_te_tr = X_tr.copy(), X_te.copy()
X_tr_tr["chi_tieu"] = X_tr_tr["chi_tieu"] / 1000
X_te_tr["chi_tieu"] = X_te_tr["chi_tieu"] / 1000


def cham(mo_hinh, X_hoc, X_kt):
    mo_hinh.fit(X_hoc, y_tr)
    return accuracy_score(y_te, mo_hinh.predict(X_kt)), mo_hinh.predict(X_kt)


def knn_tho():
    return KNeighborsClassifier(n_neighbors=7)


def knn_chuan():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))


print("Do lech chuan cua ba cot tren tap hoc")
print("  don vi          gio_on  diem_giua_ky  chi_tieu")
print(f"  nghin dong    {X_tr.std(ddof=0).iloc[0]:8.2f}  {X_tr.std(ddof=0).iloc[1]:12.2f}"
      f"  {X_tr.std(ddof=0).iloc[2]:8.2f}")
print(f"  trieu dong    {X_tr_tr.std(ddof=0).iloc[0]:8.2f}  {X_tr_tr.std(ddof=0).iloc[1]:12.2f}"
      f"  {X_tr_tr.std(ddof=0).iloc[2]:8.2f}")
print()

a_ng, du_a_ng = cham(knn_tho(), X_tr, X_te)
a_tr, du_a_tr = cham(knn_tho(), X_tr_tr, X_te_tr)
b_ng, du_b_ng = cham(knn_chuan(), X_tr, X_te)
b_tr, du_b_tr = cham(knn_chuan(), X_tr_tr, X_te_tr)
c, _ = cham(knn_chuan(), X_tr[["gio_on", "diem_giua_ky"]],
            X_te[["gio_on", "diem_giua_ky"]])

print("Accuracy tren tap kiem tra, KNN k = 7")
print(f"  Doan bua ca lop qua                 : {y_te.mean():.4f}")
print(f"  Khong chuan hoa, chi_tieu nghin dong: {a_ng:.4f}")
print(f"  Khong chuan hoa, chi_tieu trieu dong: {a_tr:.4f}")
print(f"  Co chuan hoa,    chi_tieu nghin dong: {b_ng:.4f}")
print(f"  Co chuan hoa,    chi_tieu trieu dong: {b_tr:.4f}")
print(f"  Bo cot chi_tieu, co chuan hoa       : {c:.4f}")
print()
print("So mau kiem tra bi doan khac khi doi don vi")
print(f"  khong chuan hoa: {(du_a_ng != du_a_tr).sum()} / {len(y_te)}")
print(f"  co chuan hoa   : {(du_b_ng != du_b_tr).sum()} / {len(y_te)}")
print()

# Một cặp sinh viên cụ thể để thấy cột nào chi phối khoảng cách
A = pd.Series({"gio_on": 5.0, "diem_giua_ky": 3.0, "chi_tieu": 4000.0})
C = pd.Series({"gio_on": 6.0, "diem_giua_ky": 3.5, "chi_tieu": 7000.0})
chenh = (A - C) ** 2
print("Hai ban A, C cua muc 2: binh phuong chenh lech tung cot")
print(f"  nghin dong: gio {chenh['gio_on']:.2f}, diem {chenh['diem_giua_ky']:.2f}, "
      f"chi_tieu {chenh['chi_tieu']:.0f}")
print(f"  trieu dong: gio {chenh['gio_on']:.2f}, diem {chenh['diem_giua_ky']:.2f}, "
      f"chi_tieu {chenh['chi_tieu'] / 1000 ** 2:.2f}")
