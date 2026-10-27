# -*- coding: utf-8 -*-
"""Bài tập 5: chọn k cho KNeighborsRegressor bằng kiểm định chéo.

cross_val_score với scoring="neg_mean_absolute_error" trả về MAE mang dấu âm
(scikit-learn luôn coi điểm càng lớn càng tốt), nên đổi dấu trước khi đọc.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với e5_knn_hoi_quy.py
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)
print(f"Tap hoc {len(X_tr)} can, tap kiem tra {len(X_te)} can")
print()

print("Kiem dinh cheo 5 phan tren tap hoc")
print("   k   MAE trung binh")
mae_cv = {}
for k in range(1, 21):
    diem = cross_val_score(KNeighborsRegressor(n_neighbors=k), X_tr, y_tr,
                           cv=5, scoring="neg_mean_absolute_error")
    mae_cv[k] = -diem.mean()
k_tot = min(mae_cv, key=mae_cv.get)
for k, m in mae_cv.items():
    dau = "  <- tot nhat" if k == k_tot else ""
    print(f"  {k:2d}   {m:.4f}{dau}")
print()

# Đường thẳng cũng được chấm bằng đúng cách đó để so trên cùng thước đo
mae_cv_duong = -cross_val_score(LinearRegression(), X_tr, y_tr, cv=5,
                                scoring="neg_mean_absolute_error").mean()
print(f"Duong thang, kiem dinh cheo: {mae_cv_duong:.4f}")
print()

knn = KNeighborsRegressor(n_neighbors=k_tot).fit(X_tr, y_tr)
knn5 = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)
print("MAE tren tap kiem tra (ty dong)")
print(f"  KNN, k = {k_tot:2d} (tot nhat): {mean_absolute_error(y_te, knn.predict(X_te)):.4f}")
print(f"  KNN, k =  5          : {mean_absolute_error(y_te, knn5.predict(X_te)):.4f}")
print(f"  Duong thang          : {mean_absolute_error(y_te, duong.predict(X_te)):.4f}")
print()

# Kiểm định chéo xáo trộn trước khi chia, để thấy k tốt nhất có vững không
print("Kiem dinh cheo co xao tron, doi random_state cua KFold")
for rs in range(5):
    cv = KFold(n_splits=5, shuffle=True, random_state=rs)
    m = {k: -cross_val_score(KNeighborsRegressor(n_neighbors=k), X_tr, y_tr,
                             cv=cv, scoring="neg_mean_absolute_error").mean()
         for k in range(1, 21)}
    kt = min(m, key=m.get)
    print(f"  random_state = {rs}: k tot nhat = {kt:2d}, MAE {m[kt]:.4f}")
