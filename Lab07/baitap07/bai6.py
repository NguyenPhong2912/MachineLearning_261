# -*- coding: utf-8 -*-
"""Bài tập 6: MAE, RMSE và R2 khi có một lần đoán hỏng.

Dùng lại dữ liệu giá nhà và đường thẳng của bài 1 (cùng cách chia).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với bài 1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)
duong = LinearRegression().fit(X_tr, y_tr)
du = duong.predict(X_te)


def ba_so(y_that, y_doan):
    mae = mean_absolute_error(y_that, y_doan)
    rmse = np.sqrt(mean_squared_error(y_that, y_doan))
    r2 = r2_score(y_that, y_doan)
    return mae, rmse, r2


# Cộng thêm 5 tỷ vào dự đoán của đúng một căn (căn đầu tiên của tập kiểm tra)
du_hong = du.copy()
du_hong[0] += 5.0
print(f"Tap kiem tra {len(y_te)} can. Can bi doan hong: "
      f"{X_te.iloc[0, 0]} m2, gia that {y_te.iloc[0]:.2f} ty,"
      f" doan {du[0]:.2f} -> {du_hong[0]:.2f} ty")
print()

truoc = ba_so(y_te, du)
sau = ba_so(y_te, du_hong)
print("thuoc do   truoc     sau     tang      gap")
for ten, a, b in zip(["MAE", "RMSE", "R2"], truoc, sau):
    print(f"  {ten:5s}  {a:7.4f}  {b:7.4f}  {b - a:+7.4f}  {b / a:6.2f} lan")
print()

# Vì sao RMSE nhạy hơn: xem sai số của từng căn trước và sau
sai_truoc = np.abs(y_te.to_numpy() - du)
sai_sau = np.abs(y_te.to_numpy() - du_hong)
n = len(y_te)
print(f"Sai so tuyet doi cua can do: {sai_truoc[0]:.4f} -> {sai_sau[0]:.4f}")
print(f"Phan dong gop cua can do vao tong |sai so|: "
      f"{sai_truoc[0] / sai_truoc.sum():.1%} -> {sai_sau[0] / sai_sau.sum():.1%}")
print(f"Phan dong gop cua can do vao tong binh phuong sai so: "
      f"{sai_truoc[0] ** 2 / (sai_truoc ** 2).sum():.1%}"
      f" -> {sai_sau[0] ** 2 / (sai_sau ** 2).sum():.1%}")
print(f"MAE tang dung 5 / {n} = {5 / n:.4f} (neu sai so cung dau)")
