# -*- coding: utf-8 -*-
"""Bài tập 6: mạng nơ-ron cho hồi quy, trên dữ liệu giá nhà của bài 1.

MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0) có
chuẩn hoá; so MAE với hồi quy tuyến tính và dự đoán căn 250 m2.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với bài 1 để so sánh công bằng
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

mang = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0),
).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)

print(f"Mang hoc {mang[-1].n_iter_} vong")
print("MAE tren tap kiem tra (ty dong)")
print(f"  MLP (20,)   : {mean_absolute_error(y_te, mang.predict(X_te)):.4f}")
print(f"  Duong thang : {mean_absolute_error(y_te, duong.predict(X_te)):.4f}")
print()

can_moi = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 200.0, 250.0, 400.0]})
print(f"Tap hoc: dien tich {X_tr['dien_tich'].min():.1f} toi "
      f"{X_tr['dien_tich'].max():.1f} m2, gia cao nhat {y_tr.max():.2f}")
print("Du doan cho can moi (ty dong)")
print("  dien tich    MLP   duong thang   chenh")
for dt, a, b in zip(can_moi["dien_tich"], mang.predict(can_moi),
                    duong.predict(can_moi)):
    print(f"    {dt:5.0f} m2  {a:6.2f}    {b:6.2f}     {a - b:+.2f}")
print()

# Ngoài vùng dữ liệu, mạng ReLU chỉ còn là một đường thẳng: độ dốc cố định
dt = pd.DataFrame({"dien_tich": [200.0, 300.0, 400.0]})
p = mang.predict(dt)
print(f"Do doc cua MLP tu 200 toi 300 m2: {(p[1] - p[0]) / 100:.4f} ty/m2, "
      f"tu 300 toi 400 m2: {(p[2] - p[1]) / 100:.4f} ty/m2")
print(f"Do doc cua duong thang         : {duong.coef_[0]:.4f} ty/m2")
