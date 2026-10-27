# -*- coding: utf-8 -*-
"""Bài tập 6: rừng ngẫu nhiên cho bài toán hồi quy, trên dữ liệu giá nhà.

Cùng cách chia như f7_cay_hoi_quy.py; so MAE với cây sâu 3 và đường thẳng.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với bài 1 để so sánh công bằng
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

rung = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_tr, y_tr)
cay = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)

print("MAE tren tap kiem tra (ty dong)")
for ten, mh in [("Rung 200 cay", rung), ("Cay sau 3   ", cay), ("Duong thang ", duong)]:
    print(f"  {ten}: {mean_absolute_error(y_te, mh.predict(X_te)):.4f}")
print()

can_moi = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
print("Du doan cho bon can moi (ty dong)")
print("  dien tich    rung     cay   duong thang")
for dt, a, b, c in zip(can_moi["dien_tich"], rung.predict(can_moi),
                       cay.predict(can_moi), duong.predict(can_moi)):
    print(f"    {dt:6.0f} m2  {a:6.2f}  {b:6.2f}    {c:6.2f}")
print()

# Mỗi cây trả về giá trung bình của lá mà căn 250 m2 rơi vào
can_250 = pd.DataFrame({"dien_tich": [250.0]})
tung_cay = np.array([c.predict(can_250.to_numpy())[0] for c in rung.estimators_])
rong_nhat = X_tr["dien_tich"].nlargest(5)
print("Can 250 m2")
print(f"  du doan cua rung (trung binh 200 cay): {rung.predict(can_250)[0]:.2f}")
print(f"  du doan cua tung cay: nho nhat {tung_cay.min():.2f}, lon nhat {tung_cay.max():.2f}")
print(f"  gia cao nhat trong tap hoc: {y_tr.max():.2f}")
print("  5 can rong nhat tap hoc:", ", ".join(
    f"{d:.1f} m2 = {y_tr[i]:.2f}" for i, d in rong_nhat.items()))
print(f"  rung doan can 150 m2 va 250 m2 giong nhau: "
      f"{np.isclose(rung.predict(can_moi)[2], rung.predict(can_moi)[3])}")
