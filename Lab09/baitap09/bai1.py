# -*- coding: utf-8 -*-
"""Bài tập 1: tự dự đoán bằng trọng số đã học.

Học lại mạng 8 nơ-ron ẩn trên bộ XOR như m2_xor.py, lấy coefs_ và intercepts_
rồi tự tính lan truyền tiến bằng numpy cho 5 điểm đầu của tập kiểm tra:
lớp ẩn dùng ReLU, lớp ra dùng sigmoid. So với predict_proba, cột thứ hai.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("data/xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
                    max_iter=5000, random_state=0).fit(X_tr, y_tr)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


W1, W2 = mlp.coefs_             # W1: 2 x 8, W2: 8 x 1
b1, b2 = mlp.intercepts_        # b1: 8 số, b2: 1 số
print("Kich thuoc:", "W1", W1.shape, "b1", b1.shape, "W2", W2.shape, "b2", b2.shape)
print("Ham kich hoat lop an:", mlp.activation, "| lop ra:", mlp.out_activation_)
print()

nam = X_te.iloc[:5].to_numpy()
H = relu(nam @ W1 + b1)         # lớp ẩn: 5 hàng x 8 nơ-ron
z_ra = H @ W2 + b2              # lớp ra: 5 hàng x 1 cột
p_tu_tinh = sigmoid(z_ra).ravel()
p_thu_vien = mlp.predict_proba(X_te.iloc[:5])[:, 1]

print("Lop an cua 5 diem (so no-ron dang bat = h > 0):")
for x, h in zip(nam, H):
    print(f"  ({x[0]:7.4f}, {x[1]:7.4f})  bat {int((h > 0).sum())}/8:",
          " ".join(f"{v:6.3f}" for v in h))
print()

print("   x1       x2      z ra      tu tinh         predict_proba    nhan that")
for x, z, a, b, t in zip(nam, z_ra.ravel(), p_tu_tinh, p_thu_vien, y_te.iloc[:5]):
    print(f"  {x[0]:7.4f}  {x[1]:7.4f}  {z:8.4f}  {a:.12f}  {b:.12f}     {t}")
print()
print("Chenh lech lon nhat:", f"{np.abs(p_tu_tinh - p_thu_vien).max():.2e}")
print("Khop toi 12 chu so thap phan:", np.allclose(p_tu_tinh, p_thu_vien, rtol=0, atol=1e-12))
# predict chỉ là so xác suất với ngưỡng 0.5, tức so z ra với 0
print("Du doan tu tinh (z > 0):", (z_ra.ravel() > 0).astype(int),
      "| mlp.predict:", mlp.predict(X_te.iloc[:5]))
