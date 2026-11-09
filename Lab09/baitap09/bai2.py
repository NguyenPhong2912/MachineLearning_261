# -*- coding: utf-8 -*-
"""Bài tập 2: một mạng XOR khác.

h1 = max(0, x1 - x2), h2 = max(0, x2 - x1), y = h1 + h2.
Hai nơ-ron ẩn đo "x1 lớn hơn x2 bao nhiêu" và "x2 lớn hơn x1 bao nhiêu".
Mỗi lúc nhiều nhất một nơ-ron bật, nên h1 + h2 = |x1 - x2|. Với x là 0 hoặc 1,
|x1 - x2| bằng 1 khi hai bit khác nhau và bằng 0 khi giống nhau, chính là XOR.
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


W1 = np.array([[1.0, -1.0],     # cột 1: h1 = x1 - x2
               [-1.0, 1.0]])    # cột 2: h2 = x2 - x1
b1 = np.array([0.0, 0.0])
W2 = np.array([1.0, 1.0])
b2 = 0.0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
xor = np.logical_xor(X[:, 0], X[:, 1]).astype(float)

print("Mang XOR thu hai:")
print("  x1 x2   z1 z2   h1 h2    y   XOR")
for x, dung in zip(X, xor):
    z = x @ W1 + b1
    h = relu(z)
    y = h @ W2 + b2
    print(f"  {x[0]:.0f}  {x[1]:.0f}   {z[0]:2.0f} {z[1]:2.0f}    {h[0]:.0f}  {h[1]:.0f}"
          f"    {y:.0f}    {dung:.0f}")
print()

Y = relu(X @ W1 + b1) @ W2 + b2
print("Tinh ca bang mot lan:", Y)
print("Dung voi XOR o ca bon dau vao:", np.array_equal(Y, xor))
print()

# Kiểm tra thêm: với mọi x1, x2 thực, mạng tính đúng |x1 - x2|
rng = np.random.default_rng(0)
Xr = rng.uniform(-2, 2, (1000, 2))
Yr = relu(Xr @ W1 + b1) @ W2 + b2
print("Voi 1000 diem thuc ngau nhien, y == |x1 - x2|:",
      np.allclose(Yr, np.abs(Xr[:, 0] - Xr[:, 1])))
# Nếu bỏ ReLU, lớp ẩn chỉ còn là phép nhân ma trận và hai nhánh triệt tiêu nhau
Y_khong_relu = (X @ W1 + b1) @ W2 + b2
print("Bo ReLU di, y =", Y_khong_relu, "-> khong con la XOR")
