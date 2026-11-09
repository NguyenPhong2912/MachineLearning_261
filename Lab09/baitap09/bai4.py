# -*- coding: utf-8 -*-
"""Bài tập 4: tốc độ học trên bộ chữ số viết tay.

Mạng một lớp ẩn 100 nơ-ron, learning_rate_init = 0.0001, 0.001, 0.01, 0.1.
Ghi n_iter_, mất mát cuối và accuracy tập kiểm tra; vẽ bốn loss_curve_ lên
cùng một hình, lưu thành toc_do_hoc.png ở thư mục đang đứng chạy lệnh.
"""

import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Tốc độ học nhỏ có thể chạy hết max_iter, cảnh báo đó là điều ta muốn quan sát
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

duong = {}
print("Toc do hoc   so vong   mat mat vong 1   mat mat cuoi   tap kiem tra")
for lr in [0.0001, 0.001, 0.01, 0.1]:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=lr,
                      max_iter=1000, random_state=0),
    ).fit(X_tr, y_tr)
    mlp = mang[-1]
    duong[lr] = mlp.loss_curve_
    acc = accuracy_score(y_te, mang.predict(X_te))
    print(f"  {lr:<9}   {mlp.n_iter_:6d}   {mlp.loss_curve_[0]:14.4f}"
          f"   {mlp.loss_curve_[-1]:12.4f}   {acc:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
for lr, mm in duong.items():
    ax.plot(range(1, len(mm) + 1), mm, label=f"learning_rate_init = {lr}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Vong hoc (thang loga)")
ax.set_ylabel("Mat mat tren tap hoc (thang loga)")
ax.set_title("Duong mat mat voi bon toc do hoc")
ax.legend()
fig.savefig("toc_do_hoc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print()
print("Da luu toc_do_hoc.png")
