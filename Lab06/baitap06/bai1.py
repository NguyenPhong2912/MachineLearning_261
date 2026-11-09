# -*- coding: utf-8 -*-
"""Bài tập 1: tính tay f(x) và khoảng cách tới đường ranh giới.

Dùng w1, w2, b mà g1_le_rong_nhat.py in ra (làm tròn 3 chữ số), tự nhân và
cộng, không gọi decision_function. Cuối bài so lại với số máy tính để kiểm tra.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Ba hệ số in ra ở mục 2 của đề
w1, w2, b = -0.831, -0.355, 4.820
do_dai_w = np.sqrt(w1 ** 2 + w2 ** 2)
do_rong_le = 2 / do_dai_w
print(f"|w| = sqrt({w1}^2 + {w2}^2) = {do_dai_w:.4f}")
print(f"Do rong le = 2 / |w| = {do_rong_le:.4f}, nua do rong le = {do_rong_le / 2:.4f}")
print()


def f_tay(x1, x2):
    """f(x) = w1*x1 + w2*x2 + b, tinh bang phep nhan va cong."""
    return w1 * x1 + w2 * x2 + b


# Hai điểm mới và ba vectơ hỗ trợ (toạ độ lấy từ kết quả của g1)
cac_diem = [
    ("diem moi (3, 3)    ", 3.0, 3.0),
    ("diem moi (5, 4.5)  ", 5.0, 4.5),
]

# Toạ độ đầy đủ của ba vectơ hỗ trợ: sinh lại đúng 24 điểm của mục 2
X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y
svm = SVC(kernel="linear", C=1000).fit(X, y)
for i in svm.support_:
    cac_diem.append((f"vecto HT diem {i:2d}  ", X[i, 0], X[i, 1]))

# Cột cuối dùng w, b chưa làm tròn của máy, để thấy sai số do làm tròn
w_may = np.linalg.norm(svm.coef_[0])
print(f"Nua do rong le theo he so chua lam tron: {1 / w_may:.4f}")
print()
print("                       x1     x2    f(x) tinh tay  khoang cach   f(x) may  khoang cach may")
for ten, x1, x2 in cac_diem:
    f = f_tay(x1, x2)
    kc = abs(f) / do_dai_w
    f_may = svm.decision_function([[x1, x2]])[0]
    print(f"  {ten} {x1:5.2f}  {x2:5.2f}    {f:+7.3f}       {kc:7.4f}     {f_may:+7.3f}     {abs(f_may) / w_may:7.4f}")
print()

# Phép tính chi tiết cho hai điểm mới
for x1, x2 in [(3.0, 3.0), (5.0, 4.5)]:
    f = f_tay(x1, x2)
    print(f"f({x1}, {x2}) = ({w1})*{x1} + ({w2})*{x2} + {b} = {f:+.3f};"
          f"  khoang cach = {abs(f):.3f} / {do_dai_w:.4f} = {abs(f) / do_dai_w:.4f}")
