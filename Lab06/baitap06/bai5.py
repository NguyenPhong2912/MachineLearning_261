# -*- coding: utf-8 -*-
"""Bài tập 5: bảng nhầm lẫn 10 x 10 của SVM chữ số viết tay (mục 6).

In bảng nhầm lẫn (hàng là chữ số thật, cột là chữ số đoán), liệt kê các cặp
bị nhầm, rồi vẽ và lưu từng ảnh đoán sai vào thư mục anh_doan_sai/ ở thư mục
đang đứng chạy lệnh.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Đúng mô hình của mục 6: SVC kernel RBF để mặc định, không chuẩn hoá
svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du_doan = svm.predict(X_te)

bang = confusion_matrix(y_te, du_doan)
print("Bang nham lan, hang la chu so that, cot la chu so doan")
print(pd.DataFrame(bang, index=[f"that {i}" for i in range(10)],
                   columns=[str(i) for i in range(10)]).to_string())
print()
print("Tong duong cheo (doan dung):", np.trace(bang), "tren", bang.sum())
print()

# Các ô ngoài đường chéo khác 0 là các cặp bị nhầm
print("Cac cap bi nham (that -> doan): so anh")
for that in range(10):
    for doan in range(10):
        if that != doan and bang[that, doan] > 0:
            print(f"  {that} -> {doan}: {bang[that, doan]}")
print()

# Vẽ và lưu từng ảnh đoán sai
os.makedirs("anh_doan_sai", exist_ok=True)
sai = np.flatnonzero(du_doan != y_te)
diem_quyet_dinh = svm.decision_function(X_te[sai])
for i, phieu in zip(sai, diem_quyet_dinh):
    fig, ax = plt.subplots(figsize=(2.5, 2.8))
    ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"anh {i}: that {y_te[i]}, doan {du_doan[i]}", fontsize=9)
    ax.axis("off")
    ten_tep = f"anh_doan_sai/anh_{i}_that{y_te[i]}_doan{du_doan[i]}.png"
    fig.savefig(ten_tep, dpi=120, bbox_inches="tight")
    plt.close(fig)                      # chỉ lưu tệp, không mở cửa sổ
    # decision_function kiểu "ovr": điểm cao nhất là lớp được đoán
    # (phần nguyên xấp xỉ số phiếu thắng trong 45 cặp một chọi một)
    thu_tu = np.argsort(phieu)[::-1]
    hang_that = int(np.flatnonzero(thu_tu == y_te[i])[0]) + 1
    print(f"Da luu {ten_tep}")
    print(f"    ba lop diem cao nhat: " + ", ".join(
        f"{c} ({phieu[c]:.2f})" for c in thu_tu[:3])
        + f"; lop that {y_te[i]} dung hang {hang_that} ({phieu[y_te[i]]:.2f})")
