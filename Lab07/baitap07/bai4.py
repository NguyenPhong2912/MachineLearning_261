# -*- coding: utf-8 -*-
"""Bài tập 4: tự vẽ đường ROC của logistic và rừng ngẫu nhiên, lưu roc.png."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = {
    "Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Rung 200 cay": RandomForestClassifier(200, random_state=0),
}
duong = {}
for ten, mh in mo_hinh.items():
    p = mh.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    duong[ten] = (fpr, tpr, roc_auc_score(y_te, p))
    print(f"{ten:13s} AUC = {duong[ten][2]:.4f}, duong co {len(fpr)} diem")
print()

fig, ax = plt.subplots(figsize=(6, 5.5))
for (ten, (fpr, tpr, auc)), mau in zip(duong.items(), ["#2a78d6", "#1f7a3a"]):
    ax.plot(fpr, tpr, color=mau, lw=2, label=f"{ten}, AUC = {auc:.4f}")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Doan ngau nhien, AUC = 0.5")
ax.set_xlabel("Ty le bao dong nham (FPR)")
ax.set_ylabel("Recall (TPR)")
ax.set_title("Duong ROC tren tap kiem tra")
ax.legend(loc="lower right")
fig.savefig("roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)                          # chỉ lưu tệp, không mở cửa sổ
print("Da luu roc.png")
print()

# Tìm chỗ hai đường cắt nhau: so recall của hai đường trên cùng một lưới FPR.
# Đường ROC là hàm bậc thang, nên lấy recall lớn nhất đạt được tại mỗi FPR.
luoi = np.linspace(0, 1, 2001)
def recall_tai(fpr, tpr, x):
    return np.array([tpr[fpr <= v].max() for v in x])

r_lr = recall_tai(*duong["Logistic"][:2], luoi)
r_rf = recall_tai(*duong["Rung 200 cay"][:2], luoi)
hieu = r_lr - r_rf
print(f"Ti le luoi FPR ma logistic cao hon: {(hieu > 0).mean():.3f},"
      f" rung cao hon: {(hieu < 0).mean():.3f}, bang nhau: {(hieu == 0).mean():.3f}")
dau = np.sign(hieu)
doan = []
bat_dau = 0
for i in range(1, len(luoi) + 1):
    if i == len(luoi) or dau[i] != dau[bat_dau]:
        doan.append((luoi[bat_dau], luoi[i - 1], dau[bat_dau]))
        bat_dau = i
ten_dau = {1: "logistic cao hon", -1: "rung cao hon", 0: "bang nhau"}
print("Cac doan theo FPR:")
for a, b, d in doan:
    if b - a >= 0.005 or d != 0:
        print(f"  FPR {a:.3f} - {b:.3f}: {ten_dau[int(d)]}")
