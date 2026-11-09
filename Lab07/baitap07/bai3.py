# -*- coding: utf-8 -*-
"""Bài tập 3: chọn ngưỡng theo yêu cầu recall >= 0.8.

Ngưỡng chỉ được chọn trên tập học bằng xác suất của cross_val_predict,
rồi áp cho tập kiểm tra đúng một lần.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEU_CAU = 0.8          # nhà trường muốn tìm ra ít nhất 80% số bạn bỏ học

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = make_pipeline(StandardScaler(), LogisticRegression())
xs_hoc = cross_val_predict(lr, X_tr, y_tr, cv=5,
                           method="predict_proba")[:, 1]

# Ngưỡng càng lớn thì báo càng ít, recall càng thấp:
# cần ngưỡng lớn nhất mà recall trên tập học vẫn từ 0.8 trở lên.
cac_nguong = np.round(np.arange(0.01, 1.00, 0.01), 2)
recall_hoc = np.array([recall_score(y_tr, (xs_hoc >= t).astype(int))
                       for t in cac_nguong])
dat = cac_nguong[recall_hoc >= YEU_CAU]
t_chon = dat.max()
print("Tap hoc (kiem dinh cheo 5 phan)")
print("nguong  so bao  recall  precision")
for t in cac_nguong:
    if (round(t * 100) % 5 == 0 and t <= 0.5) or t == t_chon:
        du = (xs_hoc >= t).astype(int)
        dau = "  <- chon" if t == t_chon else ""
        print(f"  {t:.2f}   {du.sum():4d}   {recall_score(y_tr, du):.4f}  "
              f"{precision_score(y_tr, du):.4f}{dau}")
print()
du_hoc = (xs_hoc >= t_chon).astype(int)
print(f"Nguong lon nhat cho recall >= {YEU_CAU} tren tap hoc: {t_chon}")
print(f"  tren tap hoc: recall {recall_score(y_tr, du_hoc):.4f},"
      f" precision {precision_score(y_tr, du_hoc):.4f},"
      f" bao {du_hoc.sum()} / {len(y_tr)} ban")
print()

# Chọn xong mới mở tập kiểm tra, và chỉ một lần
lr.fit(X_tr, y_tr)
xs = lr.predict_proba(X_te)[:, 1]
du = (xs >= t_chon).astype(int)
tp = int(((du == 1) & (y_te == 1)).sum())
fp = int(((du == 1) & (y_te == 0)).sum())
r = recall_score(y_te, du)
print(f"Tap kiem tra voi nguong {t_chon}:")
print(f"  so ban bi bao {du.sum()} / {len(y_te)} (TP = {tp}, FP = {fp})")
print(f"  recall {r:.4f} -> {'dat' if r >= YEU_CAU else 'KHONG dat'} yeu cau {YEU_CAU}")
print(f"  precision {precision_score(y_te, du):.4f}")
