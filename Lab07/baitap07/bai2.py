# -*- coding: utf-8 -*-
"""Bài tập 2: classification_report của SVM nhận dạng chữ số viết tay.

Lấy lại SVM kernel RBF của bài 6 trên bộ chữ số của scikit-learn,
cùng cách chia như các bài trước.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

svm = SVC().fit(X_tr, y_tr)
du = svm.predict(X_te)
print(f"Accuracy tren tap kiem tra: {accuracy_score(y_te, du):.4f}"
      f" ({(du != y_te).sum()} anh sai tren {len(y_te)})")
print()
print(classification_report(y_te, du, digits=4))

p, r, f, ho_tro = precision_recall_fscore_support(y_te, du)
bang = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": ho_tro})
cs_r = int(np.argmin(r))
cs_p = int(np.argmin(p))
print(f"Recall thap nhat   : chu so {cs_r} ({r[cs_r]:.4f})")
print(f"Precision thap nhat: chu so {cs_p} ({p[cs_p]:.4f})")
print()

# Ảnh nào bị đoán sai, để biết các chữ số đó bị nhầm với chữ số nào
sai = np.flatnonzero(du != y_te)
print("Cac anh doan sai:")
for i in sai:
    print(f"  that la {y_te[i]}, doan la {du[i]}")
print()

print("So mau moi chu so trong tap kiem tra:", ho_tro.tolist())
print(f"Support nho nhat {ho_tro.min()}, lon nhat {ho_tro.max()}")
print(f"F1 macro avg    = {f.mean():.4f}")
print(f"F1 weighted avg = {np.average(f, weights=ho_tro):.4f}")
