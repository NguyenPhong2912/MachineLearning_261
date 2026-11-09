# -*- coding: utf-8 -*-
"""Bài tập 5: cân bằng lớp cho rừng ngẫu nhiên và SVM."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def tao(loai, can_bang):
    """Tao mo hinh, co hoac khong co class_weight='balanced'."""
    cw = "balanced" if can_bang else None
    if loai == "Logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw))
    if loai == "Rung 200 cay":
        return RandomForestClassifier(200, class_weight=cw, random_state=0)
    return make_pipeline(StandardScaler(), SVC(class_weight=cw, random_state=0))


print("Tap kiem tra (nguong mac dinh cua predict)")
print("Mo hinh        can bang  so bao  precision  recall   F1")
thay_doi = {}
for loai in ["Logistic", "Rung 200 cay", "SVM RBF"]:
    kq = []
    for cb in [False, True]:
        du = tao(loai, cb).fit(X_tr, y_tr).predict(X_te)
        p = precision_score(y_te, du, zero_division=0)
        r = recall_score(y_te, du)
        f = f1_score(y_te, du)
        kq.append((p, r, f))
        print(f"  {loai:13s} {'co ' if cb else 'khong':5s}    {du.sum():4d}"
              f"    {p:.4f}    {r:.4f}  {f:.4f}")
    thay_doi[loai] = [kq[1][i] - kq[0][i] for i in range(3)]
print()

print("Thay doi khi them class_weight='balanced'")
print("Mo hinh        precision  recall    F1")
for loai, (dp, dr, df1) in thay_doi.items():
    print(f"  {loai:13s} {dp:+.4f}   {dr:+.4f}  {df1:+.4f}")
