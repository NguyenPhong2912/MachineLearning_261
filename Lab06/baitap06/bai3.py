# -*- coding: utf-8 -*-
"""Bài tập 3: SVM kernel đa thức bậc 2, 3, 4 trên bộ hai vầng trăng.

Mỗi bậc chuẩn hoá bằng make_pipeline và dò C trong 0.1, 1, 10, 100 bằng
GridSearchCV (kiểm định chéo 5 phần trên tập học), rồi so với kernel RBF mục 5.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tên tham số trong pipeline phải ghi kèm tên bước: "svc__C"
luoi = {"svc__C": [0.1, 1, 10, 100]}
print("Bac   C tot nhat   kiem dinh cheo   tap kiem tra   so vecto HT")
for d in [2, 3, 4]:
    pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=d))
    tim = GridSearchCV(pipe, luoi, cv=5).fit(X_tr, y_tr)
    svm = tim.best_estimator_[-1]
    print(f"  {d}     {tim.best_params_['svc__C']:6}        {tim.best_score_:.4f}"
          f"          {tim.score(X_te, y_te):.4f}        {len(svm.support_)}")
    diem = tim.cv_results_["mean_test_score"]
    print("      diem theo C:", ", ".join(
        f"C={c}: {s:.4f}" for c, s in zip(luoi["svc__C"], diem)))
print()

# Bậc chẵn với coef0 = 0 cho K(x, z) = K(-x, z), nên f(x) = f(-x).
# Thử lại với coef0 = 1 để kernel có cả các số hạng bậc lẻ.
print("Them coef0 = 1 (mac dinh coef0 = 0)")
print("Bac   C tot nhat   kiem dinh cheo   tap kiem tra")
for d in [2, 3, 4]:
    pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=d, coef0=1))
    tim = GridSearchCV(pipe, luoi, cv=5).fit(X_tr, y_tr)
    print(f"  {d}     {tim.best_params_['svc__C']:6}        {tim.best_score_:.4f}"
          f"          {tim.score(X_te, y_te):.4f}")
print()

# Kiểm tra tính đối xứng: lật mỗi điểm qua tâm dữ liệu (sau chuẩn hoá là x -> -x)
bac2 = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=2)).fit(X_tr, y_tr)
lat = 2 * X_tr.mean() - X_tr
giong = abs(bac2.decision_function(X_tr) - bac2.decision_function(lat)).max()
print(f"Bac 2, coef0 = 0: |f(x) - f(-x)| lon nhat tren tap hoc = {giong:.2e}")
print()

# Kernel RBF của mục 5 (C = 10, gamma = 1, không chuẩn hoá) để so sánh
rbf = GridSearchCV(SVC(kernel="rbf"),
                   {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]},
                   cv=5).fit(X_tr, y_tr)
print(f"Kernel RBF muc 5: {rbf.best_params_}, kiem dinh cheo {rbf.best_score_:.4f},"
      f" tap kiem tra {rbf.score(X_te, y_te):.4f}")
