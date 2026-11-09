# -*- coding: utf-8 -*-
"""Bài tập 5: dò lưới cho mạng nơ-ron trên bộ chữ số.

hidden_layer_sizes trong (10,), (50,), (100,), (50, 50); alpha trong 0.0001,
0.01, 1. Mô hình là pipeline nên tên tham số có tiền tố "mlpclassifier__".
"""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=1000, random_state=0),
)
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1],
}
do = GridSearchCV(mo_hinh, luoi, cv=5, n_jobs=-1).fit(X_tr, y_tr)

kq = pd.DataFrame(do.cv_results_["params"])
kq.columns = ["alpha", "lop_an"]
kq["kiem_dinh_cheo"] = do.cv_results_["mean_test_score"]
bang = kq.pivot(index="lop_an", columns="alpha", values="kiem_dinh_cheo")
bang = bang.loc[[(10,), (50,), (100,), (50, 50)]]
print("Diem kiem dinh cheo 5 phan (hang: lop an, cot: alpha)")
print(bang.round(4).to_string())
print()

tot = do.best_params_
print("Cap tot nhat: lop an", tot["mlpclassifier__hidden_layer_sizes"],
      ", alpha", tot["mlpclassifier__alpha"])
print(f"Diem kiem dinh cheo: {do.best_score_:.4f}")
acc_mlp = accuracy_score(y_te, do.predict(X_te))
print(f"Tap kiem tra       : {acc_mlp:.4f}")
print()

svm = SVC().fit(X_tr, y_tr)
acc_svm = accuracy_score(y_te, svm.predict(X_te))
print(f"SVM kernel RBF, tap kiem tra: {acc_svm:.4f}")
print("MLP sau do luoi vuot SVM:", acc_mlp > acc_svm,
      f"(chenh {acc_mlp - acc_svm:+.4f}, tuc {round((acc_mlp - acc_svm) * len(y_te)):+d} anh)")
