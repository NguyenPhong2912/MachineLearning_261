# -*- coding: utf-8 -*-
"""Bài tập 4: dò lưới mịn quanh cặp tốt nhất của mục 5 (C = 10, gamma = 1).

C trong 3, 10, 30 và gamma trong 0.3, 1, 3; so điểm kiểm định chéo với 0.8711
và accuracy tập kiểm tra với 0.9200 của lưới thưa.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Lưới thưa của mục 5, để so sánh
thua = GridSearchCV(SVC(kernel="rbf"),
                    {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]},
                    cv=5).fit(X_tr, y_tr)

# Lưới mịn quanh C = 10, gamma = 1
luoi = {"C": [3, 10, 30], "gamma": [0.3, 1, 3]}
min_ = GridSearchCV(SVC(kernel="rbf"), luoi, cv=5).fit(X_tr, y_tr)

bang = pd.DataFrame(min_.cv_results_["params"])
bang["diem"] = min_.cv_results_["mean_test_score"]
bang["lech chuan"] = min_.cv_results_["std_test_score"]
print("Luoi min, diem kiem dinh cheo, hang la C, cot la gamma")
print(bang.pivot(index="C", columns="gamma", values="diem").round(4).to_string())
print()
print("Do lech chuan giua 5 phan, hang la C, cot la gamma")
print(bang.pivot(index="C", columns="gamma", values="lech chuan").round(4).to_string())
print()

print("              cap tot nhat              kiem dinh cheo   tap kiem tra")
for ten, tim in [("Luoi thua", thua), ("Luoi min ", min_)]:
    print(f"  {ten}   {str(tim.best_params_):24s}   {tim.best_score_:.4f}"
          f"           {tim.score(X_te, y_te):.4f}")
print()

# Hai mô hình có đoán khác nhau trên tập kiểm tra không
khac = (thua.predict(X_te) != min_.predict(X_te)).sum()
print("So diem kiem tra hai mo hinh doan khac nhau:", khac, "tren", len(X_te))
print("So lan hoc: luoi thua", 16 * 5, "lan, luoi min", 9 * 5, "lan")
