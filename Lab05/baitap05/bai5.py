# -*- coding: utf-8 -*-
"""Bài tập 5: độ quan trọng bằng cách xáo trộn (permutation importance).

Xáo trộn một cột trên tập kiểm tra rồi đo accuracy giảm bao nhiêu; cột vô dụng
thì xáo trộn không làm mô hình kém đi.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Lấy lại rừng có cột so_ngau_nhien của f6_do_quan_trong.py
rng = np.random.default_rng(0)
X_tr_nn = X_tr.copy()
X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
rung_nn = RandomForestClassifier(n_estimators=200, random_state=0)
rung_nn.fit(X_tr_nn, y_tr)

# Thêm một cột ngẫu nhiên tương tự vào tập kiểm tra (rút tiếp từ cùng bộ sinh số)
X_te_nn = X_te.copy()
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))
print(f"Accuracy tap kiem tra cua rung co cot ngau nhien: "
      f"{accuracy_score(y_te, rung_nn.predict(X_te_nn)):.4f}")
print()

qt_cu = pd.Series(rung_nn.feature_importances_, index=X_tr_nn.columns)
kq = permutation_importance(rung_nn, X_te_nn, y_te, n_repeats=20, random_state=0)
qt_moi = pd.Series(kq.importances_mean, index=X_te_nn.columns)
lech = pd.Series(kq.importances_std, index=X_te_nn.columns)

hang_cu = qt_cu.rank(ascending=False).astype(int)
hang_moi = qt_moi.rank(ascending=False, method="min").astype(int)

print("Do quan trong theo hai cach do")
print("  cot                  feature_importances_ (hang)   xao tron: giam accuracy (hang)")
for c in qt_moi.sort_values(ascending=False).index:
    print(f"  {c:20s} {qt_cu[c]:.3f} ({hang_cu[c]})                    "
          f"{qt_moi[c]:+.4f} +- {lech[c]:.4f} ({hang_moi[c]})")
print()

# Mỗi lần xáo trộn, accuracy thay đổi bao nhiêu khách (mỗi khách là 1/n_kt)
n_kt = len(X_te_nn)
tung_lan = pd.Series(kq.importances[list(X_te_nn.columns).index("so_ngau_nhien")])
print("Cot so_ngau_nhien, 20 lan xao tron:")
print(f"  giam accuracy nho nhat {tung_lan.min():+.4f}, lon nhat {tung_lan.max():+.4f}")
print(f"  tuong ung {tung_lan.min() * n_kt:+.0f} toi {tung_lan.max() * n_kt:+.0f} khach "
      f"doan dung it di, tren {n_kt} khach kiem tra")
