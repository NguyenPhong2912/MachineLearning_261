# -*- coding: utf-8 -*-
"""Bài tập 4: cắt tỉa bằng số khách tối thiểu ở lá.

Cây không giới hạn độ sâu, chỉ đổi min_samples_leaf; chọn bằng kiểm định chéo
5 phần trên tập học, chấm tập kiểm tra một lần, rồi so với cây sâu 5 của mục 4.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("min_samples_leaf   do sau   so la   tap hoc   kiem dinh cheo")
diem = {}
for m in [1, 2, 5, 10, 20, 40]:
    mh = DecisionTreeClassifier(min_samples_leaf=m, random_state=0)
    diem[m] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    mh.fit(X_tr, y_tr)
    hoc = accuracy_score(y_tr, mh.predict(X_tr))
    print(f"  {m:6d}           {mh.get_depth():4d}    {mh.get_n_leaves():4d}    "
          f"{hoc:.4f}    {diem[m]:.4f}")
m_tot = max(diem, key=diem.get)
print()

# Chọn xong mới mở tập kiểm tra, và chỉ mở một lần
tot = DecisionTreeClassifier(min_samples_leaf=m_tot, random_state=0).fit(X_tr, y_tr)
acc = accuracy_score(y_te, tot.predict(X_te))

# Cây sâu 5 của mục 4, chọn bằng đúng cách đó
sau5 = DecisionTreeClassifier(max_depth=5, random_state=0)
diem_sau5 = cross_val_score(sau5, X_tr, y_tr, cv=5).mean()
sau5.fit(X_tr, y_tr)
acc_sau5 = accuracy_score(y_te, sau5.predict(X_te))

print("Cach cat tia              do sau   so la   diem KDC   tap kiem tra")
print(f"  min_samples_leaf = {m_tot:2d}   {tot.get_depth():6d}   {tot.get_n_leaves():5d}"
      f"   {diem[m_tot]:.4f}     {acc:.4f}")
print(f"  max_depth = 5           {sau5.get_depth():6d}   {sau5.get_n_leaves():5d}"
      f"   {diem_sau5:.4f}     {acc_sau5:.4f}")
print()

# Lá nhỏ nhất của cây sâu 5 có bao nhiêu khách
la = sau5.apply(X_tr)
print(f"Cay sau 5: la nho nhat co {pd.Series(la).value_counts().min()} khach, "
      f"so la duoi {m_tot} khach: {(pd.Series(la).value_counts() < m_tot).sum()}")
