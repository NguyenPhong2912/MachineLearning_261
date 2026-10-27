# -*- coding: utf-8 -*-
"""Bài tập 3: chạy lại f3_do_sau.py với criterion="entropy".

Entropy của một nút hai lớp: H = -p1 log2 p1 - p0 log2 p0, bằng 0 khi nút thuần
và bằng 1 khi hai lớp chia đôi.
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


def chay(tieu_chi):
    """Chon do sau bang kiem dinh cheo roi cham tap kiem tra mot lan."""
    tu_do = DecisionTreeClassifier(criterion=tieu_chi, random_state=0).fit(X_tr, y_tr)
    diem, hoc = {}, {}
    for d in range(1, 16):
        mh = DecisionTreeClassifier(criterion=tieu_chi, max_depth=d, random_state=0)
        diem[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        hoc[d] = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))
    d_tot = max(diem, key=diem.get)
    tot = DecisionTreeClassifier(criterion=tieu_chi, max_depth=d_tot,
                                 random_state=0).fit(X_tr, y_tr)
    return {
        "tu_do": tu_do, "diem": diem, "hoc": hoc, "d_tot": d_tot, "tot": tot,
        "acc": accuracy_score(y_te, tot.predict(X_te)),
        "acc_tu_do": accuracy_score(y_te, tu_do.predict(X_te)),
        "du_doan": tot.predict(X_te),
    }


g = chay("gini")
e = chay("entropy")

print("Cay moc tu do, criterion = 'entropy'")
print(f"  do sau {e['tu_do'].get_depth()}, so la {e['tu_do'].get_n_leaves()}")
print(f"  tap hoc 1.0000, tap kiem tra {e['acc_tu_do']:.4f}")
print()

print("Do sau   tap hoc (entropy)   KDC entropy   KDC gini")
for d in range(1, 16):
    print(f"  {d:2d}     {e['hoc'][d]:.4f}              {e['diem'][d]:.4f}        "
          f"{g['diem'][d]:.4f}")
print()

print("Tong ket      do sau tot   so la   diem KDC   tap kiem tra")
for ten, r in [("gini", g), ("entropy", e)]:
    print(f"  {ten:9s}   {r['d_tot']:10d}   {r['tot'].get_n_leaves():5d}   "
          f"{r['diem'][r['d_tot']]:.4f}     {r['acc']:.4f}")
print()

# Hai cây tốt nhất có hỏi cùng câu hỏi gốc và đoán giống nhau không
for ten, r in [("gini", g), ("entropy", e)]:
    t = r["tot"].tree_
    print(f"Cau hoi goc ({ten:7s}): {X.columns[t.feature[0]]} <= {t.threshold[0]:.3f}")
print(f"So khach kiem tra bi doan khac nhau: {(g['du_doan'] != e['du_doan']).sum()} / {len(y_te)}")
