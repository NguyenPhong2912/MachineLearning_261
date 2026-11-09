# -*- coding: utf-8 -*-
"""Bài tập 6: thời gian học của SVC và LinearSVC khi dữ liệu lớn dần.

Dữ liệu sinh bằng make_classification(n_samples=n, n_features=20,
random_state=0), n bằng 2000, 5000, 10000, 20000. Thời gian đo bằng mô đun time,
khác nhau giữa các máy, chỉ cần so tương đối. Mỗi lần đo lặp 3 lần và lấy
lần nhanh nhất để bớt nhiễu do máy bận việc khác.
"""

import time

from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC


def do_thoi_gian(lop_mo_hinh, X, y, so_lan=3):
    """Thoi gian .fit nhanh nhat trong so_lan lan do (giay)."""
    ket_qua = []
    for _ in range(so_lan):
        bat_dau = time.perf_counter()
        lop_mo_hinh().fit(X, y)
        ket_qua.append(time.perf_counter() - bat_dau)
    return min(ket_qua)


cac_co = [2000, 5000, 10000, 20000]
thoi_gian = {"SVC": [], "LinearSVC": []}

print("      n    SVC (giay)   LinearSVC (giay)   so vecto HT cua SVC")
for n in cac_co:
    X, y = make_classification(n_samples=n, n_features=20, random_state=0)

    t_svc = do_thoi_gian(SVC, X, y)
    t_lin = do_thoi_gian(LinearSVC, X, y)
    svc = SVC().fit(X, y)

    thoi_gian["SVC"].append(t_svc)
    thoi_gian["LinearSVC"].append(t_lin)
    print(f"  {n:6d}    {t_svc:9.3f}     {t_lin:9.4f}            {len(svc.support_)}")
print()

# Khi n tăng gấp đôi (10000 -> 20000), thời gian tăng bao nhiêu lần
print("Ti le thoi gian giua hai co lien nhau")
print("  co du lieu          gap    SVC    LinearSVC")
for i in range(1, len(cac_co)):
    gap = cac_co[i] / cac_co[i - 1]
    r_svc = thoi_gian["SVC"][i] / thoi_gian["SVC"][i - 1]
    r_lin = thoi_gian["LinearSVC"][i] / thoi_gian["LinearSVC"][i - 1]
    print(f"  {cac_co[i - 1]:5d} -> {cac_co[i]:5d}    {gap:4.1f}   {r_svc:5.2f}    {r_lin:5.2f}")
