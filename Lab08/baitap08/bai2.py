# -*- coding: utf-8 -*-
"""Bài tập 2: phân cụm trên cả ba cột tuổi, thu nhập, điểm chi tiêu.

Có chuẩn hoá; chọn lại K bằng silhouette với K từ 2 tới 10, rồi đọc tâm cụm.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/khach_hang.csv")
COT = ["tuoi", "thu_nhap", "diem_chi_tieu"]
sc = StandardScaler().fit(df[COT])
Xs = sc.transform(df[COT])

print(" K   inertia   silhouette")
diem = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    diem[k] = silhouette_score(Xs, km.labels_)
    print(f" {k:2d}   {km.inertia_:7.1f}   {diem[k]:.3f}")
k_tot = max(diem, key=diem.get)
print(f"K tot nhat theo silhouette: {k_tot}")
print()


def in_tam(k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    tam = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=COT)
    tam["so_khach"] = np.bincount(km.labels_)
    # Xếp theo thu nhập rồi chi tiêu cho dễ so với mục 3
    print(f"Tam cum voi K = {k}, ba cot, don vi goc:")
    print(tam.sort_values(["thu_nhap", "diem_chi_tieu"]).round(1).to_string())
    print()
    return km


km_tot = in_tam(k_tot)
if k_tot != 5:
    in_tam(5)

# So với cách chia của mục 3 (chỉ hai cột)
X2 = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
nhan_2 = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(X2)
print("Bang cheo: hang la cum muc 3 (hai cot), cot la cum khi them tuoi")
print(pd.crosstab(nhan_2, km_tot.labels_, rownames=["hai_cot"], colnames=["ba_cot"]))
