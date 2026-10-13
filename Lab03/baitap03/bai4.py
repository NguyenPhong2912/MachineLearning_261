# -*- coding: utf-8 -*-
"""Bài tập 4: tự viết bộ lọc Naive Bayes, không dùng thư viện học máy.

Dựa trên d4_lam_tron.py nhưng chạy trên cả 300 tin: tự đếm từ, tự làm trơn
Laplace, tự tính điểm hai lớp bằng TỔNG LOGA thay vì tích. Scikit-learn chỉ
dùng để chia dữ liệu và để đối chiếu kết quả ở cuối.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def log_an_toan(x):
    """log(x), tra ve -vo cung khi x = 0 thay vi bao loi."""
    return math.log(x) if x > 0 else float("-inf")


class BoLocTuViet:
    """Multinomial Naive Bayes tu cai dat.

    alpha          : he so lam tron Laplace
    co_tien_nghiem : False de thu loi quen nhan tien nghiem P(lop)
    cong_alpha_V   : False de thu loi quen cong alpha*V o mau so
    """

    def __init__(self, alpha=1.0, co_tien_nghiem=True, cong_alpha_V=True):
        self.alpha = alpha
        self.co_tien_nghiem = co_tien_nghiem
        self.cong_alpha_V = cong_alpha_V

    def fit(self, cac_cau, cac_nhan):
        # Từ điển chỉ lập từ tập học, giống fit của CountVectorizer
        self.tu_dien = sorted({t for cau in cac_cau for t in cau.split()})
        self.V = len(self.tu_dien)

        # Đếm số tin, số lần xuất hiện của từng từ và tổng số từ theo lớp
        self.so_tin = Counter(cac_nhan)
        self.dem = {0: Counter(), 1: Counter()}
        self.tong = {0: 0, 1: 0}
        for cau, nhan in zip(cac_cau, cac_nhan):
            for t in cau.split():
                self.dem[nhan][t] += 1
                self.tong[nhan] += 1

        # Loga tiên nghiệm và loga xác suất từ, tính sẵn một lần
        n = sum(self.so_tin.values())
        self.log_tien_nghiem = {k: math.log(self.so_tin[k] / n) for k in (0, 1)}
        self.log_xs = {}
        for k in (0, 1):
            mau_so = self.tong[k] + (self.alpha * self.V if self.cong_alpha_V else 0)
            self.log_xs[k] = {t: log_an_toan((self.dem[k][t] + self.alpha) / mau_so)
                              for t in self.tu_dien}
        return self

    def diem(self, cau, nhan):
        """log P(lop) + tong log P(tu | lop), tu ngoai tu dien thi bo qua."""
        d = self.log_tien_nghiem[nhan] if self.co_tien_nghiem else 0.0
        for t in cau.split():
            if t in self.log_xs[nhan]:
                d += self.log_xs[nhan][t]
        return d

    def predict(self, cac_cau):
        # Hòa điểm thì chọn lớp 0, giống argmax của scikit-learn
        return [1 if self.diem(c, 1) > self.diem(c, 0) else 0 for c in cac_cau]

    def xac_suat_rac(self, cac_cau):
        """P(rac | cau) = 1 / (1 + e^(diem thuong - diem rac)), khong tran so."""
        return [1 / (1 + math.exp(self.diem(c, 0) - self.diem(c, 1))) for c in cac_cau]


df = pd.read_csv("data/tin_nhan.csv")

# Chia trên cột chữ: cùng số dòng, cùng nhãn stratify, cùng random_state
# nên ra đúng các chỉ số như khi chia ma trận X trong d3_loc_thu_rac.py
cau_train, cau_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["la_rac"], test_size=0.25, random_state=17,
    stratify=df["la_rac"]
)
cau_train, cau_test = list(cau_train), list(cau_test)
y_train, y_test = list(y_train), list(y_test)

# 1. Bộ lọc tự viết
tu_viet = BoLocTuViet(alpha=1.0).fit(cau_train, y_train)
y_tu_viet = tu_viet.predict(cau_test)
print("Bo loc tu viet")
print(f"  So tu trong tu dien (lap tu tap hoc): {tu_viet.V}")
print(f"  So tin hoc: {len(cau_train)}  (thuong {tu_viet.so_tin[0]}, rac {tu_viet.so_tin[1]})")
print(f"  Tong so tu: lop thuong {tu_viet.tong[0]}, lop rac {tu_viet.tong[1]}")
print(f"  Accuracy = {accuracy_score(y_test, y_tu_viet):.4f}")
print()

# 2. MultinomialNB với từ điển lập từ tập học, đúng điều kiện với bản tự viết
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_train = vec.fit_transform(cau_train)
X_test = vec.transform(cau_test)
thu_vien = MultinomialNB(alpha=1.0).fit(X_train, y_train)
y_thu_vien = thu_vien.predict(X_test)
print("MultinomialNB")
print(f"  So tu trong tu dien: {len(vec.vocabulary_)}")
print(f"  Accuracy = {accuracy_score(y_test, y_thu_vien):.4f}")
print()

# 3. Đối chiếu từng con số, không chỉ accuracy
so_khac = sum(a != b for a, b in zip(y_tu_viet, y_thu_vien))
diem_tu_viet = np.array([[tu_viet.diem(c, 0), tu_viet.diem(c, 1)] for c in cau_test])
diem_thu_vien = thu_vien.predict_joint_log_proba(X_test)
print("Doi chieu hai ban")
print(f"  Tu dien giong nhau?          {list(vec.get_feature_names_out()) == tu_viet.tu_dien}")
print(f"  So tin du doan khac nhau:    {so_khac} / {len(cau_test)}")
print(f"  Lech lon nhat cua diem loga: {np.abs(diem_tu_viet - diem_thu_vien).max():.2e}")
print()

# 4. Vì sao phải cộng loga: thử một tin rất dài
tin_dai = " ".join([cau_test[0]] * 40)
tich = 1.0
for t in tin_dai.split():
    if t in tu_viet.tu_dien:
        tich *= math.exp(tu_viet.log_xs[1][t])
print(f"Tin dai {len(tin_dai.split())} tu: tich xac suat lop rac = {tich}"
      f"   tong loga = {tu_viet.diem(tin_dai, 1):.2f}")
print()

# 5. Cố tình cài sai từng chỗ, rồi so với thư viện ở cả nhãn lẫn xác suất
p_thu_vien = thu_vien.predict_proba(X_test)[:, 1]
print("Thu cac loi hay gap               Accuracy  Nhan khac  Lech P(rac) lon nhat")
for ten, bo_loc in [
    ("Cai dung (alpha = 1)            ", BoLocTuViet()),
    ("Quen tien nghiem P(lop)         ", BoLocTuViet(co_tien_nghiem=False)),
    ("Quen cong alpha*V o mau so      ", BoLocTuViet(cong_alpha_V=False)),
    ("Khong lam tron (alpha = 0)      ", BoLocTuViet(alpha=0.0)),
]:
    bo_loc.fit(cau_train, y_train)
    y_loi = bo_loc.predict(cau_test)
    lech_p = np.abs(np.array(bo_loc.xac_suat_rac(cau_test)) - p_thu_vien).max()
    so_nhan_khac = sum(a != b for a, b in zip(y_loi, y_thu_vien))
    print(f"  {ten} {accuracy_score(y_test, y_loi):.4f}    {so_nhan_khac:3d}        {lech_p:.4f}")

# Vì sao alpha = 0 không làm sập bộ lọc trên tập học này
so_tu_vang = {k: sum(tu_viet.dem[k][t] == 0 for t in tu_viet.tu_dien) for k in (0, 1)}
print(f"  So tu co dem bang 0 trong tap hoc: lop thuong {so_tu_vang[0]}, lop rac {so_tu_vang[1]}")
