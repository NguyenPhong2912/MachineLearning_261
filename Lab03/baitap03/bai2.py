# -*- coding: utf-8 -*-
"""Bài tập 2: từ nào tố cáo mạnh nhất.

Huấn luyện MultinomialNB trên toàn bộ 300 tin, rồi lấy hiệu
feature_log_prob_[1] - feature_log_prob_[0] cho từng từ.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

mo_hinh = MultinomialNB(alpha=1.0).fit(X, y)

# Hiệu hai loga = log[P(từ | rác) / P(từ | thường)].
# Dương: từ nghiêng về rác, âm: nghiêng về thường, gần 0: không mang thông tin.
chenh = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]

# Thêm số đếm thô của từng lớp để đối chiếu
dem_rac = np.asarray(X[(y == 1).to_numpy()].sum(axis=0)).ravel()
dem_thuong = np.asarray(X[(y == 0).to_numpy()].sum(axis=0)).ravel()

bang = pd.DataFrame({
    "tu": vec.get_feature_names_out(),
    "dem_rac": dem_rac,
    "dem_thuong": dem_thuong,
    "chenh_log": chenh.round(4),
    "gap_lan": np.exp(chenh).round(2),   # P(từ|rác) gấp bao nhiêu lần P(từ|thường)
}).sort_values("chenh_log", ascending=False)

print("So tu trong tu dien:", len(bang))
print()
print("Nam tu to cao TIN RAC manh nhat:")
print(bang.head(5).to_string(index=False))
print()
print("Nam tu cho thay TIN THUONG manh nhat:")
print(bang.tail(5).iloc[::-1].to_string(index=False))
print()

# Từ trung tính: chênh lệch gần 0 nhất
trung_tinh = bang.loc[bang["chenh_log"].abs().sort_values().index[:3]]
print("Ba tu gan nhu khong mang thong tin (chenh lech gan 0 nhat):")
print(trung_tinh.to_string(index=False))
