# -*- coding: utf-8 -*-
"""Bài tập 6: lớp hiếm và xác suất tiên nghiệm.

Giữ cả 180 tin thường nhưng chỉ lấy 20 tin rác đầu tiên, rồi huấn luyện và
chấm điểm lại như d3_loc_thu_rac.py.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")

# Bỏ mọi tin rác từ tin thứ 21 trở đi, giữ nguyên thứ tự các dòng còn lại
chi_so_rac = df.index[df["la_rac"] == 1]
df_lech = df.drop(chi_so_rac[20:]).reset_index(drop=True)

print("Bo du lieu moi:", df_lech.shape)
print(df_lech["la_rac"].value_counts())
print(f"Ty le tin rac: {df_lech['la_rac'].mean():.4f}")
print()

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df_lech["noi_dung"])
y = df_lech["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
print(f"Tap hoc {X_train.shape[0]} tin ({int(y_train.sum())} rac), "
      f"tap kiem tra {X_test.shape[0]} tin ({int(y_test.sum())} rac)")
print()


def cham_diem(ten, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1)
    print(f"  {ten:34s} {acc:.4f}    {pre:.4f}     {rec:.4f}    "
          f"{tn:3d}{fp:3d}{fn:3d}{tp:3d}")


mo_hinh = MultinomialNB(alpha=1.0).fit(X_train, y_train)
# MultinomialNB chỉ lưu loga tiên nghiệm, lấy mũ để về lại tỷ lệ
ty_le = ", ".join(f"{v:.4f}" for v in np.exp(mo_hinh.class_log_prior_))
print(f"Tien nghiem hoc duoc (thuong, rac): {ty_le}")
print()

print(f"  {'Mo hinh':34s} Accuracy  Precision  Recall(rac)  TN FP FN TP")
cham_diem("MultinomialNB, du lieu lech", mo_hinh.predict(X_test))
# Mốc đối chứng ngớ ngẩn: không chặn tin nào cả
cham_diem("Luon doan 'thuong' (khong loc gi)", [0] * X_test.shape[0])
# Bỏ tiên nghiệm, coi hai lớp ngang nhau 0.5 / 0.5
mo_hinh_deu = MultinomialNB(alpha=1.0, fit_prior=False).fit(X_train, y_train)
cham_diem("MultinomialNB, tien nghiem 0.5/0.5", mo_hinh_deu.predict(X_test))
print()

# Tập kiểm tra chỉ có 5 tin rác nên recall nhảy từng bậc 0.2. Kiểm thêm trên
# 100 tin rác đã bị bỏ ra ở trên: mô hình chưa từng thấy chúng lúc học.
rac_bi_bo = df.loc[chi_so_rac[20:], "noi_dung"]
X_bo = vec.transform(rac_bi_bo)
print(f"Kiem them tren {len(rac_bi_bo)} tin rac bi bo ra (tat ca deu la rac):")
print(f"  {'Mo hinh':34s} Bat duoc   Recall   P(rac) trung binh")
for ten, m in [("MultinomialNB, du lieu lech", mo_hinh),
               ("MultinomialNB, tien nghiem 0.5/0.5", mo_hinh_deu)]:
    bat = int(m.predict(X_bo).sum())
    print(f"  {ten:34s} {bat:3d}/{len(rac_bi_bo)}    {bat / len(rac_bi_bo):.4f}"
          f"   {m.predict_proba(X_bo)[:, 1].mean():.4f}")
print()

# Đối chiếu với bộ cân bằng 180 / 120 của d3_loc_thu_rac.py
vec_goc = CountVectorizer(token_pattern=r"(?u)\S+")
X_goc = vec_goc.fit_transform(df["noi_dung"])
a, b, c, d = train_test_split(X_goc, df["la_rac"], test_size=0.25,
                              random_state=17, stratify=df["la_rac"])
y_goc = MultinomialNB(alpha=1.0).fit(a, c).predict(b)
print("Doi chieu bo can bang 180 / 120 (d3_loc_thu_rac.py):")
print(f"  Accuracy = {accuracy_score(d, y_goc):.4f}, "
      f"Recall(rac) = {recall_score(d, y_goc, pos_label=1):.4f}")
