# -*- coding: utf-8 -*-
"""Bài tập 4: lưu ảnh đã nén 16 màu và so dung lượng tệp.

Phần 1 dùng ảnh china.jpg như k5_nen_anh.py. Phần 2 làm lại với một ảnh đọc bằng
plt.imread: ở đây dùng ảnh flower.jpg của scikit-learn, lưu thành PNG có kênh
độ trong suốt (thay cho ảnh riêng của sinh viên), để thấy phải bỏ kênh thứ tư.
Các tệp ảnh nằm ở thư mục đang đứng chạy lệnh.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image


def nen_mau(anh, k, so_mau_hoc=5000):
    """Tra ve anh chi con k mau (gia tri 0..1) va sai lech trung binh (muc sang)."""
    cao, rong, _ = anh.shape
    diem = anh.reshape(-1, 3)
    rng = np.random.default_rng(0)
    mau = diem[rng.choice(len(diem), so_mau_hoc, replace=False)]
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(mau)
    moi = km.cluster_centers_[km.predict(diem)]
    sai = np.abs(moi - diem).mean() * 255
    return moi.reshape(cao, rong, 3), sai


def kb(ten_tep):
    return os.path.getsize(ten_tep) / 1024


# Phần 1: ảnh china.jpg
anh = load_sample_image("china.jpg") / 255.0     # đưa về 0..1 cho plt.imsave
anh_moi, sai = nen_mau(anh, 16)
print("Dang mang anh_moi:", anh_moi.shape)
plt.imsave("anh_goc.png", anh)
plt.imsave("anh_16_mau.png", anh_moi)
print(f"Sai lech trung binh: {sai:.2f} muc sang")
print(f"anh_goc.png    : {kb('anh_goc.png'):7.1f} KB")
print(f"anh_16_mau.png : {kb('anh_16_mau.png'):7.1f} KB"
      f"  (nho hon {kb('anh_goc.png') / kb('anh_16_mau.png'):.1f} lan)")
print()

# Phần 2: ảnh đọc bằng plt.imread. Tạo trước một tệp PNG bốn kênh làm đầu vào.
plt.imsave("anh_rieng.png", load_sample_image("flower.jpg"))
rieng = plt.imread("anh_rieng.png")
print("anh_rieng.png doc vao co dang", rieng.shape, "kieu", rieng.dtype)
if rieng.ndim == 3 and rieng.shape[2] == 4:
    print("  Co kenh thu tu (do trong suot), gia tri:", np.unique(rieng[:, :, 3]))
    rieng = rieng[:, :, :3]                       # bỏ kênh độ trong suốt
if rieng.dtype == np.uint8:                       # ảnh JPG đọc ra 0..255, PNG ra 0..1
    rieng = rieng / 255.0
rieng_moi, sai2 = nen_mau(rieng, 16)
plt.imsave("anh_rieng_16_mau.png", rieng_moi)
print(f"Sai lech trung binh: {sai2:.2f} muc sang")
print(f"anh_rieng.png         : {kb('anh_rieng.png'):7.1f} KB")
print(f"anh_rieng_16_mau.png  : {kb('anh_rieng_16_mau.png'):7.1f} KB"
      f"  (nho hon {kb('anh_rieng.png') / kb('anh_rieng_16_mau.png'):.1f} lan)")
