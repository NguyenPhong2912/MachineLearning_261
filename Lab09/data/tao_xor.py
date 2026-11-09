# -*- coding: utf-8 -*-
"""Sinh lại data/xor.csv (đề chỉ mô tả, không phát kèm mã sinh).

400 điểm rải đều trên hình vuông [-1, 1] x [-1, 1]. Nhãn 1 khi x1 và x2 cùng dấu
(góc trên phải và góc dưới trái), rồi đảo nhãn ngẫu nhiên khoảng 5% số điểm làm
nhiễu. Chạy từ thư mục Lab09: python data/tao_xor.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (400, 2))
nhan = (X[:, 0] * X[:, 1] > 0).astype(int)
dao = rng.random(400) < 0.05            # điểm bị đảo nhãn
nhan = np.where(dao, 1 - nhan, nhan)

df = pd.DataFrame({"x1": X[:, 0].round(4), "x2": X[:, 1].round(4), "nhan": nhan})
df.to_csv("data/xor.csv", index=False)
print("Da luu data/xor.csv:", df.shape, "| so diem lop 1:", nhan.sum(),
      "| so nhan bi dao:", dao.sum())
