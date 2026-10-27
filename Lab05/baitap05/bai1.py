# -*- coding: utf-8 -*-
"""Bài tập 1: tự vẽ cây bằng plot_tree.

Vẽ lại cây năm lá của f2_cay_nam_la.py, lưu thành cay_nam_la.png; rồi vẽ cây
sâu 2, lưu thành cay_sau_2.png. Hai tệp ảnh nằm ở thư mục đang đứng chạy lệnh.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

df = pd.read_csv("data/khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
ten_cot = list(X.columns)
ten_lop = ["tre han", "dung han"]       # lớp 0 trước, lớp 1 sau


def ve_cay(cay, ten_tep, tieu_de, kich_thuoc):
    fig, ax = plt.subplots(figsize=kich_thuoc)
    plot_tree(cay, feature_names=ten_cot, class_names=ten_lop,
              filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title(tieu_de)
    fig.savefig(ten_tep, dpi=150, bbox_inches="tight")
    plt.close(fig)                      # chỉ lưu tệp, không mở cửa sổ
    print(f"Da luu {ten_tep}")


nam_la = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
ve_cay(nam_la, "cay_nam_la.png", "Cay nam la (max_leaf_nodes=5)", (14, 7))

sau_2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
ve_cay(sau_2, "cay_sau_2.png", "Cay sau 2 (max_depth=2)", (11, 5))
print()

print("Cay sau 2 dang chu:")
print(export_text(sau_2, feature_names=ten_cot, show_weights=True, decimals=3))

# Xem kỹ nút bên trái của gốc và hai lá con của nó
t = sau_2.tree_
trai = t.children_left[0]
la_1, la_2 = t.children_left[trai], t.children_right[trai]


def mo_ta(nut):
    # value lưu tỷ lệ (scikit-learn >= 1.4) hoặc số đếm (bản cũ); quy về số khách
    dem = t.value[nut][0] / t.value[nut][0].sum() * t.n_node_samples[nut]
    lop = ten_lop[int(dem.argmax())]
    return (f"{t.n_node_samples[nut]:3d} khach, tre {dem[0]:5.0f}, dung {dem[1]:5.0f}, "
            f"gini {t.impurity[nut]:.3f}, P(dung han) {dem[1] / dem.sum():.2f} -> {lop}")


print("Nhanh trai cua goc (thu_nhap <= 15.05)")
print(f"  nut cha : {mo_ta(trai)}")
print(f"  la trai : {mo_ta(la_1)}")
print(f"  la phai : {mo_ta(la_2)}")
n = t.n_node_samples
g_sau = (n[la_1] * t.impurity[la_1] + n[la_2] * t.impurity[la_2]) / n[trai]
print(f"  Gini truoc chia {t.impurity[trai]:.4f}, sau chia {g_sau:.4f}, "
      f"giam {t.impurity[trai] - g_sau:.4f}")
