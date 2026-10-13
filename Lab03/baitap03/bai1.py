# -*- coding: utf-8 -*-
"""Bài tập 1: định lý Bayes với ba nguyên nhân.

Mở rộng hàm bayes của d1_bayes_tay.py từ hai nhánh (A đúng, A sai) thành
nhiều nhánh. Mẫu số vẫn dựng từ công thức xác suất toàn phần, chỉ là cộng
qua ba nhánh thay vì hai.
"""


def bayes_nhieu_nhanh(tien_nghiem, kha_nang):
    """Tra ve (danh sach hau nghiem, P(B)).

    tien_nghiem : [P(A1), P(A2), ...], cac nhanh roi nhau, cong lai bang 1
    kha_nang    : [P(B | A1), P(B | A2), ...]
    """
    tu_so = [p * l for p, l in zip(tien_nghiem, kha_nang)]
    mau_so = sum(tu_so)               # P(B), xac suat toan phan
    return [t / mau_so for t in tu_so], mau_so


ten_nhom = ["hoc deu", "hoc don", "it hoc"]
ty_le_nhom = [0.50, 0.30, 0.20]       # tiên nghiệm P(nhóm)
ty_le_qua = [0.95, 0.70, 0.30]        # khả năng P(qua | nhóm)

hau_nghiem, p_qua = bayes_nhieu_nhanh(ty_le_nhom, ty_le_qua)

print("Ba nhom sinh vien")
print(f"  {'nhom':10s} {'P(nhom)':>8s} {'P(qua | nhom)':>14s}")
for ten, p, l in zip(ten_nhom, ty_le_nhom, ty_le_qua):
    print(f"  {ten:10s} {p:8.2f} {l:14.2f}")
print()
print(f"Ty le qua mon chung P(qua) = "
      + " + ".join(f"{p:.2f}*{l:.2f}" for p, l in zip(ty_le_nhom, ty_le_qua))
      + f" = {p_qua:.4f}")
print()

print("Gap mot ban da qua mon, kha nang ban do thuoc tung nhom:")
print(f"  {'nhom':10s} {'truoc (tien nghiem)':>20s} {'sau (hau nghiem)':>18s}")
for ten, truoc, sau in zip(ten_nhom, ty_le_nhom, hau_nghiem):
    print(f"  {ten:10s} {truoc:20.4f} {sau:18.4f}")
print(f"  Tong ba hau nghiem = {sum(hau_nghiem):.4f}")
print()

# Kiểm lại bằng cách đếm, giống ví dụ hai dây chuyền trong d1_bayes_tay.py
print("Kiem lai bang cach dem tren 1000 sinh vien:")
so_qua = [1000 * p * l for p, l in zip(ty_le_nhom, ty_le_qua)]
for ten, p, q in zip(ten_nhom, ty_le_nhom, so_qua):
    print(f"  {ten:10s}: {1000 * p:4.0f} ban, {q:4.0f} ban qua mon")
tong_qua = sum(so_qua)
print(f"  Tong so ban qua mon = {tong_qua:.0f}")
for ten, q in zip(ten_nhom, so_qua):
    print(f"  Ty le {ten:8s} = {q:4.0f} / {tong_qua:.0f} = {q / tong_qua:.4f}")
