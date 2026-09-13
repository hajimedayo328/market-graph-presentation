"""6角形の輪 → 対角線で分割 → 左が埋まる → 右が埋まる、の寿命を追う."""
import warnings; warnings.simplefilter("ignore")
import numpy as np
from ripser import ripser

# 6点 A,B,C,D,E,F を 0..5
n = 6
BIG = 9.0
D = np.full((n, n), BIG)
np.fill_diagonal(D, 0)

def edge(i, j, d):
    D[i, j] = D[j, i] = d

# 半径(=距離)の順に辺が生える
# 1.0: 6角形の外周 A-B-C-D-E-F-A
for i in range(6):
    edge(i, (i + 1) % 6, 1.0)
# 1.2: 対角線 A-D → 輪が左(ABCD)と右(ADEF)に分かれる
edge(0, 3, 1.2)
# 1.4: 左の輪を埋める辺 A-C, B-D → 左の三角形が全部揃う
edge(0, 2, 1.4); edge(1, 3, 1.4)
# 1.6: 右の輪を埋める辺 A-E, D-F
edge(0, 4, 1.6); edge(3, 5, 1.6)
# 残り(B-E, C-F, B-F, C-E)は遠い
for i, j in [(1, 4), (2, 5), (1, 5), (2, 4)]:
    edge(i, j, 2.0)

res = ripser(D, maxdim=1, distance_matrix=True)
h1 = res["dgms"][1]
print("半径の順に起きること:")
print("  1.0  6角形の外周が閉じる  → 穴が生まれる")
print("  1.2  対角線A-Dが生える     → 輪が左右に分かれる")
print("  1.4  左の輪が三角形で埋まる")
print("  1.6  右の輪が三角形で埋まる")
print()
print("ripser が出した H1 の穴:")
for b, d in sorted(h1, key=lambda x: x[0]):
    print(f"  誕生 {b:.1f} → 消滅 {d:.1f}   寿命 = {d-b:.1f}")
print(f"\n穴の本数: {len(h1)}   L1 = {sum(d-b for b,d in h1):.1f}")
