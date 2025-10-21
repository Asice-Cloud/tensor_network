#!/usr/bin/env python3
"""
Example 4: COPY and XOR tensors

实现论文中 Example 4 的 COPY 和 XOR 三阶张量（每个索引取值 0/1），并演示如何通过张量收缩重建 CNOT 门。

COPY^{i}_{j k} = 1 当且仅当 i=j=k（即所有三条腿值相同），否则 0。
XOR^{i}_{j k} = 1 当且仅当 i = j XOR k（或等价地 j+k+i = 0 mod 2 取偶数），具体按照论文对 parity 的定义。

脚本将：
 - 构造 COPY 和 XOR 的数组表示（shape (2,2,2)）。
 - 按照公式对中间索引 m 求和收缩 COPY 和 XOR，得到一个秩-4 张量，重塑为 4x4 矩阵并与标准 CNOT 比较。

注：这里采用索引排列与论文一致的约定，使用 np.einsum 进行清晰的张量收缩。
"""

import numpy as np


def copy_tensor():
    """返回 COPY 张量，shape (2,2,2)，索引顺序 (out, in1, in2) 或任意命名约定。

    定义为 1 当且仅当三个指标相等（0或1），否则 0。
    """
    C = np.zeros((2,2,2), dtype=int)
    for a in (0,1):
        C[a,a,a] = 1
    return C


def xor_tensor():
    """返回 XOR（parity）张量，shape (2,2,2)，定义为 1 当且仅当输出索引为输入索引之 XOR。"""
    X = np.zeros((2,2,2), dtype=int)
    for out in (0,1):
        for i in (0,1):
            for j in (0,1):
                if out == (i ^ j):
                    X[out,i,j] = 1
    return X


def standard_cnot():
    """返回标准 CNOT 矩阵（4x4），基序 |00>,|01>,|10>,|11>。"""
    P0 = np.array([[1,0],[0,0]], dtype=int)
    P1 = np.array([[0,0],[0,1]], dtype=int)
    X = np.array([[0,1],[1,0]], dtype=int)
    C = np.kron(P0, np.eye(2, dtype=int)) + np.kron(P1, X)
    return C


def build_cnot_from_tensors(COPY, XOR):
    """按论文 (29) 的形式收缩 COPY 和 XOR 重建 CNOT。

    具体地，我们希望对中间索引 m 求和：
      sum_m COPY^{q m}_{i} XOR^{r}_{m j} = CNOT^{q r}_{i j}

    其中我们使用数组指标顺序：COPY[out_q, i, m]，XOR[out_r, m, j]
    使用 einsum: 'qim,rmj->qrij'，然后重排为矩阵 (qr) x (ij)
    """
    # einsum 得到形状 (q,r,i,j)
    T = np.einsum('qim,rmj->qrij', COPY, XOR)
    # reshape 为 4x4 矩阵，行索引 (q,r), 列索引 (i,j)
    T_mat = T.reshape(4,4)
    return T_mat


def idx_pair_to_lin(a,b):
    return a*2 + b


def main():
    COPY = copy_tensor()
    XOR = xor_tensor()

    print('COPY tensor (out,i,m):')
    print(COPY)
    print('\nXOR tensor (out,m,j):')
    print(XOR)

    built = build_cnot_from_tensors(COPY, XOR)
    print('\nCNOT built from COPY/XOR:')
    print(built)

    std = standard_cnot()
    print('\nStandard CNOT:')
    print(std)

    diff = np.linalg.norm(built - std)
    print('\nFrobenius norm difference =', diff)
    if diff == 0:
        print('Success: built CNOT equals standard CNOT')
    else:
        print('Mismatch: check index conventions')


if __name__ == '__main__':
    main()
