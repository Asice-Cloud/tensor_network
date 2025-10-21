
#!/usr/bin/env python3
"""
example3_demo.py

这个脚本演示论文 Example 3 中的简单量子电路：
先对第一个比特施加 Hadamard 门 H，然后对两比特施加 CNOT（第一个比特为控制，比特）。
目的是验证：
  - 输入 |00> 会被映射为 Bell 态 (|00> + |11>)/sqrt(2)
  - 输入 |11> 会被映射为 (|01> - |10>)/sqrt(2)（singlet）

脚本仅使用 NumPy，基序采用标准的 lexicographic 顺序：|00>, |01>, |10>, |11>。
"""

import numpy as np


def hadamard():
    """返回单比特 Hadamard 矩阵 H。

    H = 1/sqrt(2) * [[1, 1], [1, -1]]，将基 |0>,|1> 变换到 (|0>±|1>)/sqrt(2)。
    """
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def cnot():
    """构造两比特 CNOT 门。

    我们使用投影算子表示：CNOT = |0><0| ⊗ I + |1><1| ⊗ X。
    这里第一比特为控制（control），第二比特为 target。
    基序（向量展平顺序）为 |00>,|01>,|10>,|11>。
    返回大小为 4x4 的复矩阵。
    """
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|
    X = np.array([[0, 1], [1, 0]], dtype=complex)   # Pauli-X
    C = np.kron(P0, np.eye(2)) + np.kron(P1, X)
    return C


def bell_state():
    """返回 Bell 态 (|00> + |11>)/sqrt(2) 的列向量（长度 4）。"""
    v = np.zeros(4, dtype=complex)
    v[0] = 1 / np.sqrt(2)
    v[3] = 1 / np.sqrt(2)
    return v


def singlet_state():
    """返回 (|01> - |10>)/sqrt(2) 的列向量（长度 4）。

    注意：论文在该处称其为 singlet，但严格的 Bell 态家族还有其它等价态；此处为常见的反对称组合。
    """
    v = np.zeros(4, dtype=complex)
    v[1] = 1 / np.sqrt(2)
    v[2] = -1 / np.sqrt(2)
    return v


def run_demo():
    """运行演示：对 |00> 和 |11> 分别施加 H（作用在第一比特）然后 CNOT，打印结果与期望比较。"""
    H = hadamard()
    C = cnot()

    # 我们希望先在第一比特上施加 H（因此使用 H ⊗ I），随后施加 CNOT
    U = np.kron(H, np.eye(2))  # H on qubit-0, identity on qubit-1

    # 测试 1: 初态 |00>
    psi00 = np.zeros(4, dtype=complex)
    psi00[0] = 1  # |00>
    out = C.dot(U.dot(psi00))
    print('Output for |00> -> should be Bell (|00>+|11>)/sqrt2')
    print('out =', np.round(out, 6))
    print('expected =', bell_state())
    print('norm diff =', np.linalg.norm(out - bell_state()))

    # 测试 2: 初态 |11>
    psi11 = np.zeros(4, dtype=complex)
    psi11[3] = 1  # |11>
    out2 = C.dot(U.dot(psi11))
    print('\nOutput for |11> -> should be singlet (|01> - |10>)/sqrt2')
    print('out =', np.round(out2, 6))
    print('expected =', singlet_state())
    print('norm diff =', np.linalg.norm(out2 - singlet_state()))


if __name__ == '__main__':
    run_demo()

