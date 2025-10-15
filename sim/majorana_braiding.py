"""
Majorana零模的辫子群操作模拟
演示如何用编织实现量子门操作
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class MajoranaQubit:
    """使用4个Majorana零模编码的量子比特"""
    
    def __init__(self):
        """
        初始化态在计算基 {|01⟩, |10⟩} 子空间
        4个Majorana算符: γ₁, γ₂, γ₃, γ₄
        """
        # 状态向量：[|01⟩, |10⟩] 的系数
        self.state = np.array([1.0, 0.0], dtype=complex)
        self.braiding_history = []
        
    def braiding_operator(self, i: int, j: int) -> np.ndarray:
        """
        计算交换 γᵢ 和 γⱼ 的辫子算符
        σᵢⱼ = exp(π/4 · γᵢγⱼ)
        
        在 {|01⟩, |10⟩} 子空间中的表示
        """
        if (i, j) == (1, 2) or (i, j) == (3, 4):
            # 同一费米子内部交换 → Z旋转
            return np.array([
                [np.exp(1j * np.pi/4), 0],
                [0, np.exp(-1j * np.pi/4)]
            ])
        elif (i, j) == (2, 3):
            # 不同费米子之间交换 → X旋转
            theta = np.pi/4
            return np.array([
                [np.cos(theta), 1j * np.sin(theta)],
                [1j * np.sin(theta), np.cos(theta)]
            ])
        else:
            raise ValueError(f"Invalid Majorana pair: ({i}, {j})")
    
    def braid(self, i: int, j: int, inverse: bool = False):
        """执行辫子操作"""
        U = self.braiding_operator(i, j)
        if inverse:
            U = U.conj().T
        self.state = U @ self.state
        self.braiding_history.append((i, j, inverse))
    
    def measure_z(self) -> Tuple[int, float]:
        """
        测量 Z 基
        返回：(结果, 概率)
        """
        prob_0 = np.abs(self.state[0])**2
        prob_1 = np.abs(self.state[1])**2
        
        result = 0 if np.random.random() < prob_0 else 1
        return result, prob_0 if result == 0 else prob_1
    
    def get_bloch_vector(self) -> np.ndarray:
        """计算Bloch球上的坐标"""
        rho = np.outer(self.state, self.state.conj())
        
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        return np.array([x, y, z])
    
    def __repr__(self):
        return f"MajoranaQubit(|ψ⟩ = {self.state[0]:.3f}|01⟩ + {self.state[1]:.3f}|10⟩)"


class BraidingGates:
    """常用量子门的辫子实现"""
    
    @staticmethod
    def hadamard_sequence() -> List[Tuple[int, int, bool]]:
        """Hadamard门的辫子序列（近似）"""
        return [
            (2, 3, False),
            (1, 2, False),
            (2, 3, False),
        ]
    
    @staticmethod
    def pauli_x_sequence() -> List[Tuple[int, int, bool]]:
        """Pauli X门的辫子序列"""
        return [(2, 3, False)] * 4
    
    @staticmethod
    def pauli_z_sequence() -> List[Tuple[int, int, bool]]:
        """Pauli Z门的辫子序列"""
        return [(1, 2, False)] * 4
    
    @staticmethod
    def phase_s_sequence() -> List[Tuple[int, int, bool]]:
        """相位门 S 的辫子序列"""
        return [(1, 2, False)] * 2
    
    @staticmethod
    def t_gate_sequence() -> List[Tuple[int, int, bool]]:
        """T门的辫子序列（近似）"""
        return [(1, 2, False)]


def visualize_bloch_sphere(states: List[np.ndarray], labels: List[str]):
    """在Bloch球上可视化量子态演化"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制Bloch球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # 绘制坐标轴
    ax.quiver(0, 0, 0, 1.3, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, 1.3, 0, color='g', arrow_length_ratio=0.1, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1.3, color='b', arrow_length_ratio=0.1, label='Z')
    
    # 绘制量子态轨迹
    colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
    for i, (state, label) in enumerate(zip(states, labels)):
        ax.scatter(*state, color=colors[i], s=100, label=label)
        if i > 0:
            # 连接轨迹
            prev = states[i-1]
            ax.plot([prev[0], state[0]], 
                   [prev[1], state[1]], 
                   [prev[2], state[2]], 
                   'k--', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Majorana编织操作的Bloch球轨迹')
    ax.legend()
    
    plt.tight_layout()
    return fig


def demo_single_qubit_gates():
    """演示单量子比特门的辫子实现"""
    print("=" * 60)
    print("Majorana零模辫子操作演示")
    print("=" * 60)
    
    # 测试Hadamard门
    print("\n1. Hadamard门（近似）")
    qubit = MajoranaQubit()
    print(f"   初始态: {qubit}")
    
    for i, j, inv in BraidingGates.hadamard_sequence():
        qubit.braid(i, j, inv)
    
    print(f"   H|0⟩ ≈ {qubit}")
    print(f"   Bloch向量: {qubit.get_bloch_vector()}")
    
    # 测试Pauli X门
    print("\n2. Pauli X门")
    qubit = MajoranaQubit()
    
    for i, j, inv in BraidingGates.pauli_x_sequence():
        qubit.braid(i, j, inv)
    
    print(f"   X|0⟩ = {qubit}")
    print(f"   Bloch向量: {qubit.get_bloch_vector()}")
    
    # 测试Pauli Z门
    print("\n3. Pauli Z门")
    qubit = MajoranaQubit()
    qubit.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩态
    print(f"   初始态|+⟩: {qubit}")
    
    for i, j, inv in BraidingGates.pauli_z_sequence():
        qubit.braid(i, j, inv)
    
    print(f"   Z|+⟩ = {qubit}")
    print(f"   Bloch向量: {qubit.get_bloch_vector()}")
    
    # 测试S门
    print("\n4. 相位门 S")
    qubit = MajoranaQubit()
    qubit.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    for i, j, inv in BraidingGates.phase_s_sequence():
        qubit.braid(i, j, inv)
    
    print(f"   S|+⟩ = {qubit}")
    print(f"   Bloch向量: {qubit.get_bloch_vector()}")


def demo_braiding_trajectory():
    """演示辫子操作在Bloch球上的轨迹"""
    print("\n" + "=" * 60)
    print("Bloch球轨迹可视化")
    print("=" * 60)
    
    qubit = MajoranaQubit()
    states = [qubit.get_bloch_vector()]
    labels = ['|0⟩']
    
    # 执行一系列辫子操作
    operations = [
        (2, 3, False, "σ₂₃"),
        (1, 2, False, "σ₁₂"),
        (2, 3, False, "σ₂₃"),
        (1, 2, False, "σ₁₂"),
    ]
    
    for i, j, inv, op_label in operations:
        qubit.braid(i, j, inv)
        states.append(qubit.get_bloch_vector())
        labels.append(op_label)
    
    print(f"\n最终态: {qubit}")
    
    # 可视化
    fig = visualize_bloch_sphere(states, labels)
    plt.savefig('/home/asice-cloud/projects/pyyy/quantumsss/sim/bloch_trajectory.png', dpi=150)
    print("\n✓ Bloch球轨迹已保存到 sim/bloch_trajectory.png")


def demo_quantum_interference():
    """演示编织操作的量子干涉效应"""
    print("\n" + "=" * 60)
    print("辫子群关系验证：σ₁σ₂σ₁ = σ₂σ₁σ₂")
    print("=" * 60)
    
    # 路径1: σ₁σ₂σ₁
    qubit1 = MajoranaQubit()
    qubit1.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    qubit1.braid(1, 2, False)  # σ₁
    qubit1.braid(2, 3, False)  # σ₂
    qubit1.braid(1, 2, False)  # σ₁
    
    # 路径2: σ₂σ₁σ₂
    qubit2 = MajoranaQubit()
    qubit2.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    qubit2.braid(2, 3, False)  # σ₂
    qubit2.braid(1, 2, False)  # σ₁
    qubit2.braid(2, 3, False)  # σ₂
    
    print(f"\n路径1 (σ₁σ₂σ₁): {qubit1}")
    print(f"路径2 (σ₂σ₁σ₂): {qubit2}")
    
    # 计算差异
    fidelity = np.abs(np.vdot(qubit1.state, qubit2.state))**2
    print(f"\n保真度: {fidelity:.6f}")
    
    if fidelity > 0.9999:
        print("✓ 辫子关系验证成功！")
    else:
        print("✗ 数值误差较大")


def demo_fault_tolerance():
    """演示拓扑保护下的容错特性"""
    print("\n" + "=" * 60)
    print("拓扑保护演示：局域噪声的影响")
    print("=" * 60)
    
    num_trials = 1000
    noise_strengths = np.linspace(0, 0.1, 10)
    fidelities = []
    
    for noise in noise_strengths:
        fidelity_sum = 0
        
        for _ in range(num_trials):
            # 理想辫子操作
            qubit_ideal = MajoranaQubit()
            qubit_ideal.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
            qubit_ideal.braid(2, 3, False)
            
            # 含噪声的辫子操作
            qubit_noisy = MajoranaQubit()
            qubit_noisy.state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
            
            # 添加局域噪声（随机相位）
            local_noise = np.exp(1j * noise * np.random.randn())
            qubit_noisy.state[0] *= local_noise
            
            qubit_noisy.braid(2, 3, False)
            
            # 计算保真度
            f = np.abs(np.vdot(qubit_ideal.state, qubit_noisy.state))**2
            fidelity_sum += f
        
        avg_fidelity = fidelity_sum / num_trials
        fidelities.append(avg_fidelity)
        print(f"噪声强度 {noise:.3f}: 平均保真度 {avg_fidelity:.6f}")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(noise_strengths, fidelities, 'o-', linewidth=2)
    plt.xlabel('局域噪声强度', fontsize=12)
    plt.ylabel('平均保真度', fontsize=12)
    plt.title('Majorana编织操作的抗噪声性能', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/asice-cloud/projects/pyyy/quantumsss/sim/noise_tolerance.png', dpi=150)
    print("\n✓ 噪声容忍曲线已保存到 sim/noise_tolerance.png")


if __name__ == "__main__":
    # 运行所有演示
    demo_single_qubit_gates()
    demo_braiding_trajectory()
    demo_quantum_interference()
    demo_fault_tolerance()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
