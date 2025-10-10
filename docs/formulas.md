# 公式与数值说明 - 无限深势阱（Particle in a box）

本文档列出用于 `sim/particle_in_well.py` 脚本的关键数学公式、初始条件定义，以及数值实现说明。

## 基本物理设定
无量纲单位，取 ℏ = 1，m = 1。粒子被限制在区间 [0, a] 的无限深势阱中：

V(x) = \begin{cases}
0, & 0 < x < a \\
\infty, & \text{otherwise}
\end{cases}

边界条件：ψ(0,t) = ψ(a,t) = 0。

## 时间相关薛定谔方程（TDSE）

在无外场和无势能（势为 0）区域内，TDSE 为：

$$
 i \hbar \frac{\partial}{\partial t} \psi(x,t) = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} \psi(x,t)
$$

在无量纲化（\(\hbar=1, m=1\)）后：

$$
 i \frac{\partial}{\partial t} \psi(x,t) = -\frac{1}{2} \frac{\partial^2}{\partial x^2} \psi(x,t)
$$

## 本征态与能量
无限深势阱的标准本征问题的归一化本征函数（正弦基）为：

$$
 \phi_n(x) = \sqrt{\frac{2}{a}} \sin\left(\frac{n\pi x}{a}\right), \quad n = 1,2,3,\dots
$$

对应的能量本征值为：

$$
 E_n = \frac{n^2 \pi^2 \hbar^2}{2 m a^2}.
$$

在无量纲化后（\(\hbar=1, m=1\)）：

$$
 E_n = \frac{n^2 \pi^2}{2 a^2}.
$$

## 初始波函数的本征态展开
任意初始波函数 \(\psi(x,0)=\psi_0(x)\)（满足边界与平方可积）可写为本征态的线性组合：

$$
 \psi_0(x) = \sum_{n=1}^\infty c_n \phi_n(x), \quad c_n = \langle \phi_n | \psi_0 \rangle = \int_0^a \phi_n^*(x) \psi_0(x) \, dx.
$$

数值上我们截断成前 N 项：

$$
 \psi_0(x) \approx \sum_{n=1}^N c_n \phi_n(x).
$$

投影系数的数值近似使用复值的数值积分（脚本中用 `numpy.trapezoid`）计算。

## 时间演化
每个本征分量的时间演化由相位因子给出：

$$
 \psi(x,t) = \sum_{n=1}^N c_n \phi_n(x) e^{-i E_n t / \hbar}.
$$

在无量纲化（\(\hbar=1\)）下为：

$$
 \psi(x,t) = \sum_{n=1}^N c_n \phi_n(x) e^{-i E_n t}.
$$

概率密度为：

$$
 P(x,t) = |\psi(x,t)|^2.
$$

概率守恒的数值检验可通过积分 \(\int_0^a P(x,t) \, dx\) 与 1 进行比较。

## 脚本中实现的初始条件（明确表达式）

1. 2x:

$$
 \psi_0(x) = 2x, \quad 0<x<a
$$

脚本会进行 L2 归一化：

$$
 \psi_0(x) \leftarrow \frac{\psi_0(x)}{\sqrt{\int_0^a |\psi_0(x)|^2 dx}}.
$$

2. piecewise:

$$
 \psi_0(x) = \begin{cases}
 x + 5, & 0 < x < \frac{a}{2} \\
 \frac{a}{2} + 5 - x, & \frac{a}{2} \le x < a
 \end{cases}
$$

同样会做数值归一化。

3. gauss (默认):

$$
 \psi_0(x) = e^{-\frac{1}{2} \left(\frac{x-x_c}{\sigma}\right)^2}, \quad x_c = \frac{a}{2}, \; \sigma = 0.08 a
$$

并归一化。

4. eigen1:

$$
 \psi_0(x) = \phi_1(x) = \sqrt{\frac{2}{a}} \sin\left(\frac{\pi x}{a}\right)
$$

5. eigen2:

$$
 \psi_0(x) = \phi_2(x) = \sqrt{\frac{2}{a}} \sin\left(\frac{2\pi x}{a}\right)
$$

6. superpos12:

$$
 \psi_0(x) = \frac{\phi_1(x) + \phi_2(x)}{\sqrt{2}}.
$$

## 数值实现注意事项
- 空间网格：脚本使用等间距网格 `x = linspace(0,a,Nx)`，并在边界处强制 ψ=0。增大 `Nx` 可以提高投影与积分的精度。
- 模式截断：使用 `N` 个本征态。对于不光滑的初始条件（如 piecewise），需要更大的 `N` 来减小截断误差。
- 积分：使用 `numpy.trapezoid`（等间距梯形法）进行数值积分。
- 单位：脚本使用无量纲单位（ℏ=1,m=1）。如果要使用真实物理单位，请在脚本中传入物理常数并相应调整时间范围。
- 动画：脚本用本征态展开按时刻生成 |ψ(x,t)|^2 的帧并用 `matplotlib.animation` 保存 GIF（PillowWriter）或在必要时回退到 MP4（FFMpegWriter）。

## 推荐参数（经验值）
- 平滑的初始条件（如高斯）：`Nx=800, N=120` 常常足够。  
- 不光滑的条件（piecewise）：建议 `N>=400` 并适当增大 `Nx`。
- 帧数/平滑动画：在时间区间内生成 40-200 帧，根据动画长度与帧率调整。

---

如果你要我把这些公式再转成 PDF、LaTeX 文件，或者在仓库 README 中引用此文档，我可以继续处理。