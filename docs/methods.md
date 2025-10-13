# Methods and Numerical Implementation

This document explains the physical model, analytical background, numerical method, validation checks, and implementation details used by the simulation scripts in this repository (`sim/particle_in_well.py`, `sim/validate_sim.py`). It is written to be self-contained and to help reproduce or extend the work.

## Physical model

We simulate a single non-relativistic electron confined in a one-dimensional infinite square well (infinite potential well) defined on x in [0, a]. The Hamiltonian is

$$
H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x),
$$
# Methods and Numerical Implementation (方法与原理)

下面把代码中“自由粒子（free particle）在区间 [0, a] 上的谱方法”的数学原理、数值实现与验证细节写清楚，便于复现与扩展。

## 1. 物理模型与目标

我们模拟一维自由粒子（非相对论），在区间 \([0,a]\) 上演化。哈密顿量为

$$
H = -\frac{\hbar^2}{2m}\frac{d^2}{dx^2},
$$

为简洁起见，代码中默认单位取 \(\hbar=1\)、\(m=1\)。边界条件在当前实现中采用周期边界（periodic BC）：\(\psi(0)=\psi(a)\)，这使得 Fourier 谱方法（FFT）成为自然选择。

## 2. 频谱（傅里叶）传播的数学原理

设初始波函数在均匀网格上采样为 \(\psi(x_j,0)\)，取其离散傅里叶变换为 \(\tilde\psi(k,0)\)。自由粒子的每个波数分量在时间上独立演化：

$$
	ilde\psi(k,t) = \tilde\psi(k,0)\,e^{-i E_k t/\hbar},
$$

其中动能（能量）为

$$
E_k = \frac{\hbar^2 k^2}{2m}.
$$

数值上，采样点数为 \(N_x\)，网格间距为 \(dx=a/N_x\)，傅里叶对应的离散波数为

$$
k_n = \frac{2\pi}{a} n,\quad n= -\frac{N_x}{2},\dots,\frac{N_x}{2}-1
$$

（在 numpy 中可用 `np.fft.fftfreq(Nx, d=dx)` 获取排序对应的 `k`）。具体离散流程：

1. 计算 \(\psi(x,0)\) 的 FFT： \(\tilde\psi(k,0)=\mathrm{FFT}(\psi(x,0))\)。
2. 对每个时间 \(t\) 计算相因子： \(\exp(-i E_k t/\hbar)\)。
3. 得到频域波函数 \(\tilde\psi(k,t)\)，再用逆 FFT 得到 \(\psi(x,t)\)。

这个方法对光滑且周期延拓的函数能达到谱精度，且计算复杂度为 \(O(N_x \log N_x)\)。

## 3. 离散化与归一化约定（重要）

- 网格：代码中使用 `x = np.linspace(0.0, a, Nx, endpoint=False)`，即在 \([0,a)\) 上均匀采样，使 FFT 与周期延拓一致。
- 空间积分采用 Riemann-sum 近似（与 FFT 的归一化风格一致）：

  $$
  \int_0^a f(x)\,dx \approx dx \sum_{j=0}^{N_x-1} f(x_j),\quad dx=\frac{a}{N_x}.
  $$

- Parseval 定理的离散形式（与 numpy.fft 的无缩放变换相容）为：

  $$
  dx \sum_x |\psi(x)|^2 = \frac{dx}{N_x} \sum_k |\tilde\psi(k)|^2.
  $$

  因此在计算概率（归一化）、能量期望等量时，我们使用 `dx * np.sum(|psi|**2)` 和 `(dx/Nx) * np.sum(|psi_k|**2)` 作为对应离散表达。

## 4. 数值实现要点（代码内的对应关系）

- 初始条件：`initial_psi(x, a, choice)` 提供若干 preset（`gauss`, `piecewise`, `2x`, `eigen1`/`eigen2`（平面波）等）。注意一些 preset（如 `2x`、piecewise）在周期延拓下可能在边界不连续，需提高 `Nx` 来缓和 Gibbs 效果。
- 传播器：`psi_at_times_fft(psi0, x, times, hbar=1.0, m=1.0)` 实现：
  - 计算 `psi_k0 = fft(psi0)`。
  - 计算波数数组 `k = 2*pi*np.fft.fftfreq(Nx, d=dx)`。
  - 计算 `Ek = (hbar * k**2) / (2m) / hbar = (hbar*(k**2))/(2m)`（代码中直接用 `Ek_over_hbar = hbar*(k**2)/(2*m)`，随后用 `exp(-1j * Ek_over_hbar * t)`)。
  - 对每个 t 做频域相位乘法后做 ifft。

- 输出与绘图：代码以 `|psi(x,t)|^2` 作图，可保存静态 PNG 或动画（GIF/MP4）。动画保存默认优先 MP4（ffmpeg），GIF 为可选备选。

## 5. 验证方法（`sim/validate_sim.py` 中实现的检查）

为确保实现正确并对常见误差敏感，我们在仓库中提供了多种自动化检查：

1. 概率守恒：检查 `dx * sum_x |psi(x,t)|^2` 在若干时刻保持不变（与数值误差一致）。
2. Parseval 检查：验证 `dx * sum_x |psi(x)|^2` 与 `(dx/Nx) * sum_k |psi_k|^2` 的一致性（应在数值精度范围内）。
3. 平面波驻定性：若初始为单一平面波（`eigen1` 等），则其密度 `|psi|^2` 应在时间上不变。
4. 超位置频率检验：对由两项平面波组成的超位置（例如 n=1 与 n=2），相对相位随时间线性变化，拟合相位随时间的斜率给出理论频率 \((E_2-E_1)/(2\pi)\)（脚本使用相位解卷绕并做线性拟合，此法较直接对密度取 FFT 更稳健）。
5. 能量期望：谱表示下的能量期望按离散 Parseval 写为 \(E= (dx/N_x) \sum_k |\tilde\psi(k)|^2 \cdot (k^2/(2m))\)，脚本会报告该量。

## 6. 参数建议与陷阱

- 对于光滑初始条件：`Nx=600` 通常足够可视化结果；对尖锐或分段函数，建议 `Nx>=1200` 或更高以减少 Gibbs 振荡。
- 周期边界：当前实现为周期边界。如果问题要求 Dirichlet（无限深势阱）或开边界，请说明，我可以把代码改为实空间 Crank–Nicolson 或者用正弦基展开回到无限深阱实现。
- 归一化一致性：当把结果导出给其他工具（例如 Matlab）时，务必使用相同的积分近似（dx * sum）与 FFT 规范（dx/Nx 因子）以避免尺度偏差。

## 7. 使用示例（命令行）

快速示例（烟雾/快速模式，生成小动画）：

```bash
SMOKE=true ./scripts/run_animations.sh
```

直接运行脚本生成指定初始态的静态图或动画：

```bash
# 静态：高斯初态在若些时间点
/home/asice-cloud/projects/pyyy/quantumsss/.venv/bin/python sim/particle_in_well.py --init gauss --a 1.0 --Nx 800 --times 0 0.01 0.05 --out sim/free_gauss.png

# 动画：在 [0,0.1] 上生成 200 帧 GIF
/home/asice-cloud/projects/pyyy/quantumsss/.venv/bin/python sim/particle_in_well.py --init gauss --a 1.0 --Nx 800 --frames 200 --tstart 0 --tend 0.1 --animate --out sim/free_gauss.gif
```

## 8. 拓展方向（建议）

- 非周期边界或任意势：实现 Crank–Nicolson、split-step 或有限差分 + 数值对角化（用于获得非周期域的本征态）。
- 吸收边界 / 开系统：添加复势或 Lindblad 算子，需从解析模态演化转为时间步进求解。
- 单元测试：为关键属性添加 pytest（概率守恒、Parseval、平面波驻定性、超位置频率），便于 CI 验证。

### 傅里叶规范与转换（Fourier conventions and conversions）
为避免混淆，这里把三种常见的傅里叶变换规范并列，并给出它们之间的代数关系与数值实现（numpy.fft）的对应方案。

- 对称连续规范（推荐在文档中作为主要规范）:

  $$
  \phi_{\mathrm{sym}}(k) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \psi(x) e^{-i k x}\,dx,
  \qquad
  \psi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} \phi_{\mathrm{sym}}(k) e^{i k x}\,dk.
  $$

  该规范优点是 Parseval 关系形式对称：

  $$\int |\psi|^2 dx = \int |\phi_{\mathrm{sym}}|^2 dk.$$ 

- 非对称连续规范（文中部分段落早期使用的形式）:

  $$
  \phi_{\mathrm{asym}}(k) = \int_{-\infty}^{\infty} \psi(x) e^{-i k x}\,dx,
  \qquad
  \psi(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \phi_{\mathrm{asym}}(k) e^{i k x}\,dk.
  $$

  对应的 Parseval 为

  $$\int |\psi|^2 dx = \frac{1}{2\pi} \int |\phi_{\mathrm{asym}}|^2 dk.$$ 

- 离散 DFT（numpy.fft）:

  设在区间 \([0,a)\) 上等距采样 \(x_j=j\,dx,\; j=0..N-1\)，前向 DFT 与逆 DFT 为

  $$
  	ilde\psi_n = \sum_{j=0}^{N-1} \psi_j e^{-i 2\pi j n / N},\qquad
  \psi_j = \frac{1}{N} \sum_{n=0}^{N-1} \tilde\psi_n e^{i 2\pi j n / N}.
  $$

  离散 Parseval:

  $$\sum_j |\psi_j|^2 = \frac{1}{N} \sum_n |\tilde\psi_n|^2.$$

转换关系（从离散系数到对称连续谱样本）:

  - 设 \(k_n = 2\pi n / a\) 且 \(dx=a/N\)。通过 Riemann 和采样近似，

    $$\phi_{\mathrm{asym}}(k_n) \approx dx\,\tilde\psi_n.$$ 

  - 由于 \(\phi_{\mathrm{sym}} = (1/\sqrt{2\pi})\phi_{\mathrm{asym}}\)，得到

    $$\phi_{\mathrm{sym}}(k_n) \approx \frac{dx}{\sqrt{2\pi}}\,\tilde\psi_n.$$ 

请在把连续公式与代码中的离散和相比较时使用上面的转换因子（dx、N、2π、√(2π)），以保持各处公式的一致性。若你希望，我可以把这些转换写成一个小函数（例如 `tilde_to_phi_sym(tilde_psi, dx)`）并把它加入 `sim/` 以便直接计算连续谱样本与能量分布。

## 附录：数学推导（Mathematical derivation）

下面给出从连续薛定谔方程到离散 FFT 谱传播方法的简要严格推导，包含 Parseval 的离散形式、能量期望的谱表达式，以及超位置频率的推导。

### 连续空间的谱解
起始于一维自由粒子的时间依赖薛定谔方程（记 i 为虚数单位）：

$$
i\hbar\frac{\partial}{\partial t}\psi(x,t) = -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}\psi(x,t).
$$

对空间做傅里叶变换（连续形式），定义

$$
\phi(k,t) = \int_{-\infty}^{\infty} \psi(x,t)\,e^{-i k x}\,dx,
\qquad
\psi(x,t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \phi(k,t)\,e^{i k x}\,dk.
$$

把方程代入频域，注意 \(\partial_x^2\) 在频域对应乘以 \(-k^2\)：

$$
i\hbar\frac{\partial}{\partial t}\phi(k,t) = \frac{\hbar^2 k^2}{2m} \phi(k,t) \equiv E_k \phi(k,t).
$$

因此频域解为简单的相因子乘积：

$$
\phi(k,t) = \phi(k,0)\,e^{-i E_k t / \hbar},\qquad E_k=\frac{\hbar^2 k^2}{2m}.
$$

回到实空间：

$$
\psi(x,t) = \frac{1}{2\pi}\int \phi(k,0)\,e^{-i E_k t/\hbar}\,e^{i k x}\,dk.
$$

这就是连续谱方法的核心：在频域对每个波数独立乘以相因子再变回实域。

### 在区间 [0, a] 上的周期离散化与 DFT 规范
当问题在有限区间且采用周期边界时（代码中我们取 \([0,a)\) 且采样点为 \(N_x\) 个，间距 \(dx=a/N_x\)），频域变为离散：

- 网格采样点：\(x_j = j\,dx,\; j=0,\dots,N_x-1\)。
- 对应的离散波数（基于周期长度 \(a\)）：
  $$k_n = \frac{2\pi}{a} n,\qquad n=-N_x/2,\dots,N_x/2-1,$$
  或用 numpy 的 `np.fft.fftfreq(Nx, d=dx)` 获得合适的排序。

采用 numpy.fft 的无归一化 DFT 定义（forward `fft` / inverse `ifft`，其中 `ifft` 内含 $1/N_x$ 因子）：

$$
	ilde\psi_n = \sum_{j=0}^{N_x-1} \psi_j\,e^{-i 2\pi j n / N_x},
\qquad
\psi_j = \frac{1}{N_x}\sum_{n=0}^{N_x-1} \tilde\psi_n\,e^{i 2\pi j n / N_x}.
$$

在该规范下，频域随时间的演化与连续情况完全一致（将连续的 \(k\) 替换为离散的 \(k_n\)）：

$$
	ilde\psi_n(t) = \tilde\psi_n(0)\,e^{-i E_{k_n} t / \hbar},\qquad E_{k_n}=\frac{\hbar^2 k_n^2}{2m}.
$$

因此数值流程就是：
1. 计算 `tilde_psi = fft(psi)`；
2. 对每个频率分量乘以 `exp(-1j * E_k * t / hbar)`；
3. 用 `ifft` 变回实域得到 `psi(x,t)`。

### Parseval 的离散形式与归一化
对 DFT 定义做直接代数可以得到离散 Parseval 关系：

从逆变换代入并用正交关系对求和，我们得到

$$
\sum_{j=0}^{N_x-1} |\psi_j|^2 = \frac{1}{N_x} \sum_{n=0}^{N_x-1} |\tilde\psi_n|^2.
$$

将左边乘以网格间距 \(dx\) 给出对应的 Riemann-sum 近似连续概率质量：

$$
dx\sum_j |\psi_j|^2 = \frac{dx}{N_x}\sum_n |\tilde\psi_n|^2.
$$

这就是我们在代码中使用的等价式（注意 numpy.fft 的 `fft`/`ifft` 规范与这里一致）。它确保用 `dx * np.sum(|psi|**2)` 与 `(dx/Nx) * np.sum(|psi_k|**2)` 得到相同的数值概率。

### 能量期望的谱表示
动能算符在频域上对角化：

$$
\langle E \rangle = \int \psi^*(x)\left(-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}\right)\psi(x)\,dx
= \frac{1}{2\pi}\int |\phi(k)|^2\frac{\hbar^2 k^2}{2m}\,dk
$$

在离散周期 DFT 下，对应的离散表达为：

$$
\langle E\rangle \approx \frac{dx}{N_x}\sum_{n=0}^{N_x-1} |\tilde\psi_n|^2\,\frac{\hbar^2 k_n^2}{2m}.
$$

在代码中我们通常先计算 `psi_k = fft(psi_x)`，然后用 `Ek = (hbar**2 * k**2) / (2*m)`，最后按上式求和得到能量期望。

### 两项平面波超位置的频率（相对相位推导）
考虑两个平面波的线性组合

$$
\psi(x,t) = c_1 e^{i k_1 x}e^{-i E_1 t/\hbar} + c_2 e^{i k_2 x}e^{-i E_2 t/\hbar}.
$$

其概率密度包含交叉项

$$
|\psi|^2 = |c_1|^2 + |c_2|^2 + 2\,\mathrm{Re}\left[c_1 c_2^* e^{i (k_1-k_2)x} e^{-i (E_1-E_2)t/\hbar}\right].
$$

因此时间上的振荡频率为

$$
f_{\text{beat}} = \frac{|E_2-E_1|}{2\pi\hbar}.
$$

在代码里若设 \(\hbar=1\)，则频率简写为 \((E_2-E_1)/(2\pi)\)。这与我们在验证脚本中通过拟合相对相位随时间变化得到的斜率直接对应。

### 数值实现注意事项（简短说明）
- 在实现中我们使用 `x = np.linspace(0, a, Nx, endpoint=False)` 保证周期对齐；`k = 2*np.pi*np.fft.fftfreq(Nx, d=dx)` 给出正确的物理波数。
- numpy 的 `fft`/`ifft` 规范决定了上面 Parseval 的系数 `(dx/Nx)`；在使用其它 FFT 库或不同变换规范时请相应调整系数。
- 谨防混淆采样定理：最大可解析波数约为 Nyquist \(k_{\max}=\pi/dx\)。如果初始态或演化产生大量高频能量，格点不足会导致混叠。
- 频率估计：对超位置分量，直接在频域跟踪每个模态的相位并做线性拟合，比对概率密度做短时 FFT 更稳健（后者受窗函数和泄露影响更大）。

---

以上附录旨在把实现细节与分析公式明确对应，帮助理解代码中的每一步选取与归一化约定。如需我把这些公式分别配上编号、图示或把证明步驟寫得更严謹（例如把 DFT 正交关系的代数推导逐步写出），我可以继续展开并直接把推导代入 `docs/methods.md`。