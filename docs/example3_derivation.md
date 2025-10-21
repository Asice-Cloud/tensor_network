# Example 3 — 量子电路的数学推导

# 1. 门的代数表示

- 单比特 Hadamard 门 `H`：

$$
H=\frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\[4pt]1 & -1\end{pmatrix},
$$

作用为
\[
H|0\rangle=\frac{1}{\sqrt2}(|0\rangle+|1\rangle),\qquad
H|1\rangle=\frac{1}{\sqrt2}(|0\rangle-|1\rangle).
\]

- 两比特 CNOT（control = 第一个比特，target = 第二个比特），对计算基的作用为

注意这里的加法是在2阶循环群下的， 即 1+1 = 0。
$$
\mathrm{CNOT}\,|a,b\rangle = |a,\,a\oplus b\rangle,\qquad a,b\in\{0,1\},
$$
在基序 
$$
\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}
$$


下的矩阵表示为
$$
\mathrm{CNOT}=
\begin{pmatrix}
1&0&0&0\\[2pt]
0&1&0&0\\[2pt]
0&0&0&1\\[2pt]
0&0&1&0
\end{pmatrix}.
$$

也可写成投影形式：
\[
\mathrm{CNOT}=|0\rangle\langle0|\otimes I + |1\rangle\langle1|\otimes X,
\]
其中 
$$
X=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$


整个电路的算符（先在第 1 比特施加 H，再施加 CNOT）：
$$
U = \mathrm{CNOT}\cdot (H\otimes I).
$$

---

## 2. 逐步代数作用（两个关键输入态）

### 情形 A：输入 

$$
|00\rangle
$$



先施加 
$$
H\otimes I
$$


------->
\[
(H\otimes I)\,|00\rangle = (H|0\rangle)\otimes|0\rangle
=\frac{1}{\sqrt2}(|0\rangle+|1\rangle)\otimes|0\rangle
=\frac{1}{\sqrt2}(|00\rangle+|10\rangle).
\]

再施加 CNOT（
$$
|00\rangle\mapsto|00\rangle,\;|10\rangle\mapsto|11\rangle
$$


） ---------->
\[
U|00\rangle = \frac{1}{\sqrt2}(|00\rangle+|11\rangle).
\]

这是标准的 Bell 态（记作 
$$
|\Phi^+\rangle
$$
），为最大纠缠态。

### 情形 B：输入

$$
|11\rangle
$$



先施加 
$$
H\otimes I
$$

\[
(H\otimes I)\,|11\rangle = (H|1\rangle)\otimes|1\rangle
=\frac{1}{\sqrt2}(|0\rangle-|1\rangle)\otimes|1\rangle
=\frac{1}{\sqrt2}(|01\rangle-|11\rangle).
\]

再施加 CNOT（
$$
|01\rangle\mapsto|01\rangle,\;|11\rangle\mapsto|10\rangle
$$


）：
\[
U|11\rangle = \frac{1}{\sqrt2}(|01\rangle-|10\rangle).
\]

该态同样为最大纠缠的成员（Bell 态家族的一员）。

---

## 3. 为什么产生纠缠（Schmidt 分解与还原密度矩阵）

- 对 
  $$
  |\Phi^+\rangle = \frac{1}{\sqrt2}(|00\rangle+|11\rangle)
  $$
  

  ，其 Schmidt 分解即为：
\[
|\Phi^+\rangle = \frac{1}{\sqrt2}\,|0\rangle_A\otimes|0\rangle_B + \frac{1}{\sqrt2}\,|1\rangle_A\otimes|1\rangle_B.
\]
兩個非零的 Schmidt 系數均為 $1/\sqrt2$，因此為最大纠缠。

- 还原密度矩阵（对 B 做部分迹）：
\[
\rho = |\Phi^+\rangle\langle\Phi^+|,\qquad
\rho_A = \operatorname{Tr}_B(\rho) = \frac{1}{2}|0\rangle\langle0| + \frac{1}{2}|1\rangle\langle1| = \frac{1}{2}I_2.
\]

$$
\rho_A\
$$
 为完全混合态，说明子系统 A 的态没有纯态信息，因此系统为最大纠缠。

---

## 4. 直观机制（控制-翻转如何把局部叠加转为纠缠）

若控制比特处于叠加 
$$
\alpha|0\rangle+\beta|1\rangle
$$


，目标比特为 
$$
|b\rangle
$$
，则：
\[
\mathrm{CNOT}\big[(\alpha|0\rangle+\beta|1\rangle)\otimes|b\rangle\big]
=\alpha|0,b\rangle + \beta|1,b\oplus 1\rangle.
\]

该线性组合通常不能写成簡單的張量積，因此產生纠缠。

在本例，H 把控制比特变为等幅叠加，CNOT 根据控制位条件把两个分量映射到不同目标基，从而生成纠缠态。

---

## 5. 张量网络视角（门→张量，连线→指标收缩）

- 每个量子门可以视为局部张量：
  - 单比特门 $H$：秩-2 张量 $H^{i'}_{i}$（输入索引 $i$，输出索引 $i'$）。
  - CNOT：秩-4 张量 $U^{i'j'}_{ij}$（输入索引 $i,j$，输出索引 $i',j'$），其分量为
  \[
  U^{i'j'}_{ij}=\delta_{i',i}\,\delta_{j',\,i\oplus j}.
  \]
- 电路按时间顺序把这些张量连接（输出索引与下一门的输入索引相连），这些连线对应对相应指标求和（收缩），最终将初始态张量与电路张量收缩得到输出向量。
- 产生纠缠即对应于输出张量无法分解为两个独立外露腿的张量积。

---

## 6. 结论要点

- 代数上直接计算得到：
\[
\mathrm{CNOT}(H\otimes I)|00\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},\qquad
\mathrm{CNOT}(H\otimes I)|11\rangle=\frac{|01\rangle-|10\rangle}{\sqrt2}.
\]
- Hadamard 在控制比特上产生叠加；CNOT 将该叠加的不同分量条件性地映射到目标，从而生成纠缠。
- 从张量网络视角，门是局部张量、连线是收缩，电路就是张量网络的具体实例；纠缠对应输出张量不可分解。

