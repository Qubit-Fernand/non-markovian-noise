# Pauli 噪声的模型层级：单快照 pseudo-Lindblad 与真记忆（2026-10-22）

**一句话结论：arXiv:2602.13145 的 appendix A 去掉 $\lambda\ge0$，得到的是"这个快照通道不存在全非负速率的 Markov 生成元"这一判据，不是"有记忆"的模型。二者相差整整一个层级；把前者当成记忆模型会在本项目里造成错误归因。**

本文是一次文献精读记录，用于给 SPP/SPL 那条线做术语与公式对齐。文中公式按本项目记号重推；论文原文的符号方向、分支约定见文末"未确认与边界"。本文**没有**复算该论文的任何数值或真机数据。

## 1. 来源，以及它实际做了什么

[Seif, Malekakhlagh, Majumder, Govia, *Single snapshot non-Markovianity of Pauli channels*, arXiv:2602.13145v2](https://arxiv.org/abs/2602.13145)（IBM Quantum）。相关定位：

- Ref. [24] = van den Berg, Minev, Kandala, Temme, Nat. Phys. 19, 1116 (2023)，即本仓库对照的 sparse Pauli–Lindblad（SPL）原方案。
- Ref. [29] = Groszkowski, Seif, Koch, Clerk, Quantum 7, 972 (2023)，"Pauli pseudo-Lindblad"这一叫法的来源。
- Ref. [48] = Kattemölle, Gulácsi, Burkard, *Non-Markovianity induced by Pauli twirling*, arXiv:2602.08464，结论是底层物理 Markov、twirl 之后有效通道却可能 CP 不可分。
- Ref. [47]（Zheng, Chu, Chen, Manes, Lee, Zhou, Jiang, arXiv:2601.22286）被引在"逻辑比特出现 emergent non-Markovianity"，与 SPP 那条线直接相关。

Appendix A 不是一个优化问题，是**三个 fitter**，对同一批局部保真度 $f_P$（$P\in\mathcal{P}_{\mathrm{loc}}$）分别求解：

| fitter | 目标 | 约束 | 备注 |
|---|---|---|---|
| (A3) 约束最小二乘 | $\min_{\lambda}\lVert M\lambda-\log f/2\rVert^2$ | $\lambda\ge0$ | IBM 原方案 [24] |
| (A4) 无约束最小二乘 | 同上，同上 | 无 | $\lambda$ 可负、可复 |
| (A5) CP 约束 | $\min_{p}\lVert Hp-f_{\mathrm{ext}}\rVert^2$ | $p\succeq0,\ \sum_i p_i=1$ | 必须显式构造 $4^n$ 维通道，只在 4 比特做 |

其中 $M$ 是**反对易矩阵**：

$$
M_{PQ}=\begin{cases}0,&[P,Q]=0\\[2pt] 1,&\{P,Q\}=0\end{cases}
\qquad P,Q\in\mathcal{P}_{\mathrm{loc}} .
$$

关键在于论文自己的定义（Discussion 一节明说）：

> "It does not imply temporal correlations of error across a quantum circuit."

它判定的 non-Markovianity 是 **CP-indivisibility**（不存在全非负速率的 GKSL 生成元），不是时序关联。这与本项目 README 里"这里'有记忆'指时间相关性及跨门历史的保留；代码没有将它与 CP-indivisibility 等同"的用法正好相反：**同一个词，两套定义。** 以后写文章必须显式声明用的是哪一个。

## 2. 概念分层

设 $P_j\in\{I,X,Y,Z\}^{\otimes N}$，$D=4^N$，$\mathrm{Tr}(P_jP_k)=D\delta_{jk}$。Pauli 通道与其 Pauli 转移矩阵（PTM）：

$$
\Lambda(\rho)=\sum_k p_kP_k\rho P_k,\qquad
f_j=\frac1D\mathrm{Tr}\big(P_j\Lambda(P_j)\big)=\sum_k h_{jk}p_k ,
$$

写成矢量形式 $f=Hp$，$p=Hf/D$，$H$ 为 Walsh–Hadamard 矩阵，$H^2=DI$。**CPTP 在 Pauli 语言里就是一个线性约束**，也就是 (A5) 在强制的东西：

$$
p=\frac1D Hf\succeq 0 .
$$

生成元层：Pauli 通道的生成元在 PTM 里**对角**，

$$
\mathcal{L}=\sum_k\lambda_k\big(P_k\cdot P_k-\mathbb{1}\big)
\quad\Longrightarrow\quad
\mathcal{L}P_j=-\Gamma_jP_j,\qquad
\Gamma_j=2\!\!\sum_{k:\{P_j,P_k\}=0}\!\!\lambda_k .
$$

于是 $\log f_j=-\Gamma_j$，$\lambda$ 与 $\log f$ 由 $M$ 线性相连。$\lambda\ge0$ 等价于所有 $\Gamma_j\ge0$，等价于 CP-divisible。

四层，从上到下逐步放开：

| 层级 | 条件 | 名称 | CP 可分？ | 有记忆？ |
|---|---|---|---|---|
| L0 | 时间齐次 $+$ $\lambda\ge0$ | Markov 半群 | 是 | 否 |
| L0′ | $\lambda(t)\ge0$，时间依赖 | 非齐次但 CP-divisible | 是 | 否 |
| L1 | $\lambda$ 可为负/复（单快照） | pseudo-Lindblad | **否** | **否** |
| L2 | 记忆核 / 两参数 PTM / 时序关联 | 真·非马尔可夫 | 否 | **是** |

必须钉死的两点：

1. **逐层 $\lambda$ 不同属于 L0′，不是非马尔可夫。** 只要每层 $\lambda\ge0$，动力学仍然 CP-divisible，"同一门在早/晚期有效噪声不同"这件事本身不构成记忆。
2. **L1 没有记忆。** 它仍是一个线性方程、单快照对象、没有跨时刻关联；它只是"这个通道没有 Markov 生成元"。L1→L2 才是引入记忆。

## 3. 显式公式

### 3.1 最一般的 Markov 生成元（对角只是特例）

不限于对角时，Pauli 基展开的 GKSL 写成 Kossakowski 形式：

$$
\mathcal{L}(\rho)=-i[H,\rho]+\sum_{P,Q\neq I}C_{PQ}\Big(P\rho Q-\tfrac12\{QP,\rho\}\Big),
\qquad C\succeq0 .
$$

$C$ 对角即 sparse Pauli–Lindblad（$\lambda_P=C_{PP}\ge0$）。**$C$ 的非对角项不是装饰。** 振幅阻尼 $L=\sigma_-=(X+iY)/2$ 给出

$$
C_{XY}=\frac{\gamma}{4}\begin{pmatrix}1&-i\\ i&1\end{pmatrix},
\qquad \text{eig}\{\gamma/2,\,0\}\succeq0,
$$

PSD 但非对角。这说明**对角 Pauli–Lindblad 无法精确描述 $T_1$**，只能给它 unital 的近似（$\lambda_X=\lambda_Y=\gamma/4$）。所以"考虑 $\lambda<0$"并不等于"考虑所有非 Markov 结构"；若目标是 $T_1/T_2$ 不对称，还要放开 $C$ 的非对角，而那是另一个方向（会离开 Clifford 可模拟的类别）。

### 3.2 可分性判据（最实用的一条）

两参数 PTM：

$$
\Lambda_{t,s}(\rho)=\frac1D\sum_j f_j(t,s)\,\mathrm{Tr}(P_j\rho)\,P_j,
\qquad f_I\equiv1,\qquad Hf(t,s)\succeq0\ \ \forall t\ge s .
$$

**$\Lambda$ 可分（即 Markov / CP-divisible）当且仅当**

$$
f_j(t,s)=\exp\!\Big(-2\!\int_s^t\!\sum_k M_{jk}\,\lambda_k(u)\,du\Big),
\qquad \lambda_k\ge0 ,
$$

也就是：**$\log f$ 线性于 $t$，且斜率在 $M$ 的像里、分量为正。** 打破其中任一条即非马尔可夫。这条判据的好处是只用多深度数据就能检验，不必构造 $4^N$ 通道：

- 若 $\log f_j(t)$ 对 $t$ 明显不线性 → L2，真记忆；
- 若线性但反推的 $\lambda$ 有负分量 → L1，单快照 CP 不可分（可以只是 twirling 或几何效应，见 [48]）。

论文用的正是这条判据在 $t=1$ 的**单点版本**。

### 3.3 L2 的三种显式参数化

**(a) 多指数（最一般、最好拟合）：**

$$
f_j(t)=\sum_\alpha c_{j\alpha}e^{-z_{j\alpha}t},
\qquad \sum_\alpha c_{j\alpha}=1,\ \ \mathrm{Re}\,z_{j\alpha}\ge0 .
$$

复 $z$ 给振荡（信息回流），负 $c$ 给非单调回弹。约束 $Hf(t)\succeq0$ 逐 $t$ 检查，或只在被使用的整数时刻检查。它的物理来源是记忆核极点的谱分解。

**(b) Nakajima–Zwanzig 记忆核：** Pauli 对角情形

$$
\dot f_j(t)=-\Gamma_j\int_0^t k(t-s)f_j(s)\,ds
\quad\Longrightarrow\quad
\tilde f_j(s)=\frac{1}{s+\Gamma_j\tilde k(s)} .
$$

取 $k(\tau)=\gamma e^{-\gamma\tau}$，极点 $s=\big[-\gamma\pm\sqrt{\gamma^2-4\Gamma_j\gamma}\big]/2$；当 $\Gamma_j>\gamma/4$ 为复极点：

$$
f_j(t)=e^{-\gamma t/2}\Big[\cos(\Omega_jt)+\frac{\gamma}{2\Omega_j}\sin(\Omega_jt)\Big],
\qquad \Omega_j=\sqrt{\Gamma_j\gamma-\gamma^2/4} .
$$

这就是 Shabani–Lidar 型 post-Markovian 主方程。瞬时速率 $\Gamma_j(t)\equiv-\partial_t\log f_j(t)$ 可以变负，是货真价实的回流；单参数 $\gamma\to\infty$ 连续退化回 L0，适合做嵌套模型比较。

**(c) 时序关联 Pauli（经典记忆，最便宜的 L2）：**

$$
\Pr(e_0,\ldots,e_L)=\pi(e_0)\prod_{\ell}T_\ell(e_{\ell+1}\mid e_\ell),
\qquad e_\ell\in\{I,X,Y,Z\}^{\otimes N}.
$$

仍可采样、仍可 Clifford 模拟，但已不是任何单个 $\mathcal{L}$ 能表达。这与 SPP 文档里的 HMM 是同一类对象；注意它**不等同于**任意量子环境。

## 4. 三个容易混淆的地方

1. **单快照 $\lambda<0$ 不等于记忆。** 它只说明该有效通道没有 Markov 生成元。而且 $\log f$ 有分支歧义（$E^{1/n}$ 取哪支），论文用主值分支并只讨论 $t=1$ 及其整数次幂来回避。要证非马尔可夫，需要多个 $t$ 上的 $f_j(t)$，看 log-线性是否破坏。
2. **逐层拟合不同不等于非马尔可夫。** 见 §2 第 1 点。
3. **twirling 会让 Markov 底层看起来非马尔可夫。** 这是 [48] 的结论，也是本项目 proposal 里"twirling 会改变被研究的过程，不能假定原来 $\Delta X$ 信号在 twirling 后不变"那条的同一件事。

## 5. 与本项目的关系

**与 SPP 的接口。** SPP 属于本文的 L2，SPL 属于 L0/L1。中间缺一个既保留时间相关结构、又 Clifford 可模拟、又能写出解析约束的对象；§3.2 的两参数 PTM 加 §3.3(a) 的多指数 $f_j(t)$ 就是这个缺口最自然的填充。建议在 SPP 的学习结果上先投影到"对角 PTM + 多指数"这一族，再比较残差，而不是直接跳到完整联合表。

**与配对脉冲的关系。** 配对脉冲测的是 $\Delta X$，其非零只能排除"理想瞬时脉冲下的静态失谐"等基线，不能排除所有 Lindblad 模型，也不自动证明 CP 不可分——这条边界原 proposal 已写明；本文给出它在 Pauli 语言里的对应物：$\Delta X\ne0$ 意味着 $f_j(t,s)$ 不是乘积形式，也就是 L2。

**对噪声学习的启示。** 现有深度点（$d=0,2,4,8$）其实已经足够做 log-线性检验，成本远低于 CP 约束拟合。建议把它作为 SPP 路线的第一步诊断：先用 $f_j(t)$ 的形状把 L1 与 L2 分开，再决定要不要上张量网络。

## 6. 未确认与边界

- 未复算论文的数值与真机结果，只核对了 appendix A 的形式与其 Markovianity 定义。
- 按 (5) 式的生成元约定重推，通道满足 $f_j=\exp(-2(M\lambda)_j)$，即严格写应为 $M\lambda=-\tfrac12\log f$；论文 (A1) 印的是 $+\tfrac12\log f$。不影响其结论，但复现时别抄错符号方向。
- 对角 PTM 是一个**建模选择**（等价于假设每个时刻的噪声仍是 Pauli 通道），不是任意非马尔可夫过程的通有表示。离开对角结构就会失去 Clifford 可模拟性。
- §3.3(b) 的闭式解依赖于所取核的形式；核不同则 $f_j$ 的函数形式不同，不能把"双指数"当成物理结论。
- 论文的 (A5) 在 $4^N$ 上优化，$N=12$ 时不可行；本项目若要用 CP 约束，只能在小系统或分解成局域块后做。
