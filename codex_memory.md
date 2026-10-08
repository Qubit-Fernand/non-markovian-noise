 下一步我认为真机测量那两个电路 GHZ 和 |0> 态作用量子门的 终态输出 观测 <X></x><X></x><X></x><X 期望值
然后把真机上的 schedule 多少 ns 输入到 数值中，对每种方案还有阶数的 HEOM 做参数搜索，看看哪个和真机契合的比较好，能够对得上。

xiaoyang 学长 propose 说：试图验证 lindblad 方程和 non-Markovian 比如 HEMO 之间差异的不可忽略性，比如真机上面相隔很多误差。

## 文献要点：Wang–Li 的非马尔可夫 PEC（2026-09-23 核对）

Ke Wang、Xiantao Li，*Non-Markovian Noise Mitigation: Practical Implementation, Error Analysis, and the Role of Environment Spectral Properties*，[arXiv:2501.05019v4](https://arxiv.org/html/2501.05019v4)。

- **需要预先给定噪声信息**：以浴相关函数（bath correlation functions, BCFs）为输入，由此构造时间依赖的噪声生成元，再计算 PEC 的准概率系数；Sec. 5 明确说明 BCF 是输入。
- **没有解决真机噪声学习**：Sec. 4 先指定谱，再用七极点近似表示相关函数。这里拟合的是已给定的数学函数，不是从未知真机测量中辨识环境。验证是一、两比特 spin-boson 数值实验。
- **保留近似条件**：采用弱耦合展开的 time-local 主方程，并分析近似误差和采样开销；不能直接视为无扰动截断的 HEOM 或任意强记忆环境的精确算法。
- **不是直接验证 1/f 噪声**：数值例子的 Eq. (64) 使用带指数截止的谱 J(ω)=ω³ exp(-ω/ωc)/ωc²，与本项目目标 1/f 谱不同。
- **对本项目的作用**：提供“已知浴相关函数 → PEC”的方法参考；真机噪声辨识、参数可辨识性以及学习误差对预测/缓解的影响，仍需另外研究。它不能直接补上当前模拟器上游的噪声学习环节。

## 用户原始问题（保留）

我感觉最大的困难，及时不做 mitigation，如果我们要核对一下 真机运行结果和我们的建模输出结果，我们的建模就需要按照真机的噪声模型。但是非 马尔可夫噪声没有原本那种重复层观察 decay 的方案那么好学。

可以看看 mitigation 的文章是怎么做的，PEC 需要噪声模型吧：https://arxiv.org/pdf/2501.05019

查到一篇 Xiantao Li 老师组的 非马尔可夫 噪声用于 误差缓解的文章
