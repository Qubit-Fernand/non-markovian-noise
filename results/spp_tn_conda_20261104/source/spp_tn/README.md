# SPP tensor networks：从 Hermite HEOM 到时间 MPS

这是当前仓库内独立的研究代码包。以相邻 Overleaf 项目的 `note.tex` 为公式和约定依据，以 [HTML 方案](../docs/heom-spp-proposal-20261029.html) 为算法路线。旧的 `nmnoise/` 模拟和两时段基准保持独立。

## 安装与入口

项目统一使用独立 conda 环境 **TensorCircuitNG**，不再依赖 Qubit 或项目 venv。在仓库根目录首次创建：

```bash
conda env create -f environment.yml
conda activate TensorCircuitNG
python -m pytest -q
python -m spp_tn.demo --output results/my_spp_run --convergence
```

输出目录必须不存在。也可使用安装后的 `spp-tn --output ...`。`summary.json` 保存模型参数、版本、源码哈希、构造成本、概率检查、PEC 残差、观测量和误差来源；`source/` 保存运行时源码；NPZ 保存 MPS 节点、有限 shots 的错误／修正历史和测量结果。发现失败时保留未完成目录供检查，只有存在 `summary.json` 才表示运行完成。

本机已经创建该环境，直接 `conda activate TensorCircuitNG` 即可。无激活运行可用 `conda run -n TensorCircuitNG python ...`。精确数值依赖见 [`environments/tensorcircuit-ng.txt`](../environments/tensorcircuit-ng.txt)，环境维护与验证见 [`environments/README.md`](../environments/README.md)。旧 `TensorCircuit` 环境保留给 upstream legacy 仓库，二者分别维护。

TensorCircuit-NG 和 TensorCircuit 共用 `import tensorcircuit`，**在两个独立环境中二选一安装**，不要同时安装两个 extra。旧版环境用 `.[tn-legacy,test]`，其中固定 NumPy `<2`，避免 TensorCircuit 0.12 的 `np.reshape(newshape=...)` 导入错误。复核使用 NG 1.10.0 与 legacy 0.12.0。包的收缩、QR、SVD 使用 TensorCircuit 的 NumPy 后端；线路响应使用 `tc.Circuit`。目前没有 JAX 自动微分／JIT／GPU 支持声明。

SPP 节点表示概率本身，HEOM 节点表示线性算符系数；它们都不是量子态振幅。因此这里单独实现异维物理腿的张量链，复用 TensorCircuit 后端运算，没有直接套用 `MPSCircuit` 的量子态归一化或 Born 概率采样。

## 公式到代码

| 对象 | 实现 | 证据与范围 |
|---|---|---|
| 时间 MPS `p(a1,...,aL)` | `mps.py`, `process.py` | 求值、求和、内积、加法、逐点乘积、两扫压缩、保存／加载；不会自动按量子态范数归一化 |
| Pauli 字符／Walsh 变换 | `pauli.py` | 按真实 Pauli 对易符号构造；不是直接对标签位做普通 Hadamard |
| 经典 OU／静态 Gaussian HEOM | `heom.py:HermiteHEOM` | 系统 Liouville 节点 + 每模 Hermite 节点；局域乘积算符求和，压缩 RK4 |
| 独立传播基准 | `heom.py:DenseHermiteHEOM` | 小规模稀疏生成元 + `expm_multiply`；有内存规模保护 |
| HEOM → SPP | `process.py:build_spp` | 逐层探针、保留开放 ADO 腿、Walsh、带标签浴 MPS、时间切分、末端关闭根 ADO |
| PEA | `mitigation.py:amplify` | 整数增益的 Walsh 逐点乘积及压缩；gain=0 是全 I 历史 |
| PEC | `mitigation.py:inverse_als` | 交替局部最小二乘，KKT 强制 `Fq(0,...,0)=1`；返回优化记录，不承诺全局最优 |
| 采样与成本 | `process.py`, `mitigation.py` | 概率 MPS 条件采样；有符号 q 的 Gamma 和采样暂限可枚举的小规模 |
| 四比特 Ising | `circuits.py:ising_slots` | 四层有限脉冲配置；独立验证理想线路与目标 Trotter 乘积一致 |

### 约定必须固定

- 使用 **gate-after**：`V_j = G_j o U_j^{-1}`。第 b 个探针输入是 `U_j† P_b U_j`，输出用 `Tr(P_b rho_mu)/d` 读出。HTML 的 gate-before 图不能直接代入。
- 转移映射作用于列记忆向量：`p(a1,...,aL) = e0.T T_L(aL) ... T_1(a1) e0`。MPS 节点则按时间从左至右收缩，两种表达通过节点指标转置保持一致。
- 标签为 `I,X,Y,Z` 的 base-4 编码，第 0 个量子比特最高位；忽略相位的 Pauli 乘法是标签 XOR。每个时间节点的物理维数是 `4**n`。
- 系统算符用 `vec_F`（列优先）向量化；整个联合张量的轴序为 `(system, k1, ..., kK)`。每模保留 `k=0,...,levels-1`，不是总层级截断。
- 模型是经典实 Gaussian OU 或静态噪声，`E[xi_s(t)xi_s(0)]=exp(-gamma_s |t|)`，耦合为 `sigma_s Q_s`。它不是旧包的量子谐振子浴。`jumps` 必须包含 `sqrt(rate)`；可选 `drift` 只进入物理传播，不进入理想补偿。
- Ising 的中间 `Rz` 固定放在第 **3** 层末尾（HTML 表格），第 4 层仅为 CX。脉冲内控制与噪声共同演化；理想 twirl／修正操作不计噪声。

## 最小 Python 工作流

```python
import numpy as np
from spp_tn import Truncation, walsh
from spp_tn.heom import Mode, Segment, Slot, HermiteHEOM
from spp_tn.pauli import X, Z
from spp_tn.process import build_spp, probability_report
from spp_tn.mitigation import amplify, inverse_als

engine = HermiteHEOM(
    [Mode(operator=Z/2, sigma=0.55, gamma=0.2, levels=5)],
    max_dt=0.05, truncation=Truncation(max_bond=16, cutoff=1e-13),
)
slots = [Slot((Segment(0.3*X, 0.7),), "rx") for _ in range(3)]
p, construction = build_spp(
    engine, slots, time_truncation=Truncation(max_bond=16, cutoff=1e-12),
    assembly_truncation=Truncation(cutoff=1e-13),
)
print(probability_report(p))
p2, amplification = amplify(p, n=1, gain=2)
q, optimization = inverse_als(p, n=1, bond=4)
```

`HermiteHEOM.truncation` 控制传播内部键；`assembly_truncation` 控制带标签浴响应相加时的压缩；`time_truncation` 只控制 history／memory 切面。这三个参数不能互相替代。`cutoff` 是**每次 SVD 的绝对丢弃 Frobenius 范数**，不是相对阈值，也不是最终观测量误差。记录中的丢弃范数之和没有被解释为完整传播误差上界。

层间切分前浴侧做右正交化，使第一节点包含实际 Schmidt 权重。最终关闭根后还可以 `p.round(...)` 消除不再需要的键维数；逐层保留的可达记忆基未必是最终概率张量的最小秩表示。秩一 SVD 近似不等于边缘分布乘积；后者用 `independent_model(p)` 单独构造。

四比特流程入口：

```bash
conda run -n TensorCircuitNG python -m spp_tn.demo --example ising4 --levels 3 \
  --sigma 0.03 --gamma 0.2 --max-dt 0.05 --output results/ising4_trial
```

该 CLI 配置只含 **第 0 比特上的一个 OU 模**，不是每比特静态＋OU 的完整目标。Python API 可给每个比特添加多个 `Mode`。每层要做 `4**n * chi_previous` 次探针传播；系统腿仍为 `4**n`，并没有进一步空间局域化。该示例可能显著比默认的一比特基准慢。四比特四层有 `256**4` 个历史，CLI 不枚举其非负性或 signed-PEC 采样，也不会把归一化正确标记为概率已验证。

## 验证、限制与 TODO

已保存的[一比特三层基准与收敛表](../results/spp_tn_20261104_final/README.md)：MPS 传播对同层级稠密参考的概率 L2 误差约 `1.94e-8`，约束 ALS 逆对稠密 Walsh 逆的 L2 误差约 `1.38e-12`。理想／含噪／相关 PEC 期望分别为 `0.9610554`／`0.7947236`／`0.9610554`；这些数字是给定模型的合成结果。

[四比特四层路径核验](../results/spp_tn_ising4_mps_20261104/README.md)使用第 0 比特单 OU 模、每模三级截断。MPS RK4 (`max_dt=0.1`) 与同层级稠密参考的联合张量 L2 差约 `2.47e-5`，比较通过 TN 收缩完成，没有展开 `256^4` 个历史。两者耗时约 213 s／10 s（并行启动的实施检查，不是严格性能基准），当前原型无加速证据。完整分布非负性及多模物理收敛尚未验证。

测试覆盖无噪声极限、非对易控制下的 gate-after 探针、完整 ADO 传播、静态 Gaussian 独立积分、直接多时间 twirling、时间截断的独立稠密 SVD 对照、PEA 显式卷积、PEC 稠密逆、采样频率及负概率拒绝。40 层合成 mixture-of-products 只检查无全历史展开的表示／运算路径，不能据此宣称一般 HEOM 过程易压缩。

`--convergence` 分开扫描层级截断、RK4 步长、HEOM 内部键和时间键，并报告概率虚部、负质量、归一化。**截断后不静默修复分布。** 小负值只在已验证的采样路径上按容差处理并记录；明显负概率、虚部或不能验证非负性时拒绝采样。强截断的失败结果同样保存。

当前 PEC 局部环境通过 TN 收缩重算，每 sweep 对层数有额外二次成本；局部正规方程显式构造且有大小保护。求逆收敛记录不能证明所有 Walsh 系数非零；大规模奇异性需另行研究。残差不等于任意线路的偏差界，ridge 也不是 Gamma 的代理。PEA 的二次外推有残余偏差；有限 shots 不包含模型学习误差。

- [x] 独立 TensorCircuit-NG／legacy 环境和通用 NumPy 后端路径。
- [x] 经典 Hermite 层级 MPS、开放 ADO 响应、逐层时间 MPS。
- [x] PEA、约束 ALS-PEC、稠密校验和小规模采样。
- [x] 固定四比特 Ising 层边界与理想线路核验。
- [ ] TDVP／更适合刚性层级方程的传播器；当前是截断 RK4。
- [ ] 缓存 PEC 左右环境；改善强噪声和近奇异逆的优化稳定性。
- [ ] 可验证非负的参数化、无需全枚举的 signed-PEC 采样／Gamma 估计。
- [ ] 四比特多模式多 Trotter 步的成本、独立误差和压缩扫描。
- [ ] 有限 shots 的通用 SPP 学习、可辨识性及留出控制验证。
- [ ] 有噪声 twirling／correction 的物理验证、GPU／自动微分；当前没有真机结果。

## 维护记录

- 2026-11-04：新增独立包。以 `note.tex` 修正后的 gate-after 约定为准，HTML 的 TDVP 改为明确标识的 RK4 原型；优先验证计算闭环和误差来源。
- 2026-11-04：收敛检查发现切分前丢失 Schmidt 权重的问题，改为仅右正交化后切分，并添加独立稠密截断回归测试。
- 2026-11-04：legacy 0.12 与 NumPy 2.4 导入不兼容，legacy extra 限制 NumPy `<2`，与 NG 分环境测试。
- 2026-11-04：NG 1.10.0 + NumPy 2.4.6、legacy 0.12.0 + NumPy 1.26.4 的完整本地测试均为 **36 passed**；其中 17 项为新增 TN 测试，其余是原模拟测试。CI 配置已添加，尚未远端运行。
- 2026-11-04：完成一比特 PEA／PEC 和分项收敛实验、四比特单模完整构造及 TN 差范数对照。结果保留源码快照与哈希；初始调试结果已移入回收站，正式结果链接见上文。
- 2026-11-04：本机新环境中的 editable `.pth` 文件带 macOS `hidden` 标志，被 Python 忽略，导致独立 `spp-tn` 命令找不到包；仅对该环境的 `__editable__.nonmarkovian_gate_demo-0.1.0.pth` 执行 `chflags nohidden` 后命令入口验证通过。项目根目录的 `python -m spp_tn.demo` 不依赖该入口。

上游：[TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng)、[安装文档](https://tensorcircuit-ng.readthedocs.io/en/stable/quickstart.html)。
