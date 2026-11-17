# 非马尔可夫噪声：模拟与 SPP 张量网络

本仓库研究有限时长控制下的时间相关噪声，以及 SPP 的张量网络表示和误差缓解。当前结果均为合成数值模拟，尚无真机噪声学习或硬件验证。

## 当前状态

| 模块 | 已实现 | 入口 |
|---|---|---|
| `spp_tn` | 经典 Hermite HEOM 的层级 MPS、逐层 SPP 时间 MPS、PEA 与约束 ALS-PEC；使用 TensorCircuit-NG NumPy 后端 | [说明与 API](spp_tn/README.md) |
| `heom_spp_cpu` | 独立的 HEOM → SPP CPU 包，包含 MPO/TDVP 路径和静态噪声积分参考 | [说明](heom_spp_cpu/README.md) · [已知问题](heom_spp_cpu/docs/known-issues.md) |
| `nmnoise` | 经典有色噪声的 dense／空间 MPS 轨迹、量子浴 HEOM 与显式浴对照、受限两时段 SPP 基准 | [模型与运行说明](docs/nmnoise-guide.md) |

**`spp_tn` 的当前优先事项是 MPO + TDVP，验证后再考虑 CPU 优化与 GPU 并行**，具体步骤见[待办](#待办todo)。该包的现有传播器为压缩 RK4，尚无显式 MPO、TDVP 或 GPU 路径。另一个独立包 `heom_spp_cpu` 已包含 MPO/TDVP 实现，其数值精度仍需按具体模型做收敛验证。SPP 时间 MPS 和 HEOM 内部 MPS 压缩的是不同指标，不能与旧轨迹模拟的空间 MPS 混称。

## 快速运行

在代码仓库根目录（本地项目的 `Code/`）执行：

```bash
conda env create -f environment.yml
conda activate TensorCircuitNG
python -m pytest -q
python -m spp_tn.demo --output results/my_spp_run --convergence
```

已有环境可直接激活。输出目录应使用新名字；固定依赖统一维护在 [requirements.txt](requirements.txt)，也可在干净的 Python 3.11 环境中用 `python -m pip install -r requirements.txt` 安装。

TensorCircuit-NG 与 legacy TensorCircuit 共用导入名，需分环境安装。环境配置与历史验证见 [environments/README.md](environments/README.md)。2026-11-04 两套环境各通过 36 项项目测试；这不是本次重新测试的结果。核心求解无需 LaTeX，生成论文风格图表才需要 LaTeX 和字体。

## 结果与讨论入口

- [从噪声模型到 SPP：我们准备怎么算？](docs/heom-to-spp-outline-20261119.md)：11 月 19 日讨论纲领，含完整流程图、HEOM／MPS／MPO／TDVP 分工及下一阶段任务。

- [SPP 太大了，张量网络能帮上忙吗？](docs/spp-too-big.md)：问题说明、逐层构造及本次实验结果。
- [单比特 PEA／PEC 与收敛](results/spp_tn_20261104_final/README.md)：已知模型内的缓解与误差检查。
- [四比特单模 SPP 构造对照](results/spp_tn_ising4_mps_20261104/README.md)：紧凑表示与同层级参考核对；尚未建立扩规模收益或完整历史非负性。
- [报告、方案与数据索引](docs/reading-index.md)：其余讨论稿、HTML 方案、旧基准与文献核对记录。
- [复现与备份](docs/backup-and-reproduce.md)：结果归档和复现约定。

## 文稿与文献

文稿位于同级 [Overleaf/](../Overleaf/)，使用独立 Git 远端。文献在项目 [Zotero/](../Zotero/)；逐篇主题和阅读入口见 [Zotero/README.md](../Zotero/README.md)。Zotero 群组集合为 [Error-Mitigation → Non-Markov](https://www.zotero.org/groups/6729418/collections/FH728L4C)。这些同级目录是本地项目入口，不包含在代码仓库的独立克隆中。

## 待办（TODO）

### `spp_tn` 优先事项：MPO + TDVP，验证后再优化（2026-11-15）

下一阶段先完成 HEOM 生成元的矩阵乘积算符（MPO）表示，以及基于时间依赖变分原理（TDVP）的层级 MPS 传播，再接入 HEOM → SPP。正确性与收敛通过后，才进入 CPU 优化和 GPU 并行阶段。PEA／PEC 保留为下游验证。

当前实现仍是局域乘积算符之和配合压缩 RK4，没有显式 MPO 或 TDVP 路径。保留 RK4 与小规模稀疏指数传播作为参考，不将“实现 MPO 作用”视为“已实现 TDVP”。

按以下顺序推进：

1. [ ] **MPO 表示与作用**：将 `heom.py:terms()` 的生成元构造成 MPO，固定输入／输出指标和 `vec_F` 顺序，实现 MPO–MPS 收缩；用小规模显式矩阵与原逐项实现验证作用结果。
2. [ ] **TDVP 传播**：针对 HEOM 的非幺正线性演化实现局部有效生成元、环境收缩和扫描更新；明确单站点／双站点方案、键维数增长与截断策略。层级 MPS 表示线性算符系数，不套用量子态的自动范数归一化。分别记录时间积分、局部求解与截断误差。
3. [ ] **正确性与 HEOM → SPP 验证**：保持 gate-after 探针和时间 MPS 构造约定，接入 MPO + TDVP。与同层级的稀疏指数及收敛后的 RK4 对照，检查传播状态、SPP 联合张量、边缘概率、归一化和非负性；分别扫描时间步长、HEOM 层级、内部键与时间键。先复核已有单比特和四比特单模例子，再增加模式数与线路层数。
4. [ ] **CPU 性能分析与优化（上述验证后）**：定位 MPO 环境收缩、局部求解、QR/SVD 与探针传播的耗时，评估环境复用、收缩顺序、临时数组和独立探针并行；共同控制工作数与 BLAS 线程数，每项优化重新检查数值误差。
5. [ ] **GPU 并行（上述验证后）**：核验目标设备的 complex128、QR/SVD 和局部求解支持，再迁移核心数组与运算，评估同形状探针批处理。考虑动态键维数、补零和主机／设备传输成本；当前 NumPy 路径不能仅靠切换 TensorCircuit 后端完成迁移。目标设备与后端待确定。
6. [ ] **性能验收**：在相同物理模型与目标精度下，比较参考 RK4、MPO + TDVP CPU 及可用 GPU 路径，报告键维数、运行时间和峰值内存／显存。GPU 计时须同步，分开报告编译／预热与稳定运行成本。

第一阶段完成标准是 MPO + TDVP 正确运行、HEOM → SPP 端到端通过参考对照与收敛检查。第二阶段再判断优化是否带来加速，以及在哪些规模上受益；不预设 TDVP 或 GPU 必然更快。

以下为后续研究方向。已完成配对序列的合成噪声数值扫描，尚未执行真机实验。

- [x] 实现可供对接的受限 4 比特 Ising 数值基准，完成 HEOM→SPP、末态计数学习、独立层/相关模型留出预测、L1 历史卷积求逆和 PEC 成本核验（2026-10-29；[报告](docs/spp-ising-benchmark.md)）。
- [ ] 接收合作者源码、参数、SPP 指标和优化操作集合，与本地基准逐项比较；本地自建基准不等于已验证其报告。
- [ ] 在扩展前明确多站点浴、更多时段与有限时长控制的范围；当前四历史模型无张量压缩或一般空间相关。器件拟合与硬件验证继续保留为未完成。

- [ ] 真机测量 GHZ 电路与从 `|0>` 出发的门电路终态，并测量相应的 X 乘积观测量期望值；比特数和具体测量协议仍需明确。
- [ ] 将真机 schedule 与波形输入数值模型，拟合环境/控制参数并比较硬件预测；HEOM 层级单独做数值收敛检查。
- [ ] 按 xiaoyang 学长建议，研究 Lindblad 与非马尔可夫模型差异是否可在真机上分辨；这是研究设想，尚非已证实的 non-Markovian effect。
- [ ] 在已有配对扫描基础上输入实验波形、校准控制/读出误差，完成参数不确定度下的实验选择与真机留出预测。
- [ ] 按 [SPP 新方案](docs/paired-pulse-proposal.md#合作者新方案学习时空-pauli-过程并用于预测与缓解2026-10-21)先确认多时刻对象与可用测量，完成一比特受控相关噪声正对照及可辨识性检查，再推进真机留出预测。
- [ ] 用现有 $d=0,2,4,8$ 的保真度检验时间齐次单指数基线（在保真度为正且对数误差可控时检查 $\log f_j$ 的线性），并比较准静态与有限相关时间模型的留出预测；非指数衰减本身不能证明“真记忆”或 CP 不可分性，见 [IBM 核对说明](docs/ibm-2607-25998-review.md)。
- [ ] 建立或选择噪声学习方案，研究浴相关函数/谱参数的可辨识性，以及学习误差对预测或缓解的影响。
- [ ] 选择具体器件/论文参数，统一谱归一化、温度和截止，拟合 BCF 并检查连续谱与 HEOM 层级收敛，再扩展到两比特 CR 控制。
- [ ] 旧量子浴路线的层级 MPS／TEMPO、可扩展局域观测量输出及更一般控制接口仍待实现；新 `spp_tn` 已有经典 Hermite HEOM 的层级 MPS，二者需区分。

## 维护约定与记录

- 区分已知模型数值验证、真机噪声学习和器件拟合，记录各自的完成范围。
- 每次运行保存参数、版本、源码快照及独立输出目录；分别检查时间步长、层级和张量截断误差。
- 区分时间相关性与 CP 不可分性，见[噪声模型层级](docs/pauli-noise-model-hierarchy.md)。

维护历史、已完成事项与环境迁移记录见 [maintenance-history.md](docs/maintenance-history.md)。2026-11-15 整理文档入口；未修改模拟实现或已有结果。
