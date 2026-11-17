# HEOM → SPP 误差缓解

用 HEOM 计算逐层 Pauli twirling 之后的时空 Pauli 过程（SPP），把全部错误历史的联合概率写成时间方向的 MPS，再在 SPP 上做噪声放大（PEA/ZNE）和变分求逆（PEC）。算法路线见 [docs/heom-spp-proposal-20261029.html](../docs/heom-spp-proposal-20261029.html)。

只依赖 numpy、scipy、threadpoolctl，纯 CPU。

## 结构

| 模块 | 职责 |
|---|---|
| `heom_spp/circuit.py` | 逻辑门、并行门块、学习层；层的理想幺正 $U_j$ 在这里由逻辑门定义；Qiskit 导入导出 |
| `heom_spp/native.py` | 分解到脉冲级原生门 `r`（一个 XY 方波）、`rz`（理想 Z）、`sqisw`；Ising Trotter 步的 sqiSWAP 版本 |
| `heom_spp/plot.py` | 脉冲时序图 |
| `heom_spp/pulse.py` | `Segment`（有限时长控制）、`Frame`（零时长虚拟门）、`Slot`（一个学习层）；脉冲编译器 |
| `heom_spp/bath.py` | 浴的指数项 `Mode`（经典实高斯或复指数左/右节点）、白噪声余项 `White`、`noise.json` 预设、BSD 拟合表读取 |
| `heom_spp/heom.py` | 三个共用同一方程的引擎：`TDVPHEOM`（生产）、`HEOM`（压缩 RK4）、`DenseHEOM`（稀疏指数参考） |
| `heom_spp/tdvp.py` | 单点 KSL/TDVP 与非厄米 Arnoldi |
| `heom_spp/spp.py` | 按层并行构造 SPP、概率检查、边缘分布、抽样 |
| `heom_spp/mitigation.py` | 噪声放大、稠密与 ALS 变分求逆、线路响应 |
| `heom_spp/analyze.py` | 对已保存的 SPP 做 ZNE 与 PEC 分析（不再演化） |
| `heom_spp/reference.py` | 静态噪声的积分参考解（不经过 HEOM） |
| `heom_spp/cli.py` | 命令行：`run` / `reference` / `compare` / `converge` / `summarize` |
| `heom_spp/demo.py` | 单比特端到端演示（SPP → PEA → PEC → 有限 shots） |

线路层、脉冲、浴三者互相独立：换原生门分解只改层文件，换真实脉冲只改 `pulse.py`，换噪声只改 `bath.py`，传播器和 SPP 构造不动。

## 约定

- **gate-after**：含噪层写成 $R_j = V_j O_j$，探针输入 $U_j^\dagger P_b U_j$，用 $\mathrm{Tr}(P_b\,\cdot)/d$ 读出，浴腿全部保持开放。这与 HTML 里"噪声在前"的写法不同，正式计算前需统一。
- **理想门与脉冲分离**：`Slot.unitary` 来自逻辑层。脉冲的相干误差会作为误差出现在 SPP 里，不会被吸收进 $U_j$。
- **浴默认不重置**（`--reset` 基线除外）：层边界只对系统腿读出再注入，ADO 原样穿过；只有最后一层读根 ADO。
- 时间单位 ns，Hamiltonian 系数 rad/ns；q0 是张量积最左因子；Pauli 标签按 I,X,Y,Z 四进制、q0 为最高位。

## 安装与检查

```bash
python -m pip install -e ".[test]"
```

```bash
python -m pytest -q
```

```bash
bash scripts/smoke.sh
```

需要 Python ≥ 3.10。Qiskit 只在导入导出线路时需要（`.[circuits]`）。

## 运行

```bash
python -m heom_spp.cli run --layers-json configs/html_ising4_layers.json --noise-json configs/static4_noise.json --engine tdvp --workers 24 --heom-bond 16 --time-bond 16 --assembly-bond 64 --output results/static4
```

输出目录必须不存在。里面有 `spp.npz`、`marginals.npz`、实际使用的 `layers.json` / `noise.json` / `pulse_schedule.json`、源码快照和 `summary.json`。加 `--virtual-z` 把 `rz`、`z` 当作零时长的帧变换。

比较两次运行（收敛检查时每次只改一个参数）：

```bash
python -m heom_spp.cli compare results/case_a results/case_b
```

## sqiSWAP 原生门线路

先看分解、分层和时序，再决定是否演化：

```bash
python scripts/show_native.py --output docs/native_sqisw
```

生成 `layers.txt`（层划分与每个门的角度）、`circuit.txt`（Qiskit 线路图）、`timing.png`（Ωx、Ωy、g 的方波时序）和两个 JSON。当前结果在 [docs/native_sqisw](docs/native_sqisw/)。

- 每个单比特幺正 = 一个 XY 方波 + 一个理想 Z：$R_z(a)R_x(\theta)R_z(b)=R_z(a+b)\,R_{\phi=-b}(\theta)$。
- $e^{-i\theta ZZ/2}$ 用两个 sqisw：$(k_3\otimes k_4)\,S\,(R_x(\alpha)\otimes R_y(\pi))\,S\,(k_1\otimes k_2)$，$\sin\frac{\pi-\alpha}{2}=\sqrt2\sin\frac{|\theta|}{2}$，$|\theta|\le\pi/2$。外侧 $k_i$ 数值求解，解不唯一。
- `sqisw` $=e^{-i\pi/8\,(XX+YY)}$，即 $H=\tfrac g2(XX+YY)$、$gT=\pi/4$，与 HEOM_two-qubit-gate 的符号一致（是教科书 $\sqrt{\mathrm{iSWAP}}$ 的逆）。
- 约定的时序：单比特 15 ns、sqisw 20 ns、每个脉冲块前 5 ns 空闲、Z 全部理想；命令行用 `--example ising4-sqisw --square-native`。
- 坐标系：`RotatingFrames` 记录可调比特进出交换门时的 $R_z(\mp\Delta t)$ 及其补偿，默认补偿后不改变线路；`compensate=False` 时保留为相干误差。

## 服务器（Slurm）

把整个目录复制过去即可，不需要 `.git`、`results/` 和本地虚拟环境。在仓库根目录、激活环境后提交：

```bash
sbatch scripts/spp.slurm
```

脚本默认 `-p gpu`、单节点 24 核、排除 `gpu3`、不申请 GPU，跑进程数伸缩扫描。正式计算用 `RUN_ARGS`：

```bash
RUN_ARGS="--layers-json configs/html_ising4_layers.json --noise-json configs/static4_noise.json --output results/static4" sbatch scripts/spp.slurm
```

进程数取自 `SLURM_CPUS_PER_TASK`，每个进程一个 BLAS 线程。如果分区要求必须申请 GPU，按脚本注释加一行 `--gres`。

## 收敛扫描（批量提交）

以时间键维 8、HEOM 键维 8、步长 1 ns、层级 6 为基线，每次只改一个参数，每组一个作业：

```bash
STEPS=1 bash scripts/scan_convergence.sh
```

- `STEPS=N` 是 Trotter 步数（每步 4 个学习层）；结果放在 `results/scan_stepsN/`。
- 默认阶梯：`TB_LIST="16 32"`、`HB_LIST="16 24"`、`DT_LIST="0.5 0.25"`、`LV_LIST="8 10"`，共 9 个作业；都可以用环境变量改，`SIGMA`、`CPUS` 同理。
- 重复执行会跳过已完成的组，并指出未完成的目录。不走 Slurm 时加 `LOCAL=1 WORKERS=6`。
- 脚本先在当前节点直接算出**积分参考解**（见下，单核、不提交作业）放到 `reference/`；`REFERENCE=0` 跳过。
- 有参考解后不必逐个参数改：`GRID_DT="0.1 0.05" GRID_HB="8 16 24"` 追加“步长 × HEOM 键维”网格（命名 `dt0.1-hb16`），已有的组会跳过。
- 参数选定后，`BASELINES=only HB=16 DT=0.5 bash scripts/scan_convergence.sh` 只提交 `reset-layer`、`reset-pulse` 两个独立噪声基线（见下）。

全部跑完后，一条命令给出该用哪一组：

```bash
python -m heom_spp.cli converge results/scan_steps1
```

- **有参考解时**（`reference/` 存在，或用 `--reference` 指定）：每个 HEOM 运行直接与参考解比较 $\|p_{\rm run}-p_{\rm ref}\|$，列出误差和用时，推荐误差在容差内、用时最短的一组。
- **没有参考解时**：相邻两档联合分布的 L2 距离小于容差，就取较粗的一档；扫描是逐个参数改的，推荐的组合如果没有现成的运行，需要再单独跑一次。
- 容差默认是关联信号 $\|p-\prod_j p_j\|$ 的 1%（`--relative`），也可以用 `--tolerance` 给绝对值。结果写入 `convergence.json`。

## 静态噪声的积分参考解（不经过 HEOM）

静态无序下，给定各比特的失谐 $\delta$ 后演化是纯幺正的，twirl 后各层的 Pauli 错误相互独立，所以

$$p(a_1\dots a_L)=\int d\delta\,P(\delta)\prod_j p_j(a_j|\delta),\qquad p_j(a|\delta)=\left|\mathrm{Tr}\big(P_a W_j(\delta)U_j^\dagger\big)/d\right|^2 .$$

用 Gauss–Hermite 积分，每个节点是时间 MPS 的一个乘积项，再压缩：

```bash
python -m heom_spp.cli reference --example ising4-sqisw --square-native --steps 1 --sigma 0.001 --output results/ref
```

- 没有层级、HEOM 键维和时间步长；唯一的数值参数是每个模的节点数 `--nodes`（默认 6），输出里的 `quadrature_check_l2` 是与少两个节点的结果之差。
- 输出目录与 `run` 相同（`spp.npz`、`layers.json`），可以直接给 `compare`、`converge --reference` 和 `heom_spp.analyze` 用。
- 只适用于静态经典噪声、封闭系统（无 $T_1$、无白噪声）。有限关联时间和量子浴仍然要靠 HEOM；参考解的用途是给 HEOM 的参数定标。

## 独立噪声基线：重置 ADO

```bash
python -m heom_spp.cli run ... --reset layer --output results/reset_layer
python -m heom_spp.cli run ... --reset pulse --output results/reset_pulse
```

- `--reset layer`：每个学习层从零 ADO 出发、层末读根 ADO，得到逐层独立的模型（时间键维 1）。对经典噪声它严格等于完整 SPP 的边缘乘积；对量子浴两者不同。
- `--reset pulse`：每个脉冲块（含其前的缓冲）之后也重置，即门级独立模型，层内的相干累积也被切断。
- 每层只需 $4^n$ 次探针，比完整 SPP 便宜。

## ZNE 与 PEC 分析（与构造 SPP 分开）

每次 `run` 都会保存 `spp.npz` 和 `layers.json`，分析只读这两个文件：

```bash
python -m heom_spp.analyze results/scan_steps1/base
```

- 期望值是精确收缩，不抽样：用各层理想门的 Pauli 转移矩阵与 SPP 的 Walsh 形式逐层缩并。
- **ZNE**：增益 `--gains 1,2,3`，插入的额外噪声分别取自 SPP 本身和逐层独立模型，各自外推到零噪声。
- **PEC**：独立模型的逐层逆，以及 SPP 的变分逆（时间键维 `--pec-bonds 1,2,4`），给出缓解后的偏差、残差和 Γ。SPP 逆的 Γ 用重要性抽样估计并给出标准误差。
- 观测量用 `--observables ZIII,ZZII,...`（q0 在最左），初态 `--initial zero|plus`。结果写入运行目录的 `mitigation.json`。
- `--baseline 名字=运行目录`（可重复）加入别的独立模型，例如 `--baseline reset-layer=results/reset_layer --baseline reset-pulse=results/reset_pulse`。器件始终是所分析的 SPP，基线只替换放大或求逆时用的模型。
- 每行输出带已用时间；PEC 阶段还给出剩余时间的上限估计（逆提前收敛会更早结束）。
- 表里的 `max |err|` 是各观测量 $|\langle O\rangle-\langle O\rangle_{\rm ideal}|$ 中最大的一个。`Gamma` 是修正准概率的 $\Gamma=\sum|q|$，PEC 达到同样统计误差需要的采样次数乘以 $\Gamma^2$。
- ZNE 的放大噪声与器件噪声并排缩并，不截断；只有放不下时才把插入的噪声压缩到 `--zne-bond`。

### 误差随线路长度的变化

```bash
bash scripts/scan_steps.sh                       # 每个 Trotter 步数一个 Slurm 作业
LOCAL=1 bash scripts/scan_steps.sh               # 或在当前机器上依次运行
tail -f results/steps_sig0.001_tb32/steps*.log        # 进度与剩余时间
python scripts/plot_steps.py results/steps_sig0.001_tb32   # 表格、steps.csv、steps.png
```

每个步数先用积分参考解得到 SPP，再做上面的分析。`STEPS_LIST`、`SIGMA`、`PEC_BONDS`、`TIME_BOND` 等用环境变量改；画图需要 matplotlib，服务器上没有时把 `steps*/mitigation.json` 拷回本机再画。 参考解默认压缩到时间键维 32，结果目录名带 `_tb32`；表格里的 `1-sum(p)` 是压缩丢掉的概率，它不远小于要比较的误差时应加大 `TIME_BOND`。

## 浴模型

`noise.json` 目前只覆盖经典退相位预设（静态、OU、对数等距 OU 之和）。一般的复相关函数用 Python 接口：

```python
from heom_spp.bath import White, load_exponentials, quantum_modes
from heom_spp.heom import TDVPHEOM
from heom_spp.pauli import Z, embed

w, g, k0 = load_exponentials("cfdata_qubit1.dat", max_decay=100)
q = embed(Z, (0,), 2)
modes = quantum_modes(q, w, g, levels=10, channel="z:q0")   # 左节点 + 共轭的右节点
engine = TDVPHEOM(modes, whites=[White(q, k0, "z:q0")], max_dt=0.05)
```

分段浴：`Segment(h, duration, {"z:q1": 5.0})` 在该段把通道 `z:q1` 的耦合放大 5 倍（$Q\to 5Q$），`idle_gain=0` 的模只在点名它的段里打开。`compile_layers(plan, durations, gains={"cz": {"z:q{1}": 5.0}})` 可以按门类型自动加。

`cfdata` 表的单位是拟合时的单位，和 ns 脉冲混用前要自己换算；`k0` 的符号和大小按表里的值原样进入 $-k_0[Q,[Q,\rho]]$。

## 目前没有做的

- **原生门分解**：目前只有 Ising Trotter 步到 sqiSWAP 的分解；一般线路的转译和 CZ 版本还没有写。
- **真实脉冲**：编译器仍是理想恒幅脉冲。绝热/非绝热 CZ 需要三能级，而引擎假定系统维数是 $2^n$。
- **含时包络**：每个 `Segment` 是常数 Hamiltonian，包络要先切成分段常数。
- **有限 shots 的抽样**：分析里的期望值和 Γ 已不需要展开稠密张量，但按 shots 抽取错误/修正历史的 `sample_histories`、`sample_corrections` 仍只适用于可枚举的小模型。
- **GPU**：未实现。每层的探针相互独立且 TDVP 固定秩、形状一致，适合把探针作为批维度。
- **收敛**：`configs/static4_noise.json` 是参数示例，不是器件测量值；四比特完整 SPP 还没有做过层级、键维数、步长的收敛扫描。

## 文档

- [docs/LAYERS_AND_STATIC_HEOM.md](docs/LAYERS_AND_STATIC_HEOM.md)：层、脉冲、静态 HEOM 的公式与配置
- [docs/cpu-validation-20261111.md](docs/cpu-validation-20261111.md)：重构前的本地验证记录（文件名是旧的）
- SPP 的讨论与会议记录在仓库根目录的 [docs/](../docs/)
