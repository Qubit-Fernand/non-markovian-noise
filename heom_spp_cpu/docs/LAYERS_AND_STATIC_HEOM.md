# 独立构造学习层、脉冲和静态 HEOM（2026-11-11）

当前最小模型固定为：HTML 的四比特 Ising 四层，一个比特一个独立静态 Gaussian 噪声坐标。统一 gate-after。

## 1. 三种输入分开保存

| 任务 | 程序 | 输出 |
|---|---|---|
| 读逻辑线路、指定学习层、记录层内并行块 | scripts/build_layers.py、heom_spp/circuit.py | layers.json |
| 将逻辑门编译成有限时长控制 Hamiltonian | heom_spp/pulse.py | Slot/Segment、pulse_schedule.json |
| 构造相关函数系数、耦合算符和层级 | scripts/build_noise.py、heom_spp/bath.py | noise.json、Mode 列表 |
| 传播并构造 SPP | heom_spp/heom.py、heom_spp/spp.py | spp.npz、诊断 |

heom_spp/cli.py 的 make_model 只负责连接这些输入；不在其中写固定 Ising Hamiltonian 和噪声生成公式。

逻辑层没有噪声参数或脉冲时间，NoiseSpec 没有线路门序。残余 ZZ 与 T1 由 pulse.system_terms 单独构造，加入物理传播，排除在理想补偿 U 之外。

## 2. HTML 的层是当前模板，不是所有 Trotter 步的普遍划分

代码使用 q0、q1、q2、q3，依次对应 HTML 的 Q1、Q2、Q3、Q4。

| 学习层 | 按时间先后执行的门块 |
|---|---|
| 1 | 所有比特 Rx；并行 CX(0→1)、CX(2→3) |
| 2 | 并行 Rz(1)、Rz(3)；并行 CX(0→1)、CX(2→3) |
| 3 | CX(1→2)；Rz(2) |
| 4 | CX(1→2) |

第三层末尾包含中间 Rz。每行是一个 SPP 时间槽；一行内有两个顺序门块，不表示要插两次 twirling。

已有可直接使用的模板：

- configs/html_ising4_layers.json
- configs/html_ising4_layers.txt
- configs/static4_noise.json

如需生成自己的新文件：

~~~bash
python scripts/build_layers.py --qubits 4 --steps 1 \
  --theta-x 0.5 --theta-z -0.075 --output settings/my_layers.json
python scripts/build_noise.py --qubits 4 --kind static \
  --gwval 1e-6 --levels 6 --output settings/my_noise.json
~~~

文件采用独占创建，不覆盖已有设置。改用不同 Trotter 顺序、每条边不同角度、CZ 或直接 RZZ，应修改逻辑线路／层 JSON；不需要修改 HEOM 传播器。直接 RZZ 和 CX-RZ-CX 的有限脉冲噪声不同，编译器不会擅自互换。

## 3. Qiskit 的并行门块和学习层不等同

Qiskit 的 DAG 深度能描述门的依赖，但不能唯一决定我们希望采用的 twirling 边界。支持两种明确输入：

1. QuantumCircuit 中的全寄存器 barrier 定义学习层边界；heom:block 标签仅划分层内门块。
2. 没有 barrier 的 QuantumCircuit，另外给定累计门数作为层末位置。HTML 的四层对应 [6,10,12,13]。

没有指定边界时，整条线路保留为一个学习层，不自动对每个 DAG 门深度 twirl。

可选安装 Qiskit（HEOM 与 JSON 运行不需要它）：

~~~bash
python -m pip install ".[circuits]"
~~~

Python 接口：

~~~python
from qiskit import QuantumCircuit
from heom_spp.circuit import from_qiskit

qc = QuantumCircuit(4)
qc.rx(0.5, 0)
qc.barrier()  # 第一学习层结束
qc.cx(0, 1)   # 第二学习层
plan = from_qiskit(qc)
# 或对无 barrier 的完整线路：
# plan = from_qiskit(qc, layer_end_indices=[6, 10, 12, 13])
plan.save("my_layers.json")
~~~

这个两层小例子可直接运行；换成完整 Ising 线路时再选择相应的层末位置。若已有 QPY：

~~~bash
python scripts/build_layers.py --input-qpy my_circuit.qpy \
  --output settings/imported_layers.json
~~~

也可添加 --layer-ends 6,10,12,13，用于没有 barrier 的 QPY。HTML 模板可以用 --output-qpy 导出给 Qiskit 查看。当前适配器测试版本为 Qiskit 2.6.2。

逻辑门目前支持 rx、ry、rz、cx、cz、rzz、id。测量、经典条件、未绑定参数和其他门会明确拒绝，不会被静默丢弃。需要其他门时先显式分解，或扩展 gate_hamiltonian。

q0 始终是我们算符张量积的最左因子，Qiskit 矩阵则以 q0 为最低位。适配器保持逻辑比特编号；与 Qiskit Operator 的矩阵对比须做位反转排列。自动测试已核对，不能直接比较两种排列的矩阵元素。

## 4. 脉冲编译器怎样处理时间

每个学习层产生一个 Slot；其内部顺序门块产生一个或多个 Segment。并行门的控制 Hamiltonian 相加，作用在整个系统上。若同一块中一个门 20 ns、另一个 80 ns，编译器生成 20 ns 的共同控制段和 60 ns 的剩余控制段；先完成的比特进入 idle，其噪声仍然演化。

CNOT 当前为

$$
H_{\mathrm{CX}}
=\frac{\pi}{\tau_{\mathrm{CX}}}
|1\rangle\langle1|_c\otimes\frac{I-X_t}{2}.
$$

这是理想有限脉冲模型，不是器件标定的 CR。逻辑层模块无需知道这个实现；换真实脉冲时修改／扩展 Hamiltonian 编译器，保留同一个学习层接口。

PulseDurations 独立配置 rx/ry/rz、CX、CZ、RZZ 和 idle 的持续时间。底层生成元在每个 Segment 中同时包含控制、噪声、残余项与 Lindblad 耗散。

## 5. 一个静态 node 不等于只保留一个 ADO

对每个比特定义

$$
H_{\mathrm{noise}}=\sum_q\delta_q Q_q,\qquad
Q_q=Z_q/2,\qquad
\langle\delta_q\rangle=0,\quad
\langle\delta_q\delta_r\rangle=\delta_{qr}\sigma_q^2.
$$

静态相关函数为

$$
C_q(t)=\sigma_q^2
=\mathrm{gwval}_q e^{-\mathrm{wval}_q t},
\quad
\boxed{\mathrm{gwval}_q=\sigma_q^2,\quad \mathrm{wval}_q=0.}
$$

因此四比特独立模型有四个浴物理节点，每个节点代表一个指数坐标。每个节点的物理腿仍包含 n_q=0,...,levels-1。levels=6 表示每个节点六个取值；未压缩层级是 6^4 个指标组合，不是六个或二十四个。MPS 避免将其全部展开。

不额外复制一个共轭／负频率节点。这个模型是实系数静态 Gaussian，不需要将同一个实项重复两次。

若只保留 levels=1，升降层耦合会被截掉，非零静态 Gaussian 噪声也会被误删。噪声构造器对此作了明确检查。有限 levels 是层级截断，需要独立收敛扫描，不代表 Gaussian 模型本身只有有限几个失谐值。

程序从解析系综层级出发，没有先随机采样固定偏移再重复演化。静态变量 delta 仅用于定义噪声统计。

## 6. 原始与归一化静态 HEOM

未归一化 ADO 的方程为

$$
\boxed{
\dot\rho_{\boldsymbol n}
=\mathcal L_S(t)\rho_{\boldsymbol n}
-i\sum_q[Q_q,\rho_{\boldsymbol n+\boldsymbol e_q}]
-i\sum_q n_q\sigma_q^2[Q_q,\rho_{\boldsymbol n-\boldsymbol e_q}].
}
$$

一般 OU 还包含 $-\sum_q n_q\,\mathrm{wval}_q\rho_{\boldsymbol n}$；静态时它严格为零，升降项仍保留。这就是 gwval、wval 直接进入方程的位置。

当前默认采用归一化 ADO：

$$
\widetilde\rho_{\boldsymbol n}
=\frac{\rho_{\boldsymbol n}}
{\prod_q \sigma_q^{n_q}\sqrt{n_q!}}.
$$

于是

$$
\boxed{
\dot{\widetilde\rho}_{\boldsymbol n}
=\mathcal L_S(t)\widetilde\rho_{\boldsymbol n}
-i\sum_q\sigma_q
\left[Q_q,
\sqrt{n_q+1}\widetilde\rho_{\boldsymbol n+\boldsymbol e_q}
+\sqrt{n_q}\widetilde\rho_{\boldsymbol n-\boldsymbol e_q}
\right].
}
$$

与本地程序相对应：单个实静态模 normfac=sqrt(gwval)=sigma；下降系数 gwval/normfac=sigma，上升系数 normfac=sigma。因此归一化实现出现 sigma 而非 sigma^2，没有漏一个 sigma，也不应再乘一次 sigma^2。

Mode.coupling_matrix 分别生成：

~~~python
# scaled
B = sigma * (diag(sqrt(1..nb-1), +1) + diag(sqrt(1..nb-1), -1))
# unscaled
B = diag(ones(nb-1), +1) + diag(gwval * arange(1, nb), -1)
~~~

使用 --hierarchy-basis unscaled 可走原始形式。若使用 noise.json，则基底取自该文件；CLI 的其他噪声设置不再覆盖文件。两种基底在相同有限层级、不额外截断时是精确的相似变换，根 ADO 与 SPP 不变；MPS/SVD 的范数会改变，因此有限秩结果要分别做收敛检查。

Q=Z/2 与旧程序可能采用的 Q=Z 必须明确区分。构造脚本支持 --coupling "Z/2" 或 --coupling Z，不暗中修改 gwval。相同 sigma 下两个 Hamiltonian 的噪声幅度相差两倍，不能直接混用数值。

## 7. 连接到 CPU/SPP

直接使用已有配置：

~~~bash
python -m heom_spp.cli run \
  --layers-json configs/html_ising4_layers.json \
  --noise-json configs/static4_noise.json \
  --engine tdvp --workers 26 --blas-threads 1 \
  --rotation-ns 20 --cx-ns 80 --max-dt 2 \
  --heom-bond 16 --time-bond 16 --assembly-bond 64 \
  --output results/cpu_static4_configured
~~~

static4_noise.json 当前示例为每比特 gwval=1e-6、wval=0、levels=6、Q=Z/2。它是模型参数示例，不是器件测量值。

每次运行保存实际的 layers.json、noise.json、pulse_schedule.json。只要使用文件输入，噪声以 noise.json 为准；学习层以 layers.json 为准。改步长或 worker 数不需要改这两个文件。

scaling.sh 的默认模型也已改成 HTML 四比特、每比特一个独立静态模。脚本 levels=3 是有界的伸缩测试起点，不能用来宣称物理层级收敛。

## 8. 验证范围

本轮检查包括：

- HTML 四层的理想总算符与独立 Ising Trotter 公式一致。
- Qiskit 往返、显式门数分层，以及位序变换。
- 不等长并行脉冲的 idle 尾段。
- 每比特一个实静态模，gwval/wval 与耦合算符。
- 原始／归一化生成元相似变换。
- 两个独立静态比特的 Ramsey 解析包络。
- CPU 与独立稀疏参考的全部 ADO 响应。
- 两层静态 SPP 与 Gaussian 联合公式，以及两种基底结果一致。

静态噪声的两层解析概率可独立核对：两段长度均为 t 的纯退相干，令 $c=\exp(-\sigma^2t^2/2)$、$c_2=\exp(-2\sigma^2t^2)$，则

$$
p_Z=\frac{1-c}{2},\qquad
p_{ZZ}=\frac{3-4c+c_2}{8}.
$$

一般 $p_{ZZ}\ne p_Z^2$。自动测试验证了这个差异；没有在层边界重置静态浴。

逻辑 JSON 可以表示更大的线路；当前稠密系统 Hamiltonian、噪声算符编译和完整 Pauli 探针仍限于 1..5 比特。没有因为分离模块就解决系统维数的指数增长。
