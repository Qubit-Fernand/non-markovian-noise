# 安装、运行和复现

除明确切换目录的代码块外，命令均从代码仓库根目录（本地项目的 `Code/`）执行。主线与归档计算包的模型和用途见 [overview.md](overview.md)，现有结果见 [experiments.md](experiments.md)。

## 安装环境

```bash
git clone https://github.com/Qubit-Fernand/non-markovian-noise.git
cd non-markovian-noise
conda env create -f environment.yml
conda activate TensorCircuitNG
```

已创建环境时只需激活。统一依赖入口是 [requirements.txt](../requirements.txt)，由 [environment.yml](../environment.yml) 创建 Python 3.11 环境并安装；也可在独立 Python 3.11 环境执行 `python -m pip install -r requirements.txt`。不再使用项目旧 venv，也不维护第二份根目录复现清单。

该清单安装主线 `spp_tn`，旧 `nmnoise` 已移入 `archive/`，**不会安装独立 CPU 包**。若要运行 MPO／TDVP，再执行：

```bash
python -m pip install -e './heom_spp_cpu[test]'
```

CPU 包的依赖范围定义在其 [pyproject.toml](../heom_spp_cpu/pyproject.toml)；额外的 threadpoolctl 等不在根清单中全部锁死。Qiskit 只在需要线路导入／导出时安装 `./heom_spp_cpu[circuits]`。做严格复现时另存该次实际环境的 `pip freeze`，不要把根清单称为 CPU 包的完整锁文件。

TensorCircuit-NG 与 legacy TensorCircuit 共用 `import tensorcircuit`，要分环境二选一。固定版本、legacy 对照和历史验证集中在 [environments/README.md](../environments/README.md)。核心求解不需 LaTeX；论文风格绘图另需 LaTeX 和 Times New Roman。

## 分别运行两套测试

根配置只收集主线 `tests/`，不会自动包含 CPU 包或归档测试：

```bash
python -m pytest -q
```

CPU 包单独检查：

```bash
cd heom_spp_cpu
python -m pytest -q
cd ..
```

文档中的历史测试数只对应原记录日期，不表示本次运行通过。

## 沿流程选择入口

| 目的 | 入口 | 输出／限制 |
|---|---|---|
| 从已有小模型了解整条链 | `python -m spp_tn.demo` | 单比特 SPP、PEA、PEC 和 shots；RK4 |
| MPO／TDVP 传播及可配置 SPP | `python -m heom_spp.cli` | 独立 CPU 包；控制层／噪声 JSON |
| 经典轨迹与量子浴动力学对照 | 在 `archive/` 执行 `python run_demo.py` | 旧 `nmnoise` 模型 |
| 配对脉冲的记忆分辨实验 | 在 `archive/` 执行 `python run_paired_scan.py` | 合成扫描，可能耗时 |
| 受限两时段 SPP 学习与 PEC | 在 `archive/` 执行 `python run_spp_benchmark.py` | 小型四历史基准 |

### 单比特完整流程

```bash
python -m spp_tn.demo --output results/my_spp_run --convergence
```

每次选未存在的目录。输出包括参数、版本、源码快照、SPP 节点、shots 样本及 `summary.json`；失败目录保留供诊断。API、gate-after、`vec_F`、三种截断的含义见 [spp_tn/README.md](../spp_tn/README.md)。

### MPO／TDVP 的最小检查与配置运行

从 CPU 包目录先做 smoke：

```bash
cd heom_spp_cpu
bash scripts/smoke.sh
```

需要四比特配置时，再从该目录运行下例；它比 smoke 大得多，不能作为已经完成收敛的默认参数：

```bash
python -m heom_spp.cli run \
  --layers-json configs/html_ising4_layers.json \
  --noise-json configs/static4_noise.json \
  --engine tdvp --workers 1 \
  --heom-bond 16 --time-bond 16 --assembly-bond 64 \
  --output results/my_static4
```

输出保存实际层、噪声、脉冲 schedule、SPP 和版本快照。`--workers` 应按机器及问题规模选择；多进程并不总更快。该包时间单位为 ns，Hamiltonian 为 rad/ns。原生门、静态积分参考、Slurm 和分析参数见 [CPU 包说明](../heom_spp_cpu/README.md)。

**当前两项已知问题必须纳入复现流程：** 收敛扫描可能复用不同参数的旧结果，基线分析只核对形状而未核对线路。每组物理参数使用新的 `ROOT`，比较前核对 `layers.json`、`noise.json` 和层划分。具体限制见 [known-issues.md](../heom_spp_cpu/docs/known-issues.md)。

### 旧模拟与受限基准

旧包、脚本和对应测试统一保留在 [archive/](../archive/README.md)。从代码根目录先切换到归档目录，再按需要单独运行：

```bash
cd archive
python -m pytest -q
python run_demo.py --samples 2048 --output ../results/my_dynamics
python run_paired_scan.py --output ../results/my_paired_scan
python run_spp_benchmark.py --output ../results/my_restricted_spp
```

三条实验命令互不依赖。归档有独立 `pyproject.toml`，需要安装时在其中执行 `python -m pip install -e '.[test]'`。原配对扫描依赖当前目录保存源码，因此应从 `archive/` 运行。旧详细模型与命令保留在 [nmnoise 原指南](archive/nmnoise-guide.md)，阅读历史命令时按上述工作目录与新输出路径调整。

## 保存与复现约定

1. 每次使用独立输出目录，保存物理配置、所有截断与求解容差、随机种子、版本、源码快照和机器信息。分别变化层级、步长和各类键，避免同时改变后无法归因。
2. 原始 `results/` 及其中 `source/` 不随代码整理迁移或改写；重跑用新目录。跨平台允许舍入差异，比较应依据事先定义的容差。
3. 从仓库根目录验证已归档清单：`shasum -a 256 -c results/SHA256SUMS`。清单只覆盖其中列出的文件；新增结果单独生成或更新清单，不将“命令通过”解释为所有结果均受覆盖。
4. 四份文献原稿的校验从 `docs/archive/literature/` 执行 `shasum -a 256 -c SHA256SUMS`。原稿双链和外部笔记缺失情况见 [literature.md](literature.md)。
5. `tmp/`、缓存、虚拟环境和本机审计文件不作为研究结果提交；不批量清理这些目录来代替结果确认。需要删除时使用 `trash`。

## 文稿、PDF 与同步

本地项目的 `Code/` 与 `Overleaf/` 是独立 Git 仓库；共享 PDF 通过同级 `Zotero/` 链接访问。独立克隆代码不携带这两个同级入口，文献可用 [Zotero 群组](https://www.zotero.org/groups/6729418/collections/FH728L4C)及文中的 arXiv 地址访问。

同步前检查工作区与远端，显式选择文件；提交不等于推送。需要更新时使用普通 `fetch`／`pull --ff-only`／`push`，遇到分叉先检查，不使用 force push。推送后核对本地 HEAD 与远端引用。完整旧维护记录保留在 [archive/maintenance-history.md](archive/maintenance-history.md)。
