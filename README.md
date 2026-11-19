# 非马尔可夫噪声：从模型到 SPP

研究有限时长控制下的时间相关噪声，通过 HEOM 与张量网络构造时空 Pauli 过程（SPP），用于预测与误差缓解。当前结果均为合成数值模拟，尚无真机噪声学习或硬件验证。

## 从这里读

| 文档 | 回答的问题 |
|---|---|
| [Overview：从噪声模型到 SPP](docs/overview.md) | 完整流程图；HEOM、MPS、MPO、TDVP 各负责什么 |
| [研究任务](docs/research.md) | 按流程推进的工作、验收标准与待解决问题 |
| [文献叙述](docs/literature.md) | 测控、SPP、张量网络和缓解文献怎样接起来 |
| [实验结果](docs/experiments.md) | 已有表示、收敛、PEA／PEC 和 CPU 结果及其范围 |
| [安装与复现](docs/usage.md) | 环境、三个包的运行入口、结果保存与校验 |

## 当前实现

| 包 | 职责与状态 | 详细接口 |
|---|---|---|
| `spp_tn` | TensorCircuit-NG NumPy 后端；经典 Hermite 层级 MPS、压缩 RK4、SPP 时间 MPS、PEA／PEC | [包说明](spp_tn/README.md) |
| `heom_spp_cpu` | 独立 NumPy／SciPy CPU 包；显式 MPO、单站点 TDVP、RK4／稀疏参考、探针并行和静态积分参考 | [包说明](heom_spp_cpu/README.md) |

**已有 MPO／TDVP 实现，下一步重点是目标模型的正确性与收敛验证。** 保留 RK4 和独立参考进行比较，通过后再扩展 CPU 优化与 GPU 并行；当前无 GPU 路径。两项影响结果比较的[已知问题](heom_spp_cpu/docs/known-issues.md)仍未修复。

旧 `nmnoise` 模拟及其脚本、测试已移入 [archive/](archive/README.md)，不再随主包安装或默认测试。已有结果仍保留在 `results/`。

## 快速运行

在代码仓库根目录执行；已有环境只需激活：

```bash
conda env create -f environment.yml
conda activate TensorCircuitNG
python -m pytest -q
python -m spp_tn.demo --output results/my_spp_run --convergence
```

输出目录使用新名字。固定依赖主入口是 [requirements.txt](requirements.txt)。MPO／TDVP 所在 CPU 包需要另行安装和测试，见 [usage.md](docs/usage.md)；根目录 pytest 不会包含它。

## 文稿与文献

本地文稿在同级 [Overleaf/](../Overleaf/)，是独立 Git 仓库。PDF 与逐篇主题索引在 [Zotero/README.md](../Zotero/README.md)；该目录链接到共享文献库的 `Group/Error-Mitigation/Non-Markov/`。这两个同级目录不包含在代码仓库的独立克隆中。在线入口：[Zotero 群组 Non-Markov](https://www.zotero.org/groups/6729418/collections/FH728L4C)。

## 待办（TODO）

任务只在 [research.md](docs/research.md) 维护，按 Overview 流程推进：固定控制与噪声输入 → 验证 HEOM／MPO／TDVP → 验证逐层 SPP 及预测／缓解 → 扩规模 → CPU／GPU 性能比较。文献中的计划、已有代码和经过收敛验证的结果分别标注。

## 维护约定与记录

- 2026-11-19：归档 nmnoise 及其脚本／测试，主包只发现 spp_tn；旧 egg-info 与 references-local 用 trash 移至回收站，保留 ignore 规则。归档源码与旧结果未改写。验证：主线 17 项、归档 19 项测试通过；包发现与本轮相关的 104 个本地链接检查通过。

- 研究说明保留上述五个入口，API 与配置细节留在包 README；不再为每次讨论新建同级索引。
- 旧讨论稿、原始文献 Markdown、HTML 和旧维护记录统一放在 [docs/archive/](docs/archive/README.md)。历史状态以日期为界，当前状态看本页。
- 原始 `results/` 及源码快照保持原位；每次新运行保存配置、版本和独立输出目录。PDF 由共享 Zotero 目录管理。
- 2026-11-19：合并文档入口，Overview 保留并更新流程图；同步 MPO／TDVP 状态与任务顺序。四份用户原稿及 HTML 保留原内容，修正普通文档迁移后的链接。本轮未修改求解代码、已有结果或环境依赖，也未重新运行数值实验。
- 此前环境迁移、模拟与文献核对记录见 [维护历史](docs/archive/maintenance-history.md)。
