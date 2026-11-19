# 安装与环境复现

本项目推荐 **Python 3.11 + TensorCircuit-NG 1.10.0**。TensorCircuit-NG 与 legacy TensorCircuit 共用 `import tensorcircuit`，应分别安装在不同环境中。

## 默认环境

统一创建、运行和分包测试见 [usage.md](../docs/usage.md)。[requirements.txt](../requirements.txt) 是固定版本主入口，[environment.yml](../environment.yml) 引用它；[tensorcircuit-ng.txt](tensorcircuit-ng.txt) 保留为兼容入口，pip 命令从代码根目录运行。

根清单安装主线 `spp_tn`；旧 `nmnoise` 已归档，可按 [归档说明](../archive/README.md)单独安装。独立 `heom_spp_cpu` 需另外安装，其全部传递依赖未由根清单锁定。此页保留版本与环境验证记录。

| 主要依赖 | 验证版本 |
|---|---|
| TensorCircuit-NG / tensornetwork-ng | 1.10.0 / 0.5.1 |
| NumPy / SciPy | 2.4.6 / 1.15.2 |
| QuTiP / Matplotlib | 5.2.2 / 3.10.3 |
| pytest / ipykernel | 9.0.2 / 6.30.0 |

当前 SPP 算法使用 NumPy 后端；不要求安装 TensorFlow、JAX 或 PyTorch。核心测试和 SPP 基准不需要 LaTeX；原有绘图脚本需要系统 LaTeX 和 Times New Roman。依赖版本固定到主要包及数值后端，不保证不同平台的浮点结果逐位相同，也不是所有传递依赖的跨平台锁文件。

## Legacy 环境

如果需要单独复核 legacy TensorCircuit，可在另一个 Python 3.11 环境安装 `python -m pip install -e '.[test,tn-legacy]'`；该 extra 限制 NumPy `<2`。这是本项目 NumPy 后端的兼容性路径。

[`tensorcircuit-legacy.txt`](tensorcircuit-legacy.txt) 则记录上游 legacy 仓库的一套 **macOS ARM / Python 3.10** 四后端配置：TensorCircuit 0.12.0、TensorFlow 2.15.1、JAX/JAXLIB 0.4.26、Optax 0.1.9、Chex 0.1.86、PyTorch 2.2.2、NumPy 1.26.4、SciPy 1.15.2。它不用于本项目的默认安装，也不包括上游全部可选集成或文档构建依赖。

在验证机器上，TensorFlow 与 PyTorch 2.2.2 同进程、多线程 `searchsorted` 会触发原生崩溃；`OMP_NUM_THREADS=1` 的对照通过。因此运行该旧版多后端组合时使用：

```bash
OMP_NUM_THREADS=1 python environments/verify_backends.py --backends numpy tensorflow jax pytorch
```

此限制针对已验证的旧版组合，默认 NG 环境无需该设置。Cirq 翻译等可选集成另行安装。

## 验证记录

- 2026-11-04：独立 NG conda 环境 `pip check` 通过，项目 **36 项测试通过**。
- 一比特完整基准重跑见 [`../results/spp_tn_conda_20261104/`](../results/spp_tn_conda_20261104/)，与前一环境的概率张量 L2 差为 `8.27e-15`。
- Legacy 上游门操作／后端测试：**124 passed, 3 skipped**，不等于完整上游 CI。
- NumPy、TensorFlow、JAX、PyTorch 的 Bell 态检查通过；后三者自动微分通过，TensorFlow／JAX 的 JIT 检查通过。脚本为 [`verify_backends.py`](verify_backends.py)。
- 本机修复日志、环境导出和编辑器路径不随公开仓库分发；数值运行的参数、版本和源码快照保留在结果目录中。
