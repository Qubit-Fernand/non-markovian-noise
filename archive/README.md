# 旧 nmnoise 模拟归档

这里保留经典有色噪声轨迹、量子浴 HEOM／显式浴对照，以及配对脉冲和受限两时段 SPP 基准。它们不再属于默认安装和默认测试的主线；当前主线是 [spp_tn 与 heom_spp_cpu](../README.md)。

## 内容

- `nmnoise/`：原模拟包，算法源文件原样移动。
- `run_demo.py`、`run_paired_scan.py`、`run_spp_benchmark.py`：三个原运行脚本。
- `tests/`：对应的三份测试；主线 `test_spp_tn.py` 仍在仓库根测试目录。
- `pyproject.toml`：独立安装与测试配置，分发名为 `nmnoise-archive`，Python 导入名仍为 `nmnoise`。

历史结果和源码快照仍在仓库根 [results/](../results/)，没有迁移或改写。详细模型见 [nmnoise 原指南](../docs/archive/nmnoise-guide.md)，结果汇总见 [experiments.md](../docs/experiments.md)。

## 运行旧实验

激活 TensorCircuitNG 后，从代码仓库根切换到本目录。已有项目数值依赖时可直接运行；需要独立安装时在本目录执行 `python -m pip install -e '.[test]'`。

```bash
cd archive
python -m pytest -q
```

按需要单独选择一条实验命令，每次使用新的输出目录：

```bash
python run_demo.py --samples 2048 --output ../results/my_dynamics
python run_paired_scan.py --output ../results/my_paired_scan
python run_spp_benchmark.py --output ../results/my_restricted_spp
```

必须从本目录运行这些命令，尤其配对扫描会按工作目录保存源码快照。若沿用历史文档中的命令，应先切换到这里，并将输出路径调整为 `../results/<新名字>`。新结果不写入旧归档目录，生成图表仍需 LaTeX 与 Times New Roman。

## 维护记录

- 2026-11-19：将 nmnoise、三份运行脚本和三份测试原样归档，新增独立安装配置；主包只发现 `spp_tn`。截图中的旧 `nonmarkovian_gate_demo.egg-info/` 与 `references-local/` 已用 `trash` 移至回收站，PDF 软链接的目标原件未移动。保留 ignore 规则，防止安装生成的元数据或本机文献链接误入版本管理。

本次归档后验证：TensorCircuitNG 环境中主线 17 项、归档 19 项测试通过；归档测试有 14 条来自 Matplotlib／pyparsing 的弃用警告。源文件与原 Git 版本逐字节一致，主包发现仅包含 `spp_tn`，归档安装仅包含 `nmnoise`。
