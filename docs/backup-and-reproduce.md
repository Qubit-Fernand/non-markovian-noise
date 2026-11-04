# 备份与复现

GitHub 公开仓库：[Qubit-Fernand/non-markovian-noise](https://github.com/Qubit-Fernand/non-markovian-noise)。默认分支 `main`。

## 归档内容

- `docs/`：研究设计与实际模拟报告。
- [README.md](../README.md)：项目说明、研究讨论、待办与维护记录。
- [研究 proposal](paired-pulse-proposal.md)：配对脉冲实验与扫描结果、SPP 学习及缓解研究设想。
- `nmnoise/`、`run_demo.py`、`run_paired_scan.py`、`tests/`：求解代码及验证。
- `results/validated/`：2048 条轨迹的主运行、图表、终态、噪声轨迹及测试记录。
- `results/demo/`：512 条轨迹的早期试运行，保留作历史数据；正文报告以 validated 为准。
- 原参考论文 `2509.07693v2.pdf` 已于 2026-10-27 移出项目根目录并放入回收站；改由 [Zotero 群组条目](https://www.zotero.org/groups/6729418/items/AXEGG4GX)及[在线 v2 PDF](https://arxiv.org/pdf/2509.07693v2)访问。旧 Git 历史可能仍包含该文件。
- `results/SHA256SUMS`：已归档结果文件的 SHA-256 清单。

`tmp/`、`.history/`、venv、Python 缓存、pytest 缓存、本机编辑器配置和环境修复日志不进入 Git。主要安装入口为 `requirements.txt` 与 `environment.yml`；旧环境的约束和验证范围见 `environments/README.md`。

## 从远端恢复

```bash
git clone https://github.com/Qubit-Fernand/non-markovian-noise.git
cd non-markovian-noise
shasum -a 256 -c results/SHA256SUMS
```

## 重跑

从仓库根目录创建或激活项目环境：

```bash
conda env create -f environment.yml
conda activate TensorCircuitNG
python -m pytest -q
python -m spp_tn.demo --output results/spp_reproduced --convergence
python run_demo.py --samples 2048 --output results/reproduced
```

也可在 Python 3.11 的独立环境执行 `python -m pip install -r requirements.txt`。旧安装清单的五项依赖已按相同版本合并到该文件，不再维护第二个根目录安装入口；历史清单可从 Git 历史查看。绘图还依赖 LaTeX 与 Times New Roman，运行时间随平台和首次导入开销变化。随机种子和模型参数保存在 JSON；跨 BLAS/系统的浮点结果允许舍入差异。

## 后续同步

每次先检查工作区与远端更新，再提交和推送：

```bash
git status --short --branch
git fetch origin
git pull --ff-only
git add <明确要备份的文件>
git commit -m "Describe the change"
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

`pull --ff-only` 失败时先检查分叉或冲突；不使用 force push。新增结果时更新 SHA-256 清单。GitHub 备份是此次提交的快照，不会自动同步未来本地修改。
