# 备份与复现

GitHub 公开仓库：[Qubit-Fernand/non-markovian-noise](https://github.com/Qubit-Fernand/non-markovian-noise)。默认分支 `main`。

## 归档内容

- `docs/`：研究设计与实际模拟报告。
- `codex_memory.md`：研究讨论笔记，包含待验证的设想及文献要点。
- `nmnoise/`、`run_demo.py`、`tests/`：求解代码及验证。
- `results/validated/`：2048 条轨迹的主运行、图表、终态、噪声轨迹及测试记录。
- `results/demo/`：512 条轨迹的早期试运行，保留作历史数据；正文报告以 validated 为准。
- `2509.07693v2.pdf`：用户提供的参考论文，原作者版权不因仓库公开而改变。
- `results/SHA256SUMS`：已归档结果文件的 SHA-256 清单。

`tmp/`、`.history/`、`.venv/`、Python 缓存和 pytest 缓存不进入 Git；它们不属于研究交付。

## 从远端恢复

```bash
git clone https://github.com/Qubit-Fernand/non-markovian-noise.git
cd non-markovian-noise
shasum -a 256 -c results/SHA256SUMS
```

## 重跑

在具备依赖的 Qubit 环境运行：

```bash
conda run -n Qubit python -m pytest -q
conda run -n Qubit python run_demo.py --samples 2048 --output results/reproduced
```

新环境可安装 `requirements-reproduce.txt` 中记录的核心版本。绘图还依赖 LaTeX 与 Times New Roman，运行时间随平台和首次导入开销变化。随机种子和模型参数保存在 JSON；跨 BLAS/系统的浮点结果允许舍入差异。

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
