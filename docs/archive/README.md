# 旧稿与原始材料

这里保存 2026-11-19 整理前的讨论稿、专题笔记、原始 HTML 和用户提供的四份 Markdown。普通文档只调整迁移所需的相对链接；四份 `05_` 原稿与 HTML 保持原始字节。研究结论、旧任务和完成状态均按原日期理解，不作为当前实现状态。

当前阅读入口只有 [Overview](../overview.md)、[研究任务](../research.md)、[文献叙述](../literature.md)、[实验结果](../experiments.md)、[安装与复现](../usage.md)。文件移动对照如下：

| 归档文件（保留原文件名） | 当前内容入口 |
|---|---|
| [heom-to-spp-outline-20261119.md](heom-to-spp-outline-20261119.md) | Overview：流程图、HEOM 与两种 MPS |
| [spl-to-spp-explainer.md](spl-to-spp-explainer.md) | Overview：概念；完整手算留在此稿 |
| [spp-too-big.md](spp-too-big.md) | Overview + 实验结果 |
| [spp-tensor-network-meeting-20261105.md](spp-tensor-network-meeting-20261105.md) | 实验结果；旧组会图仍在 docs/figures |
| [spp-memory-discussion-20261102.md](spp-memory-discussion-20261102.md) | Overview + 研究任务 |
| [research-plan.md](research-plan.md) | 研究任务 |
| [paired-pulse-proposal.md](paired-pulse-proposal.md) | 研究任务 + 实验结果；细节保留在此 |
| [simulation-report.md](simulation-report.md) | 实验结果；旧动力学详细报告 |
| [spp-ising-benchmark.md](spp-ising-benchmark.md) | 实验结果；受限基准详细报告 |
| [pauli-noise-model-hierarchy.md](pauli-noise-model-hierarchy.md) | 文献叙述；定义与推导留在此稿 |
| [ibm-2607-25998-review.md](ibm-2607-25998-review.md) | 文献叙述；逐节核对依据 |
| [nmnoise-guide.md](nmnoise-guide.md) | 安装与复现；旧模型技术细节 |
| [backup-and-reproduce.md](backup-and-reproduce.md) | 安装与复现 |
| [maintenance-history.md](maintenance-history.md) | 历史记录；新维护记录在代码 README |
| [reading-index.md](reading-index.md) | 被代码 README 的五个入口取代 |
| [literature/README.md](literature/README.md) | 被文献叙述中的阅读顺序取代 |
| [heom-spp-proposal-20261029.html](heom-spp-proposal-20261029.html) | 原始方案；内容和实现声明未因归档而获独立核验 |

此前位于 `docs/literature/` 的四份 `05_` 原稿，现在同名保存在 [literature/](literature/)；阅读顺序和标准 Markdown 链接见[文献叙述](../literature.md)。校验从该目录执行 `shasum -a 256 -c SHA256SUMS`。原稿内的 Obsidian 双链和未提供的外部笔记引用保留原样。

特别注意：旧讨论稿中的“尚未实现 MPO／TDVP”描述当时的 `spp_tn`。当前独立 `heom_spp_cpu` 已含该路径，仍待针对目标模型做收敛验证。历史记录中的通过测试数也不代表当前版本重新测试。

2026-11-19 补充：旧 nmnoise 代码、脚本与测试现位于仓库根的 [archive/](../../archive/README.md)。此目录的历史命令保留原文，重跑时按代码归档 README 调整工作目录与输出路径。
