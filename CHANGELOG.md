# Changelog

本仓库演进的如实记录。条目日期 = git 提交日期（可用 `git log` 逐条对上）；
建仓之前的私有复现运行不在本仓库 git 历史里，那段时间的运行日志当时未随仓库导入，
本 CHANGELOG 不为其补写叙事。

## [Unreleased]

- 计划：标注量学习曲线（ridge/MLP 头 × 10%–100% 训练数据 × 全 seed，bootstrap CI）
- 计划：外部对照表（TAPE / ESM-1v / ESM2 论文公布数字，逐条带出处与口径差异）

## [v1.0] - 2026-11-18

仓库建立并当日冻结为可引用版本（Release v1.0，含 tag 与 `.zenodo.json`）。

### Added

- `cca7863` 建仓：TAPE 荧光蛋白（FLU-S）复现主体——数据登记、两种零样本口径、
  回归头、bootstrap CI、突变数分层、四张图、README 红线约定
- `252b318` E2：MLP 头 5 seeds + 逐层 bootstrap CI
- `b6a6ad8` E3：统一头协议下四编码器横评（ESM2-650M / ESM2-35M / SaProt-35M-seqOnly / ProtBERT-BFD）
- `cd8116b` E4：分层差双口径检验（Δρ bootstrap CI + Fisher-z 交叉验证，层内配对 bootstrap，Bonferroni 校正）
- `cf8c848` `CITATION.cff` + README「引用（How to cite）」节
- `50bd800` `.zenodo.json` 发布元数据（DOI 生效需一次性 Zenodo-GitHub 集成）
