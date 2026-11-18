# Changelog

本仓库演进的如实记录。条目日期 = git 提交日期（可用 `git log` 逐条对上）；
建仓之前的私有复现运行不在本仓库 git 历史里，那段时间的运行日志当时未随仓库导入，
本 CHANGELOG 不为其补写叙事。

## [Unreleased]

- 2026-11-18 E5：标注量学习曲线 `src/learning_curve.py`——ridge/MLP × 嵌套 10 档
  （10%–100%，rng seed 42 单次置换）× 与 analyze.py 同一头协议；100% 档对
  `metrics.json` 的复现断言内置为控制门（ridge 精确、MLP 容差 0.01），不达即拒写；
  产物 `results/learning_curve.{json,csv}`、`results/figures/fig5_learning_curve.png`；
  README 加「标注量学习曲线（E5）」节与复现步骤
- 2026-11-18 新增 `docs/external_baselines.md`：TAPE / ESM-1v / ESM2 / ProteinGym 原文数字
  逐条带表号核对，含口径差异清单与"不可引用项"声明；README 加「外部对照」节
- 2026-11-18 新增 `docs/DECISIONS.md`：长度过滤、双 marginals 并报、ManualBertTok、
  E4 双口径检验等决定的仓内依据逐条登记（日期对齐 git）

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
