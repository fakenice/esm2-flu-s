# 外部对照（已发表数字，逐条核过原文）

核验日期 2026-11-18；取证方式：抓取论文原文页面/PDF 并抽取表格区文本，
本文件每个数字都能对上「出处 + 表号」。与本仓库口径不可比之处逐条列在 §6，
不可比的部分只作量级与方向参照，不并列进同一张成绩表。

## 1. 本仓库锚（先固定自己的口径）

- 数据：TAPE `fluorescence` 官方切分，过滤非 237 aa 后 train 20,963 / valid 5,235 / test 25,517。
- 指标：test 集 Spearman ρ（bootstrap 95% CI，1000 次，seed 0）。
- 本仓库代表数字：ridge +0.630；MLP 五 seed 均值 +0.617；零样本 WT-marginal −0.304；
  masked-marginal（5,000 条子样本）+0.031；只看突变数基线 +0.342。

## 2. TAPE 论文（Rao et al., NeurIPS 2019, arXiv:1906.08230）——同任务已发表数字

Table 2 "Fluorescence" 列（test 集 Spearman ρ），逐字核自原文表：

| 设置 | 模型 | Fluorescence ρ |
|---|---|---|
| 自监督预训练 + 微调 | Transformer（12 层 / hidden 512 / **38M**） | **0.68**（表内最佳） |
| 预训练 | LSTM | 0.67 |
| 预训练 | Alley et al. | 0.67 |
| 预训练 | Bepler et al. | 0.33 |
| 预训练 | ResNet | 0.21 |
| 无预训练 | Transformer | 0.22 |
| 特征基线 | One-hot | 0.14 |

切分与协议（原文 §4.2 / §5）：train = avGFP 亲本的 Hamming-3 邻域，test = ≥4 个突变
（外推切分）；预训练语料 Pfam；所有模型接同一套 task-specific 回归头做端到端微调。
"38M" 需要点名：不少后续论文把 TAPE 基线标成 120M 参数，原文是 38M。

与本仓库的关系：同源数据、同为 test Spearman，**但**本仓库过滤了长度（test 25,517 vs
官方 27,217）、头协议完全不同（冻结 ESM2-650M 嵌入 + 岭回归/MLP vs 38M Transformer
端到端微调）。+0.630 与 0.68 可作同任务量级参照，不能写成"与 TAPE 最佳同协议"。

## 3. ESM-1v（Meier et al., NeurIPS 2021）——没有可引的 FLU 专项数字

其评测集是 41 个 DMS 数据集（EVE/Riesselman 协议），**不使用 TAPE 切分**；
正文出现的 "FLU" 是流感蛋白的数据集前缀（`HG_FLU_Bloom2016`、`PA_FLU_Sun2015`），
不是 TAPE 的 GFP 荧光任务。所以"ESM-1v 论文报告的 FLU Spearman"不存在，不可引用。

可引的是跨 41 数据集的均值（Table 2，zero-shot，平均 |Spearman ρ|，full/test）：

| 模型 | full | test |
|---|---|---|
| UniRep | 0.156 | 0.151 |
| TAPE 表征 | 0.171 | 0.175 |
| ProtBERT-BFD | 0.428 | 0.399 |
| ESM-1b | 0.459 | 0.424 |
| ESM-1v（5 模型均值） | 0.484 | 0.457 |
| ESM-1v（ensemble） | 0.509 | 0.482 |

Table 1 另有：MSA Transformer 0.542/0.524；ESM-1v zero-shot 0.509/0.482 →
fine-tune 0.538/0.519。这是"跨任务均值"，与本仓库单任务数字不可直接对比。

**与本仓库真正呼应的是打分口径结论**（附录 Table 5）：该论文在 41 数据集上比较了
masked marginals / wildtype marginals / mutant marginals / pseudolikelihood 四种打分，
masked marginal（其 Eq. 1）最好。本仓库 E3 在单一荧光任务上看到更极端的方向——
WT-marginal 在全部四个编码器上都稳定为 −0.29 ~ −0.30，masked-marginal 才回到 +0.031。
方向一致（wt 口径更差），但本仓库观察到"负排序"这一现象本身只在单任务成立。

## 4. ESM2 论文（Lin et al., Science 2023, bioRxiv:2022.07.20.500902）——正文无 FLU 表

该文主题是结构预测（ESM Fold），正文未报告 TAPE FLU 或任何荧光零样本 Spearman；
GitHub README（含历史版本）亦无。补充材料本次抓取被反爬拦截、未能核验。
因此**不引用"ESM2 论文的 FLU 数字"**——若后续核到补充材料表号，再补本节。

## 5. ProteinGym（Notin et al., NeurIPS 2023 Datasets & Benchmarks）——跨 217 assay 均值

Table 2（zero-shot substitution，217 个 assay 上的 "corrected average" Spearman），逐字核自原文表：

| 模型 | Spearman |
|---|---|
| TranceptEVE L | 0.456 |
| GEMME | 0.455 |
| EVE (ensemble) | 0.439 |
| MSA Transformer (ensemble) | 0.434 |
| Tranception L | 0.434 |
| ESM-1v (ensemble) | 0.407 |
| ESM-2 (15B) | 0.401 |
| ESM-1b | 0.394 |
| RITA XL | 0.372 |
| UniRep | 0.190 |

- 集合中含 GFP 类 assay（Table A19 列有 "GFP / Sarkisyan et al. [2016]"，即 TAPE 数据源，
  另有 Gonzalez Somermeyer 2022 的三个 GFP 变体 assay），但 0.40x 是**跨全部 assay 的均值**，
  与单一荧光任务数值不可比，也不能反过来用单任务数字去"验证"该均值。
- **Table 2 无 SaProt**（全文检索 0 命中）——本仓库 E3 横评用了 SaProt-35M-seqOnly，
  在该发表表中没有可对照的数字；引用 SaProt 只能引其自身论文，本次未核，不引。
- 出处说明：arXiv API 按标题检索 "ProteinGym" 命中 0（未分配可检索的 arXiv 条目），
  本文件以 NeurIPS proceedings PDF 为引用锚（即核验所据），bioRxiv:2023.12.07.570727 为预印本。

## 6. 口径差异清单（引用本节任何数字前必读）

1. **样本集**：TAPE 官方全 test vs 本仓库长度过滤后的子集（差约 6%）。
2. **头/协议**：端到端微调（TAPE 0.68）vs 冻结嵌入 + 线性/MLP 头（本仓库 0.630）
   vs 零样本打分（ESM-1v、ProteinGym 线）。三种协议同表并列没有意义。
3. **统计量范围**：单任务单蛋白（本仓库、TAPE FLU 列）vs 跨 41/217 数据集均值（§3、§5）。
   后者对模型规模、assay 构成极其敏感，只作量级参照。
4. **指标定义**：ProteinGym 的 "corrected average" 先在 assay 内平均（对分布做校正），
   与本仓库直接在合并 test 集上算 ρ 不是同一件事。
5. **零样本口径**：masked marginal（文献主流，Eq. 1）vs WT-marginal（本仓库默认展示的那种），
   后者在本任务上系统性翻负——两口径的分数间相关只有 +0.309，必须分开引用。

## 7. 这组对照支撑什么、不支撑什么

- 支撑：冻结表征 + 轻量头到 +0.63 的量级是合理的（同任务端到端微调的发表最佳为 0.68）。
- 支撑：WT-marginal 的负排序不是本仓库的实现 bug——文献（ESM-1v Table 5）早于本仓库
  发现 wt 口径劣于 masked 口径；本仓库把它推到了"方向反转"的程度并给了四编码器证据。
- 不支撑：把 +0.630 写成"复现了 TAPE 最佳"或"优于 TAPE 基线"（协议不同、子集不同）。
- 不支撑：用 ProteinGym/ESM-1v 的跨任务均值给单一荧光任务排名，或反之。

## 8. 引用锚（均于 2026-11-18 实际抓取核对）

- TAPE：arXiv:1906.08230；正文与 Table 2 经 PMC7774645 原文核对。
- ESM-1v：NeurIPS 2021 proceedings（proceedings.neurips.cc，f51338d7…-Paper.pdf）；
  Table 1/2 与"Scoring methods"/Table 5 经 PDF 全文抽取核对。
- ProteinGym：NeurIPS 2023 Datasets & Benchmarks proceedings
  （cac723e5…-Paper-Datasets_and_Benchmarks.pdf）；Table 2 与 Table A19 经 PDF 全文抽取核对。
- ESM2：Science 379:1123–1130 (2023)；仅核到"正文无 FLU 表"，补充材料未核到、不引用。
