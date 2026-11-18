# E3 PLM 横评（同一头协议 × 四个冻结编码器）

- 任务/切分/头部/CI 协议与主报告完全一致（ridge 同 14 点 alpha 网格、MLP 同 5 seeds 取均值、bootstrap 95% CI 1000 次 seed 0）；模型间唯一变量是冻结嵌入与 WT-marginal 分数本身。
- 本表不含 masked-marginal 零样本口径（主报告 5,000 条子样本口径仅对 esm2_650M 计算过）。

| 模型 | 参数量级 | 维度 | 嵌入总耗时(s) | zero-shot ρ [95%CI] | ridge ρ [95%CI] | MLP(5 seed) ρ [95%CI] |
|---|---|---|---|---|---|---|
| `esm2_650M` | 650.0M | 1280 | 1893 | -0.3036 [-0.3144, -0.2924] | +0.6297 [+0.6208, +0.6380] | +0.6166 [+0.6080, +0.6246] |
| `esm2_35M` | 33.5M | 480 | 195 | -0.2997 [-0.3108, -0.2885] | +0.6115 [+0.6026, +0.6195] | +0.6101 [+0.6016, +0.6181] |
| `saprot_35m_seqonly` | 33.7M | 480 | 203 | -0.2857 [-0.2964, -0.2741] | +0.6073 [+0.5987, +0.6159] | +0.6054 [+0.5969, +0.6143] |
| `protbert_bfd` | 419.9M | 1024 | 946 | -0.2997 [-0.3110, -0.2880] | +0.6170 [+0.6085, +0.6254] | +0.5775 [+0.5684, +0.5864] |

## 判读（数据驱动，措辞随重跑自动更新）

- 线性头排序（ridge）：esm2_650M > protbert_bfd > esm2_35M > saprot_35m_seqonly。相邻名次是否可区分要看 CI 是否重叠，不能只看点估计。
- 零样本 WT-marginal 排序：saprot_35m_seqonly > esm2_35M > protbert_bfd > esm2_650M；所有模型同一口径（Σ log p(mt)−log p(wt)，无掩蔽），该口径自带自洽偏差，主报告已量化（masked-marginal 与其 ρ 仅 +0.309）。
- `esm2_650M` 零样本为负（-0.3036，CI [-0.3144, -0.2924]）：CI 完全在 0 以下 ⇒ 该口径下排序方向系统性反转（不是抽样噪声）。
- `esm2_35M` 零样本为负（-0.2997，CI [-0.3108, -0.2885]）：CI 完全在 0 以下 ⇒ 该口径下排序方向系统性反转（不是抽样噪声）。
- `saprot_35m_seqonly` 零样本为负（-0.2857，CI [-0.2964, -0.2741]）：CI 完全在 0 以下 ⇒ 该口径下排序方向系统性反转（不是抽样噪声）。
- `protbert_bfd` 零样本为负（-0.2997，CI [-0.3110, -0.2880]）：CI 完全在 0 以下 ⇒ 该口径下排序方向系统性反转（不是抽样噪声）。

## 口径与限制

- SaProt 用的是作者发布的 seqOnly 变体（训练时 3Di 槽全为 `#`，模型卡：用于与 ESM-2 35M 对比）；普通 SaProt 35M/650M 的冻结嵌入按官方 README 需要真实 3Di 结构 token，本基准不提供结构输入。
- ProtBERT 的输入 id 按发布 vocab.txt 行号手工构造（transformers 5 的 BertTokenizer 无 tokenizer.json 可复现路径，见 embed.py `ManualBertTok` 注释）；每残基一 token、[CLS]/[SEP] 位置与官方用法一致。
- 单一蛋白（avGFP）、单一任务（FLU-S）、n=25,517：这是「同一协议下编码器快照的差别」，不是对模型通用强弱的一般结论。
- ridge CI 两两重叠（这些模型对的排序差异不可区分）：esm2_650M vs protbert_bfd；esm2_35M vs saprot_35m_seqonly；esm2_35M vs protbert_bfd；saprot_35m_seqonly vs protbert_bfd
- 复跑：`python src/embed.py --model <tag>`（需 GPU）后 `python src/compare_models.py`（仅 CPU+sklearn/torch 头）。
