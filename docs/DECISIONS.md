# Decision log

本仓库关键口径与协议决定的记录。每条注明**仓内依据**（文件/代码位置）与可核对的日期；
只收当时已留下书面或代码痕迹的决定，不补写无痕迹的叙事。

## 2026-11-18（提交 `cca7863`，建仓即定）

1. **长度过滤：只保留 237 aa（众数长度）**。235/236/237 三种长度不齐，丢弃
   71 + 2,239 条并把丢弃数写入 `data/processed/manifest.json` 的
   `dropped_length_counts`。后果：本仓数字是"长度过滤子集"上的结果，
   不能与 TAPE 全量切分直接并表（`README.md` §数据；`docs/external_baselines.md` §6 口径差异）。
2. **野生型共识序列用逐位众数构造，并用 TAPE 自带 `num_mutations` 做外部校验**：
   51,715 条全部复现、100% 一致（`manifest.json` 的 `splits.*.num_mutations_match`）。
   决定动机：WT 是零样本打分基准，构造必须可证伪而不只是"看起来对"。
3. **两种零样本口径（WT-marginal 与 masked-marginal）必须同时报告**。红线写明
   "不能只报好看的那个"（`README.md` §红线）。masked-marginal 成本高，
   限定 5,000 条子样本（方法表 + `src/zeroshot_masked.py --variants 5000`）。
4. **test 是外推切分（train 平均 2.43 突变 vs test 5.37），据此把"按突变数分层"
   定为主解读轴之一**，而非只报 pooled Spearman（`README.md` §数据、§结果；
   E4 进一步把它变成检验结论，见下）。

## 建仓后追加

5. **2026-11-18 `b6a6ad8`（E3）：SaProt 只用作者发布的 seqOnly 变体**。无结构 token
   输入时 seqOnly 是合法对比口径；不拿结构版 SaProt 冒充"同输入"对比
   （`README.md` §横评；`results/model_comparison.md`）。
6. **2026-11-18 `b6a6ad8`：ProtBERT 输入 id 用 `ManualBertTok` 按发布 `vocab.txt`
   行号手工构造**，不用 transformers 5 的 BertTokenizer——后者会把大写残基小写化、
   special token 位置与 checkpoint 冲突（`src/embed.py` `ManualBertTok` 注释；
   `README.md` §环境）。
7. **2026-11-18 `cd8116b`（E4）：分层差的双口径推断**——Δρ bootstrap CI 为主、
   Fisher r-to-z 交叉核对，9 比较 Bonferroni；层内改用配对 bootstrap（同批变体不独立）。
   并且**必须同时报可检上限**（11–15 层 n=191，MDE≈0.21），"检不出"不得写成"无差异"
   （`README.md` §分层收窄的显著性检验、§红线最后一条）。
8. **2026-11-18 `cca7863`：全部置信区间统一 bootstrap 95% 百分位（1000 次、rng seed 0）**，
   主表、子样本表、分层表同一口径（`README.md` §结果首段）。
9. **权重经 hf-mirror 拉取并逐文件记 sha256**（本机直连 huggingface.co DNS 失败，
   且必须带浏览器 User-Agent 否则 403）；哈希清单 `models/manifest_e3.json`，
   权重不进 git（`README.md` §环境；`src/fetch_weights.py`）。
