# ESM2 on TAPE fluorescence (FLU-S): zero-shot vs fine-tuned

- train 20963 / valid 5235 / test 25517 variants, length 237, wild type = consensus
- test split is the extrapolation split: mean 5.37 mutations, max 15
- CI protocol: 95% bootstrap percentile, 1000 resamples, rng seed 0 — identical protocol in the headline table, the subset table and the per-stratum table below

| model | Spearman | 95% CI (bootstrap, 1000) | RMSE | n |
|---|---|---|---|---|
| null_mean | undefined (constant) | — | 1.4666 | 25517 |
| null_nmut | +0.3416 | [+0.3302, +0.3524] | 7.6178 | 25517 |
| zero_shot | -0.3036 | [-0.3144, -0.2924] | 16.2810 | 25517 |
| ridge | +0.6297 | [+0.6208, +0.6380] | 0.9342 | 25517 |
| mlp_mean5seed | +0.6166 | [+0.6080, +0.6246] | 0.8404 | 25517 |

MLP: 5 seeds, per-seed Spearman [0.611, 0.6108, 0.6046, 0.612, 0.6057], seed spread 0.0074

## Two zero-shot scoring conventions, on the same 5,000-variant subset

| scorer | Spearman | 95% CI |
|---|---|---|
| WT-marginal (biased: mutant is its own context) | -0.3187 | [-0.3435, -0.2936] |
| masked-marginal (position masked before scoring) | +0.0310 | [+0.0014, +0.0574] |
| ridge on frozen embeddings (reference) | +0.6331 | [+0.6130, +0.6517] |
| MLP 5-seed mean (reference) | +0.6175 | [+0.5976, +0.6365] |

Correlation between the two zero-shot scores themselves: +0.3093

## Spearman by number of mutations in the variant (95% bootstrap CI in brackets)

| mutations | n | zero-shot | ridge | mlp |
|---|---|---|---|---|
| 4-6 | 20510 | -0.212 [-0.225, -0.200] | +0.693 [+0.685, +0.700] | +0.673 [+0.665, +0.680] |
| 7-10 | 4816 | -0.046 [-0.075, -0.018] | +0.136 [+0.107, +0.164] | +0.140 [+0.114, +0.168] |
| 11-15 | 191 | +0.072 [-0.074, +0.225] | +0.018 [-0.130, +0.151] | -0.010 [-0.149, +0.133] |

## 分层 ρ 差异的显著性检验（E4）

主推断：Δρ 的 95% bootstrap 百分位 CI（95% bootstrap percentile CI on the delta, 1000 resamples, seed 0 (independent streams per disjoint stratum; paired index draw within a stratum)）；交叉核对：Fisher r-to-z（独立分层适用，Spearman 上为近似）。

最小可检差（末层 vs 首层，Fisher-z 尺度，80% 功效）≈ 0.205——小样本层「检不出」不等于「无差异」。

| 模型 | 层对比 | Δρ | Δρ 95% CI | Fisher z | p（原始/Bonferroni） | 判定 |
|---|---|---|---|---|---|---|
| zero_shot | 4-6 vs 7-10 | -0.166 | [-0.197, -0.135] | -10.57 | 3.95e-26 / 3.55e-25 | 显著 |
| ridge | 4-6 vs 7-10 | +0.557 | [+0.528, +0.585] | +44.75 | <1e-300 / <1e-300 | 显著 |
| mlp | 4-6 vs 7-10 | +0.533 | [+0.503, +0.560] | +42.17 | <1e-300 / <1e-300 | 显著 |
| zero_shot | 7-10 vs 11-15 | -0.118 | [-0.277, +0.028] | -1.59 | 0.111 / 0.997 | 不可区分 |
| ridge | 7-10 vs 11-15 | +0.117 | [-0.016, +0.262] | +1.59 | 0.112 / 1 | 不可区分 |
| mlp | 7-10 vs 11-15 | +0.149 | [+0.005, +0.293] | +2.02 | 0.043 / 0.387 | 边缘（CI 不含 0 但校正后 p≥0.05） |
| zero_shot | 4-6 vs 11-15 | -0.284 | [-0.436, -0.137] | -3.93 | 8.51e-05 / 0.000765 | 显著 |
| ridge | 4-6 vs 11-15 | +0.674 | [+0.543, +0.820] | +11.40 | 4.39e-30 / 3.95e-29 | 显著 |
| mlp | 4-6 vs 11-15 | +0.683 | [+0.543, +0.818] | +11.27 | 1.81e-29 / 1.63e-28 | 显著 |

同层内（同一批变体、相关不独立，配对 bootstrap）：

| 层 | 对比 | Δρ | 95% CI | 判定 |
|---|---|---|---|---|
| 4-6 | ridge - zero_shot | +0.905 | [+0.891, +0.920] | 可区分 |
| 4-6 | ridge - mlp | +0.020 | [+0.015, +0.025] | 可区分 |
| 7-10 | ridge - zero_shot | +0.182 | [+0.141, +0.223] | 可区分 |
| 7-10 | ridge - mlp | -0.004 | [-0.021, +0.012] | 不可区分 |
| 11-15 | ridge - zero_shot | -0.054 | [-0.249, +0.141] | 不可区分 |
| 11-15 | ridge - mlp | +0.028 | [-0.087, +0.144] | 不可区分 |

判读：

- zero_shot: ρ(4-6)−ρ(7-10) = -0.166，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著（负信号随外推减弱，朝 0 靠拢）
- ridge: ρ(4-6)−ρ(7-10) = +0.557，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著
- mlp: ρ(4-6)−ρ(7-10) = +0.533，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著
- zero_shot: ρ(7-10)−ρ(11-15) = -0.118，CI 跨 0 ⇒ 差异不可区分
- ridge: ρ(7-10)−ρ(11-15) = +0.117，CI 跨 0 ⇒ 差异不可区分
- mlp: ρ(7-10)−ρ(11-15) = +0.149，CI 勉强不含 0 但 Bonferroni 后 p=0.39 ⇒ 边缘结果，不作为结论
- zero_shot: ρ(4-6)−ρ(11-15) = -0.284，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著（负信号随外推减弱，朝 0 靠拢）
- ridge: ρ(4-6)−ρ(11-15) = +0.674，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著
- mlp: ρ(4-6)−ρ(11-15) = +0.683，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著
