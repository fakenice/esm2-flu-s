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
