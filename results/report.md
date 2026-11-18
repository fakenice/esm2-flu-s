# ESM2 on TAPE fluorescence (FLU-S): zero-shot vs fine-tuned

- train 20963 / valid 5235 / test 25517 variants, length 237, wild type = consensus
- test split is the extrapolation split: mean 5.37 mutations, max 15

| model | Spearman | 95% CI (bootstrap, 1000) | RMSE | n |
|---|---|---|---|---|
| null_mean | undefined (constant) | — | 1.4666 | 25517 |
| null_nmut | +0.3416 | [+0.3302, +0.3524] | 7.6178 | 25517 |
| zero_shot | -0.3036 | [-0.3144, -0.2924] | 16.2810 | 25517 |
| ridge | +0.6297 | [+0.6208, +0.6380] | 0.9342 | 25517 |
| mlp_mean3seed | +0.6150 | [+0.6063, +0.6230] | 0.8453 | 25517 |

MLP: 3 seeds, per-seed Spearman [0.611, 0.6108, 0.6046], seed spread 0.0064

## Two zero-shot scoring conventions, on the same 5,000-variant subset

| scorer | Spearman | 95% CI |
|---|---|---|
| WT-marginal (biased: mutant is its own context) | -0.3187 | [-0.3435, -0.2936] |
| masked-marginal (position masked before scoring) | +0.0310 | [+0.0014, +0.0574] |
| ridge on frozen embeddings (reference) | +0.6331 | [+0.6130, +0.6517] |
| MLP 3-seed mean (reference) | +0.6155 | [+0.5951, +0.6343] |

Correlation between the two zero-shot scores themselves: +0.3093

## Spearman by number of mutations in the variant

| mutations | n | zero-shot | ridge | mlp |
|---|---|---|---|---|
| 4-6 | 20510 | -0.212 | +0.693 | +0.671 |
| 7-10 | 4816 | -0.046 | +0.136 | +0.140 |
| 11-15 | 191 | +0.072 | +0.018 | +0.001 |
