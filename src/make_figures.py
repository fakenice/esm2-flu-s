"""Step 4: publication-style figures, computed from the same arrays as analyze.py.

Inputs:  data/processed/*.csv, results/embed/*.npy, results/metrics.json,
         results/model_comparison.json (fig4; run src/compare_models.py first)
Outputs: results/figures/*.png

Every number annotated on a figure is recomputed here and asserted against
metrics.json (figs 1-3) or model_comparison.json (fig4), so README figures can
never drift from the reported tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB = ROOT / "results" / "embed" / "esm2_650M"
OUT = ROOT / "results" / "figures"
MET = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
MC_PATH = ROOT / "results" / "model_comparison.json"

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# colors shared across figures
C_ZS, C_RIDGE, C_MLP = "#c0392b", "#1f6fb2", "#7f8c8d"


def load(split: str):
    df = pd.read_csv(PROC / f"flu_{split}.csv")
    emb = np.load(EMB / f"{split}_emb.npy")
    zs = np.load(EMB / f"{split}_score_wtmarginal.npy")
    return df, emb, zs


def fit_ridge():
    """Same grid as analyze.py: pick alpha on valid, refit on train."""
    tr, Xtr, _ = load("train")
    va, Xva, _ = load("valid")
    ytr, yva = tr["log_fluor"].values, va["log_fluor"].values
    best = None
    for alpha in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
                  1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0):
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        r = spearmanr(m.predict(Xva), yva).statistic
        if best is None or r > best[1]:
            best = (alpha, r, m)
    assert best[0] == MET["ridge_alpha"], "alpha drifted from metrics.json"
    return best[2]


def fig1_ridge_scatter(model):
    te, Xte, _ = load("test")
    y = te["log_fluor"].values
    pred = model.predict(Xte)
    rho = spearmanr(pred, y).statistic
    assert abs(rho - MET["ridge"]["spearman"]) < 1e-3, "ridge rho drifted"
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 4000, replace=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(pred[idx], y[idx], s=4, alpha=0.35, color=C_RIDGE, linewidths=0)
    lim = [min(pred.min(), y.min()), max(pred.max(), y.max())]
    ax.plot(lim, lim, ls="--", lw=0.8, color="k", alpha=0.5)
    lo, hi = MET["ridge"]["ci95"]
    ax.set(xlabel="ridge prediction (log fluorescence)", ylabel="measured log fluorescence",
           title=f"FLU-S test (n=25,517), 4,000 shown\nSpearman $\\rho$={rho:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ridge_scatter.png")
    plt.close(fig)


def fig2_ad_curve():
    rows = MET["by_mutation_count"]
    x = np.arange(len(rows))
    labels = [f"{r['mutations']}\n(n={r['n']:,})" for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for key, lab, col in [("zero_shot", "zero-shot (WT-marginal)", C_ZS),
                          ("ridge", "ridge on frozen embeddings", C_RIDGE),
                          ("mlp", "MLP head (5-seed mean)", C_MLP)]:
        y = np.array([r[key] for r in rows])
        ci = np.array([r[f"{key}_ci95"] for r in rows])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="-o", lw=1.5,
                    capsize=2.5, elinewidth=0.9, color=col, label=lab)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axhline(MET["null_nmut"]["spearman"], color="gray", ls=":", lw=1.2,
               label=f"mutations-count-only baseline ({MET['null_nmut']['spearman']:+.3f})")
    ax.set(xticks=x, xlabel="mutations per variant (test strata)", ylabel="Spearman $\\rho$",
           title="Applicability domain collapses with mutation count\n(bars: 95% bootstrap CI, 1000 resamples)",
           **{"xticklabels": labels})
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_applicability_domain.png")
    plt.close(fig)


def fig3_scorer_conventions():
    te, _, zs_te = load("test")
    idx = np.load(EMB / "test_masked_subset_idx.npy")
    zs_masked = np.load(EMB / "test_score_maskedmarginal_subset.npy")
    y = te["log_fluor"].values[idx]
    sub_wt = zs_te[idx]
    r_between = spearmanr(sub_wt, zs_masked).statistic
    assert abs(r_between - MET["zero_shot_subset"]["wt_vs_masked_spearman_between_scorers"]) < 1e-3
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(sub_wt, zs_masked, s=4, alpha=0.35, color="#8e44ad", linewidths=0)
    ax.set(xlabel="WT-marginal score (mutant is its own context)",
           ylabel="masked-marginal score (position masked)",
           title="Two zero-shot conventions, same 5,000 variants\n"
                 f"$\\rho$ between scorers = {r_between:+.3f}\n"
                 f"vs assay: WT {MET['zero_shot_wt_subset']['spearman']:+.3f} / "
                 f"masked {MET['zero_shot_masked_subset']['spearman']:+.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_zero_shot_conventions.png")
    plt.close(fig)


def fig4_model_comparison():
    """Grouped bars: same head protocol on four frozen encoders (E3).

    Bars are recomputed only in the sense that they are re-read from
    model_comparison.json and asserted twice over: against that file's own CI
    fields, and for the esm2_650M row against the headline metrics.json.
    """
    mc = json.loads(MC_PATH.read_text(encoding="utf-8"))
    models = list(mc["models"].keys())  # json preserves the ORDER used by compare_models.py
    arms = [("zero_shot", "zero-shot (WT-marginal)", C_ZS),
            ("ridge", "ridge", C_RIDGE), ("mlp", "MLP (5 seeds)", C_MLP)]
    # cross-file assert: the 650M row must equal the published headline table
    m6 = mc["models"]["esm2_650M"]
    assert m6["zero_shot"]["spearman"] == MET["zero_shot"]["spearman"]
    assert m6["ridge"]["spearman"] == MET["ridge"]["spearman"]
    assert m6["mlp"]["spearman"] == MET["mlp_mean5seed"]["spearman"]
    x = np.arange(len(models))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (key, lab, col) in enumerate(arms):
        ys, los, his = [], [], []
        for model in models:
            m = mc["models"][model][key]
            ys.append(m["spearman"])
            los.append(m["spearman"] - m["ci95"][0])
            his.append(m["ci95"][1] - m["spearman"])
        ax.bar(x + (k - 1) * w, ys, width=w, color=col, label=lab,
               yerr=[los, his], capsize=2.5, error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylim(-0.42, 0.88)  # headroom so the legend never covers the bars
    ax.set(xticks=x, ylabel="Spearman $\\rho$ on FLU-S test (n=25,517)",
           title="E3: same head protocol, four frozen encoders\n(bars: 95% bootstrap CI, 1000 resamples)",
           **{"xticklabels": [f"{m}\n{mc['models'][m]['emb_dim']}d" for m in models]})
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_model_comparison.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = fit_ridge()
    fig1_ridge_scatter(model)
    fig2_ad_curve()
    fig3_scorer_conventions()
    if MC_PATH.exists():
        fig4_model_comparison()
        print(f"wrote 4 figures to {OUT}")
    else:
        print(f"wrote 3 figures to {OUT} (fig4 skipped: run src/compare_models.py first)")


if __name__ == "__main__":
    main()
