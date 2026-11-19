"""Follow-up experiment: how much labeled training data does the head actually need?

Question this answers (asked in README as future work, now measured): the
ridge / MLP numbers in metrics.json are trained on the full 20,963-variant
train split. Is that split already more than a linear head can use, or is the
headline still data-limited? This is the natural held-back experiment for a
frozen-embedding benchmark: everything except the training-set size is held
fixed.

Protocol (identical to analyze.py in every other respect):
  - fractions 10%..100% of the train split, NESTED subsets from one
    permutation of rng seed 42 (fixed in this script; subset choice is a
    design constant, not tuned);
  - ridge: alpha re-selected on the valid split over the same
    RIDGE_ALPHAS grid at every fraction (no alpha carried over);
  - MLP: same 5 seeds (MLP_SEEDS), mean-of-5 predictions, same hidden/lr/wd
    defaults from analyze.mlp_head;
  - scoring: score() -> Spearman + 1000-resample bootstrap percentile CI
    (seed 0) + RMSE on the full 25,517-variant extrapolation test split.

Control check (self-imposed): the 100% fraction must reproduce the
metrics.json headline (ridge and mlp_mean5seed). The ridge line is exact
(deterministic alpha selection + identical seeds); the MLP line is compared
within a tolerance because torch on GPU is not bit-reproducible run to run --
but the tolerance is derived from the seed-to-seed scatter measured in this
same run (2 x sd over the 5 seeds at 100%), not hand-set, so it can never grow
to the size of the ridge-minus-MLP gap it is supposed to leave interpretable.
If the control fails, this script refuses to write the curve.

Inputs:  data/processed/flu_*.csv, results/embed/esm2_650M/*.npy, results/metrics.json
Outputs: results/learning_curve.csv, results/learning_curve.json,
         results/figures/fig5_learning_curve.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from analyze import (load, score, RIDGE_ALPHAS, MLP_SEEDS,
                     mlp_head, OUT, PROC)

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUBSET_SEED = 42          # design constant, fixed before looking at any result
MLP_TOLERANCE = 0.01      # absolute ceiling for the MLP control line
MIN_MLP_TOLERANCE = 0.002  # never demand more precision than the gate can mean


def main() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

    tr, Xtr, _ = load("train")
    va, Xva, _ = load("valid")
    te, Xte, _ = load("test")
    ytr, yva, yte = tr["log_fluor"].values, va["log_fluor"].values, te["log_fluor"].values
    n = len(tr)

    order = np.random.default_rng(SUBSET_SEED).permutation(n)
    sizes = {f: max(1, int(round(f * n))) for f in FRACTIONS}

    rows: list[dict] = []
    print(f"train n={n}  fractions={FRACTIONS}  subset_seed={SUBSET_SEED}")

    for f in FRACTIONS:
        sub = order[: sizes[f]]
        Xs, ys = Xtr[sub], ytr[sub]

        # --- ridge, alpha re-selected on valid at this fraction ---
        best = None
        for alpha in RIDGE_ALPHAS:
            m = Ridge(alpha=alpha).fit(Xs, ys)
            r = spearmanr(m.predict(Xva), yva).statistic
            if best is None or r > best[1]:
                best = (alpha, r, m)
        alpha, _, model = best
        res_r: dict = {}
        row_r = score(f"ridge@{f:.0%}", model.predict(Xte), yte, res_r)
        row_r["alpha"] = alpha

        # --- MLP head, 5 seeds, mean prediction ---
        preds = [mlp_head(Xs, ys, Xte, s)[0] for s in MLP_SEEDS]
        res_m: dict = {}
        row_m = score(f"mlp@{f:.0%}", np.mean(preds, axis=0), yte, res_m)
        # per-seed Spearman, recorded so the MLP control tolerance below is measured
        # rather than asserted: re-running the same 5-seed ensemble moves the mean
        # only within the scatter the seeds themselves show.
        seed_sp = [float(spearmanr(p, yte).statistic) for p in preds]

        rows.append({
            "fraction": f, "n_train": sizes[f],
            "ridge_spearman": row_r["spearman"],
            "ridge_ci95": row_r["ci95"], "ridge_rmse": row_r["rmse"],
            "ridge_alpha": alpha,
            "mlp_spearman": row_m["spearman"],
            "mlp_ci95": row_m["ci95"], "mlp_rmse": row_m["rmse"],
            "mlp_seed_sd": round(float(np.std(seed_sp, ddof=1)), 4),
            "mlp_seed_spread": round(max(seed_sp) - min(seed_sp), 4),
        })

    # --- control: the 100% row must reproduce metrics.json ---
    last = rows[-1]
    # The MLP line keeps a tolerance (torch on GPU is not bit-reproducible), but the
    # tolerance is no longer a bare constant: it is 2 x the measured seed scatter at
    # the full-data fraction, capped by MLP_TOLERANCE and floored by
    # MIN_MLP_TOLERANCE. That matters because the headline ridge-minus-MLP gap at
    # 100% is itself small; a hand-set 0.01 allowance was nearly as large as the
    # difference the README reports, so the gate could have passed while the claim
    # was already swallowed. The derived value and the gap are both recorded.
    seed_sd_100 = last["mlp_seed_sd"]
    tol_mlp = min(MLP_TOLERANCE, max(2 * seed_sd_100, MIN_MLP_TOLERANCE))
    mlp_dev = abs(metrics["mlp_mean5seed"]["spearman"] - last["mlp_spearman"])
    gap_100 = last["ridge_spearman"] - last["mlp_spearman"]
    ctrl = {
        "ridge_expected": metrics["ridge"]["spearman"],
        "ridge_at_100pct": last["ridge_spearman"],
        "mlp_expected": metrics["mlp_mean5seed"]["spearman"],
        "mlp_at_100pct": last["mlp_spearman"],
        "mlp_seed_sd_at_100pct": seed_sd_100,
        "mlp_seed_spread_at_100pct": last["mlp_seed_spread"],
        "mlp_tolerance_used": round(tol_mlp, 4),
        "mlp_tolerance_rule": ("min(0.01, max(2 * seed sd at 100%, 0.002)) -- derived "
                               "from measured seed scatter, not hand-set"),
        "mlp_deviation": round(mlp_dev, 4),
        "headline_ridge_minus_mlp_at_100pct": round(gap_100, 4),
        # the point of deriving the tolerance: the gate must stay far smaller than the
        # effect the README reports, otherwise passing it would say nothing.
        "tolerance_vs_headline_gap": round(tol_mlp / gap_100, 4) if gap_100 else None,
    }
    ctrl["ridge_ok"] = abs(ctrl["ridge_expected"] - ctrl["ridge_at_100pct"]) < 1e-4
    ctrl["mlp_ok"] = abs(ctrl["mlp_expected"] - ctrl["mlp_at_100pct"]) <= tol_mlp
    ctrl["pass"] = ctrl["ridge_ok"] and ctrl["mlp_ok"]
    print("\ncontrol vs metrics.json:", json.dumps(ctrl, indent=2))
    if not ctrl["pass"]:
        raise SystemExit("CONTROL FAILED - curve not written; "
                         "protocol drifted from analyze.py")

    out = {
        "protocol": {
            "fractions": FRACTIONS,
            "subsets": f"nested, first round(f*n) of one permutation of rng seed {SUBSET_SEED}",
            "ridge": f"alpha re-selected on valid over {list(RIDGE_ALPHAS)} at every fraction",
            "mlp": f"mean of {MLP_SEEDS} seeds, analyze.mlp_head defaults",
            "ci": "1000-resample bootstrap percentile CI, seed 0 (analyze.score)",
            "control": ("100% row vs metrics.json; ridge exact (<1e-4); mlp within "
                        "min(0.01, max(2 * measured seed sd at 100%, 0.002)) -- see "
                        "control.mlp_tolerance_used"),
        },
        "control": ctrl,
        "curve": rows,
    }
    (OUT / "learning_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)

    # --- figure ---
    xs = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for key, label, color in [("ridge", "ridge (alpha re-tuned)", "#1B5E93"),
                              ("mlp", "MLP head, 5-seed mean", "#B3541E")]:
        y = [r[f"{key}_spearman"] for r in rows]
        lo = [r[f"{key}_spearman"] - r[f"{key}_ci95"][0] for r in rows]
        hi = [r[f"{key}_ci95"][1] - r[f"{key}_spearman"] for r in rows]
        ax.plot(xs, y, marker="o", color=color, label=label)
        ax.fill_between(xs, np.array(y) - np.array(lo), np.array(y) + np.array(hi),
                        color=color, alpha=0.18, label=None)
    ax.axvline(n, ls=":", lw=1, color="grey")
    ax.set_xlabel("training variants (nested subsets, seed 42)")
    ax.set_ylabel("Spearman rho on FLU-S test split")
    ax.set_title("ESM2 frozen embeddings: label-count learning curve\n(band = 95% bootstrap CI, 1000 resamples)")
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig5_learning_curve.png")
    print(f"\nwrote {OUT/'learning_curve.json'}, {OUT/'learning_curve.csv'}, {FIGS/'fig5_learning_curve.png'}")


if __name__ == "__main__":
    main()
