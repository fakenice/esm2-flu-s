"""Step 3: zero-shot vs fine-tuned, with baselines, seeds and confidence intervals.

Inputs (from src/embed.py):  results/embed/{split}_emb.npy, {split}_score_wtmarginal.npy
Outputs: results/metrics.json, results/report.md

Models compared on the TAPE fluorescence test split (25,517 variants, length 237):
  null_mean      predict the training mean for every variant
  null_nmut      predict -(number of mutations) -- "more mutations, less fluorescence"
  zero_shot      ESM2 WT-marginal log-likelihood ratio, no training at all
  ridge          linear head on frozen ESM2 mean-pooled embeddings
  mlp            ~~~~ 2-layer head, 5 seeds ~~~~

No number here is hard-coded: every figure is computed from the arrays on disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB = ROOT / "results" / "embed"
OUT = ROOT / "results"


def load(split: str):
    df = pd.read_csv(PROC / f"flu_{split}.csv")
    emb = np.load(EMB / f"{split}_emb.npy")
    zs = np.load(EMB / f"{split}_score_wtmarginal.npy")
    assert len(df) == len(emb) == len(zs), f"{split}: length mismatch"
    return df, emb, zs


def boot_spearman(pred, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = spearmanr(pred[idx], y[idx]).statistic
        if np.isfinite(r):
            vals.append(r)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(name, pred, y, out: dict):
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    if np.std(pred) == 0:
        # a constant prediction has no rank correlation at all -- that is a fact about
        # the baseline, not a number to paper over
        out[name] = {"spearman": None, "note": "constant prediction; Spearman undefined",
                     "rmse": round(rmse, 4), "n": int(len(y))}
        print(f"{name:12s} spearman=undefined (constant)  rmse={rmse:.4f}")
        return out[name]
    rho = spearmanr(pred, y).statistic
    lo, hi = boot_spearman(pred, y)
    out[name] = {"spearman": round(float(rho), 4), "ci95": [round(lo, 4), round(hi, 4)],
                 "rmse": round(rmse, 4), "n": int(len(y))}
    print(f"{name:12s} spearman={rho:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  rmse={rmse:.4f}")
    return out[name]


def mlp_head(Xtr, ytr, Xte, seed, epochs=300, hidden=256, lr=1e-3, wd=1e-4):
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.MSELoss()
    xb = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yb = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
    t0 = time.time()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xb), yb)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy().ravel()
    return pred, {"seconds": round(time.time() - t0, 1),
                  "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 0) if dev == "cuda" else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, Xtr, zs_tr = load("train")
    va, Xva, zs_va = load("valid")
    te, Xte, zs_te = load("test")
    ytr, yva, yte = tr["log_fluor"].values, va["log_fluor"].values, te["log_fluor"].values

    res: dict = {"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                 "ci_protocol": {"method": "bootstrap percentile 95% CI",
                                 "n_boot": 1000, "rng_seed": 0,
                                 "note": "same protocol for headline, subset and per-stratum CIs"}}
    print(f"train {len(tr)}  valid {len(va)}  test {len(te)}  dim={Xtr.shape[1]}")

    # ---- baselines ----------------------------------------------------------
    score("null_mean", np.full_like(yte, ytr.mean()), yte, res)
    score("null_nmut", -te["num_mutations"].values.astype(float), yte, res)

    # ---- zero-shot ---------------------------------------------------------
    score("zero_shot", zs_te, yte, res)

    # ---- ridge on frozen embeddings ----------------------------------------
    best = None
    for alpha in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0):
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        r = spearmanr(m.predict(Xva), yva).statistic
        print(f"  ridge alpha={alpha:<8} valid spearman={r:+.4f}")
        if best is None or r > best[1]:
            best = (alpha, r, m)
    alpha, r_valid, model = best
    res["ridge_alpha"] = alpha
    res["ridge_valid_spearman"] = round(float(r_valid), 4)
    score("ridge", model.predict(Xte), yte, res)

    # ---- MLP head, 5 seeds --------------------------------------------------
    seeds = [0, 1, 2, 3, 4]
    preds, infos = [], []
    for s in seeds:
        p, info = mlp_head(Xtr, ytr, Xte, s)
        preds.append(p)
        infos.append(info)
        score(f"mlp_seed{s}", p, yte, res)
    pm = np.mean(preds, 0)
    score("mlp_mean5seed", pm, yte, res)
    res["mlp_ensemble"] = {"seeds": seeds, "per_seed_spearman": [res[f"mlp_seed{s}"]["spearman"] for s in seeds],
                           "train_seconds": [i["seconds"] for i in infos],
                           "peak_gpu_MiB": [i["peak_gpu_MiB"] for i in infos]}
    spread = max(res[f"mlp_seed{s}"]["spearman"] for s in seeds) - min(res[f"mlp_seed{s}"]["spearman"] for s in seeds)
    res["mlp_ensemble"]["seed_spread"] = round(float(spread), 4)

    # ---- where zero-shot fails: by mutation count, with per-stratum CIs -----
    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 15)]
    ridge_te = model.predict(Xte)
    rows = []
    for lo, hi in bins:
        m = (te["num_mutations"] >= lo) & (te["num_mutations"] <= hi)
        if m.sum() < 30:
            continue
        row = {"mutations": f"{lo}-{hi}", "n": int(m.sum())}
        for key, pred in [("zero_shot", zs_te[m]), ("ridge", ridge_te[m]), ("mlp", pm[m])]:
            rho = float(spearmanr(pred, yte[m]).statistic)
            c_lo, c_hi = boot_spearman(pred, yte[m])
            row[key] = round(rho, 4)
            row[f"{key}_ci95"] = [round(c_lo, 4), round(c_hi, 4)]
        rows.append(row)
    res["by_mutation_count"] = rows
    print("\nby mutation count (95% bootstrap CI):")
    for r in rows:
        print(f"  {r['mutations']:>6} n={r['n']:5d}  zs={r['zero_shot']:+.3f}{r['zero_shot_ci95']}  "
              f"ridge={r['ridge']:+.3f}{r['ridge_ci95']}  mlp={r['mlp']:+.3f}{r['mlp_ci95']}")

    # ---- zero-shot on a common subset, two scoring conventions -------------
    masked_path = EMB / "test_score_maskedmarginal_subset.npy"
    idx_path = EMB / "test_masked_subset_idx.npy"
    if masked_path.exists() and idx_path.exists():
        idx = np.load(idx_path)
        zs_masked = np.load(masked_path)
        sub_y = yte[idx]
        sub_zs = zs_te[idx]
        res["zero_shot_subset"] = {"n": int(len(idx))}
        score("zero_shot_wt_subset", sub_zs, sub_y, res)
        score("zero_shot_masked_subset", zs_masked, sub_y, res)
        # ridge / mlp restricted to the same rows, so the comparison is fair
        score("ridge_subset", model.predict(Xte)[idx], sub_y, res)
        score("mlp_subset", pm[idx], sub_y, res)
        res["zero_shot_subset"]["wt_vs_masked_spearman_between_scorers"] = round(
            float(spearmanr(sub_zs, zs_masked).statistic), 4)

    (OUT / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---- markdown report ---------------------------------------------------
    lines = [
        "# ESM2 on TAPE fluorescence (FLU-S): zero-shot vs fine-tuned", "",
        f"- train {len(tr)} / valid {len(va)} / test {len(te)} variants, length 237, wild type = consensus",
        f"- test split is the extrapolation split: mean {te['num_mutations'].mean():.2f} mutations, max {int(te['num_mutations'].max())}",
        "- CI protocol: 95% bootstrap percentile, 1000 resamples, rng seed 0 — identical protocol in the "
        "headline table, the subset table and the per-stratum table below",
        "", "| model | Spearman | 95% CI (bootstrap, 1000) | RMSE | n |", "|---|---|---|---|---|",
    ]
    order = ["null_mean", "null_nmut", "zero_shot", "ridge", "mlp_mean5seed"]
    for k in order:
        v = res[k]
        if v.get("spearman") is None:
            lines.append(f"| {k} | undefined (constant) | — | {v['rmse']:.4f} | {v['n']} |")
        else:
            lines.append(f"| {k} | {v['spearman']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['rmse']:.4f} | {v['n']} |")
    lines += ["", f"MLP: {len(seeds)} seeds, per-seed Spearman {res['mlp_ensemble']['per_seed_spearman']}, "
                  f"seed spread {res['mlp_ensemble']['seed_spread']}", ""]

    if "zero_shot_subset" in res:
        lines += ["## Two zero-shot scoring conventions, on the same 5,000-variant subset", "",
                  "| scorer | Spearman | 95% CI |", "|---|---|---|"]
        for k, label in [("zero_shot_wt_subset", "WT-marginal (biased: mutant is its own context)"),
                         ("zero_shot_masked_subset", "masked-marginal (position masked before scoring)"),
                         ("ridge_subset", "ridge on frozen embeddings (reference)"),
                         ("mlp_subset", "MLP 5-seed mean (reference)")]:
            v = res[k]
            lines.append(f"| {label} | {v['spearman']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
        lines += ["", f"Correlation between the two zero-shot scores themselves: "
                      f"{res['zero_shot_subset']['wt_vs_masked_spearman_between_scorers']:+.4f}", ""]

    lines += ["## Spearman by number of mutations in the variant (95% bootstrap CI in brackets)", "",
              "| mutations | n | zero-shot | ridge | mlp |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mutations']} | {r['n']} | "
                     f"{r['zero_shot']:+.3f} [{r['zero_shot_ci95'][0]:+.3f}, {r['zero_shot_ci95'][1]:+.3f}] | "
                     f"{r['ridge']:+.3f} [{r['ridge_ci95'][0]:+.3f}, {r['ridge_ci95'][1]:+.3f}] | "
                     f"{r['mlp']:+.3f} [{r['mlp_ci95'][0]:+.3f}, {r['mlp_ci95'][1]:+.3f}] |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT/'metrics.json'} and {OUT/'report.md'}")


if __name__ == "__main__":
    main()
