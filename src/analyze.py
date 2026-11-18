"""Step 3: zero-shot vs fine-tuned, with baselines, seeds and confidence intervals.

Inputs (from src/embed.py):  results/embed/esm2_650M/{split}_emb.npy, {split}_score_wtmarginal.npy
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
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB = ROOT / "results" / "embed" / "esm2_650M"
OUT = ROOT / "results"

# shared with src/compare_models.py so the cross-model benchmark uses the
# identical head protocol (same alphas, same seeds) as the main report
RIDGE_ALPHAS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0, 10000.0)
MLP_SEEDS = [0, 1, 2, 3, 4]


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


MODEL_ORDER = ["zero_shot", "ridge", "mlp"]


def boot_rho_matrix(preds: dict, y, n_boot=1000, seed=0) -> np.ndarray:
    """[n_boot, 3] bootstrap Spearman for the three models under one shared
    index draw per replicate: within-stratum deltas stay paired (same variants),
    and deltas across strata stay independent (each stratum draws its own seed-0
    stream over disjoint variants)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty((n_boot, len(MODEL_ORDER)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for j, k in enumerate(MODEL_ORDER):
            out[b, j] = spearmanr(preds[k][idx], yb).statistic
    return out


def fisher_z_two_rhos(r1: float, n1: int, r2: float, n2: int):
    """Approximate two-sided test that two INDEPENDENT correlations differ
    (disjoint strata). Fisher r-to-z with the standard 1/(n-3) SE applied to
    Spearman is an approximation; it is reported only as a cross-check of the
    bootstrap CI, which carries the inference."""
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (math.atanh(r1) - math.atanh(r2)) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


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
    for alpha in RIDGE_ALPHAS:
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
    seeds = MLP_SEEDS
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
    strata_arrays: dict[str, dict] = {}
    for lo, hi in bins:
        m = (te["num_mutations"] >= lo) & (te["num_mutations"] <= hi)
        if m.sum() < 30:
            continue
        label = f"{lo}-{hi}"
        row = {"mutations": label, "n": int(m.sum())}
        strata_arrays[label] = {"zero_shot": zs_te[m], "ridge": ridge_te[m],
                                "mlp": pm[m], "y": yte[m]}
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

    # ---- E4: is the between-stratum rho difference itself significant? ------
    # The strata are disjoint variant sets, so between-stratum deltas compare
    # INDEPENDENT correlations (bootstrap: one seed-0 resample stream per stratum;
    # cross-check: Fisher r-to-z). Within a stratum the models share variants and
    # the target, so those deltas are DEPENDENT correlations and must be
    # bootstrapped paired (same index draw across models) -- the nonparametric
    # counterpart of a Steiger-type test.
    labels = [r["mutations"] for r in rows]
    rho_boot = {lab: boot_rho_matrix(strata_arrays[lab], strata_arrays[lab]["y"])
                for lab in labels}
    row_by_lab = {r["mutations"]: r for r in rows}

    between = []
    pairs = [(labels[0], labels[1]), (labels[1], labels[2]), (labels[0], labels[2])] \
        if len(labels) == 3 else list(zip(labels, labels[1:]))
    for lab_a, lab_b in pairs:
        for model_key in MODEL_ORDER:
            j = MODEL_ORDER.index(model_key)
            d = rho_boot[lab_a][:, j] - rho_boot[lab_b][:, j]
            lo_c, hi_c = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            rho_a, rho_b = row_by_lab[lab_a][model_key], row_by_lab[lab_b][model_key]
            z, p = fisher_z_two_rhos(rho_a, row_by_lab[lab_a]["n"],
                                     rho_b, row_by_lab[lab_b]["n"])
            between.append({
                "model": model_key, "pair": f"{lab_a} vs {lab_b}",
                "rho_a": rho_a, "rho_b": rho_b,
                "delta": round(rho_a - rho_b, 4),
                "delta_ci95": [round(lo_c, 4), round(hi_c, 4)],
                "significant_ci": bool(hi_c < 0 or lo_c > 0),
                "fisher_z": round(z, 3), "p_raw": p,
                "p_bonferroni": min(1.0, p * len(pairs) * len(MODEL_ORDER)),
                # a delta counts as a finding only if BOTH calibers agree
                "significant": bool((hi_c < 0 or lo_c > 0) and min(1.0, p * len(pairs) * len(MODEL_ORDER)) < 0.05),
            })

    within = []
    for lab in labels:
        for ka, kb in [("ridge", "zero_shot"), ("ridge", "mlp")]:
            ia, ib = MODEL_ORDER.index(ka), MODEL_ORDER.index(kb)
            d = rho_boot[lab][:, ia] - rho_boot[lab][:, ib]
            lo_c, hi_c = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            within.append({
                "stratum": lab, "pair": f"{ka} - {kb}",
                "delta": round(row_by_lab[lab][ka] - row_by_lab[lab][kb], 4),
                "delta_ci95": [round(lo_c, 4), round(hi_c, 4)],
                "distinguishable": bool(hi_c < 0 or lo_c > 0),
            })

    # power ceiling of the small stratum, on the Fisher-z scale (80% power,
    # two-sided alpha 0.05): MDE ~ (1.96+0.84)*SE -- what the data CANNOT detect
    mde_small_vs_first = None
    if len(labels) >= 3:
        n_s, n_f = row_by_lab[labels[-1]]["n"], row_by_lab[labels[0]]["n"]
        mde_small_vs_first = round(2.8 * math.sqrt(1.0 / (n_s - 3) + 1.0 / (n_f - 3)), 3)

    readout = []
    for b in between:
        label_pair = b["pair"].replace(" vs ", ")−ρ(")
        prefix = f"{b['model']}: ρ({label_pair}) = {b['delta']:+.3f}"
        if b["significant"]:
            direction = ""
            if b["model"] == "zero_shot":
                direction = "（负信号随外推减弱，朝 0 靠拢）" if b["delta"] < 0 and b["rho_a"] < 0 \
                    else "（随突变数增多向 0 靠拢）" if b["delta"] > 0 else ""
            readout.append(f"{prefix}，CI 不含 0 且 Bonferroni 后 p<0.05 ⇒ 差异显著{direction}")
        elif b["significant_ci"]:
            readout.append(f"{prefix}，CI 勉强不含 0 但 Bonferroni 后 p="
                           f"{b['p_bonferroni']:.2f} ⇒ 边缘结果，不作为结论")
        else:
            readout.append(f"{prefix}，CI 跨 0 ⇒ 差异不可区分")
    res["strata_tests"] = {
        "protocol": {
            "primary": "95% bootstrap percentile CI on the delta, 1000 resamples, seed 0 "
                       "(independent streams per disjoint stratum; paired index draw within a stratum)",
            "cross_check": "Fisher r-to-z for independent correlations, SE = sqrt(1/(n1-3)+1/(n2-3)); "
                           "applied to Spearman as an approximation; Bonferroni over the between-stratum family",
            "note": "bootstrap CI is the inference; the z-test only corroborates it",
        },
        "between_strata": between,
        "within_stratum": within,
        "mde_fisher_z_scale_small_vs_first_stratum": mde_small_vs_first,
        "readout": readout,
    }
    print("\nbetween-stratum deltas (bootstrap CI + Fisher z cross-check):")
    for b in between:
        print(f"  {b['model']:>9} {b['pair']:>16}: Δ={b['delta']:+.4f} "
              f"CI[{b['delta_ci95'][0]:+.4f},{b['delta_ci95'][1]:+.4f}] "
              f"z={b['fisher_z']:+.2f} p={b['p_raw']:.3g} -> {'显著' if b['significant'] else '不可区分'}")
    print("within-stratum deltas (paired bootstrap):")
    for w in within:
        print(f"  {w['stratum']:>6} {w['pair']:>18}: Δ={w['delta']:+.4f} "
              f"CI[{w['delta_ci95'][0]:+.4f},{w['delta_ci95'][1]:+.4f}] "
              f"-> {'可区分' if w['distinguishable'] else '不可区分'}")

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

    st = res["strata_tests"]
    lines += ["", "## 分层 ρ 差异的显著性检验（E4）", "",
              f"主推断：Δρ 的 95% bootstrap 百分位 CI（{st['protocol']['primary']}）；"
              "交叉核对：Fisher r-to-z（独立分层适用，Spearman 上为近似）。", ""]
    if st["mde_fisher_z_scale_small_vs_first_stratum"] is not None:
        lines += [f"最小可检差（末层 vs 首层，Fisher-z 尺度，80% 功效）≈ "
                  f"{st['mde_fisher_z_scale_small_vs_first_stratum']}——小样本层「检不出」不等于「无差异」。", ""]
    lines += ["| 模型 | 层对比 | Δρ | Δρ 95% CI | Fisher z | p（原始/Bonferroni） | 判定 |", "|---|---|---|---|---|---|---|"]
    for b in st["between_strata"]:
        p_raw = "<1e-300" if b["p_raw"] < 1e-300 else f"{b['p_raw']:.3g}"
        p_bon = "<1e-300" if b["p_bonferroni"] < 1e-300 else f"{b['p_bonferroni']:.3g}"
        verdict = ("显著" if b["significant"] else
                   "边缘（CI 不含 0 但校正后 p≥0.05）" if b["significant_ci"] else "不可区分")
        lines.append(f"| {b['model']} | {b['pair']} | {b['delta']:+.3f} | "
                     f"[{b['delta_ci95'][0]:+.3f}, {b['delta_ci95'][1]:+.3f}] | "
                     f"{b['fisher_z']:+.2f} | {p_raw} / {p_bon} | {verdict} |")
    lines += ["", "同层内（同一批变体、相关不独立，配对 bootstrap）：", "",
              "| 层 | 对比 | Δρ | 95% CI | 判定 |", "|---|---|---|---|---|"]
    for w in st["within_stratum"]:
        lines.append(f"| {w['stratum']} | {w['pair']} | {w['delta']:+.3f} | "
                     f"[{w['delta_ci95'][0]:+.3f}, {w['delta_ci95'][1]:+.3f}] | "
                     f"{'可区分' if w['distinguishable'] else '不可区分'} |")
    lines += ["", "判读：", ""] + [f"- {t}" for t in st["readout"]]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT/'metrics.json'} and {OUT/'report.md'}")


if __name__ == "__main__":
    main()
