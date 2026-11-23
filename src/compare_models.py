"""E3: cross-model benchmark - the same head protocol on four frozen encoders.

For every model in src/embed.py's registry (esm2_650M, esm2_35M,
saprot_35m_seqonly, protbert_bfd) we re-run the three scoring arms of the
main report on that model's own embeddings/zero-shot scores, with the
identical protocol imported from analyze.py (same ridge alphas, same 5 MLP
seeds, same bootstrap-CI function):

  zero_shot  WT-marginal log-likelihood ratio, no training
  ridge      ridge regression on frozen mean-pooled embeddings (alpha on valid)
  mlp        2-layer head, 5 seeds, mean of the seed predictions

Deterministic given the cached embeddings: no model forward pass here, only
sklearn/torch heads on stored .npy files. The esm2_650M row is asserted
against results/metrics.json so the comparison table can never drift from
the headline report.

Outputs: results/model_comparison.{csv,json,md}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from analyze import RIDGE_ALPHAS, MLP_SEEDS, PROC, OUT, boot_spearman, mlp_head
from embed import MODELS

EMB = Path(__file__).resolve().parents[1] / "results" / "embed"
ORDER = ["esm2_650M", "esm2_35M", "saprot_35m_seqonly", "protbert_bfd"]


def load_split(split: str, model: str):
    df = pd.read_csv(PROC / f"flu_{split}.csv")
    d = EMB / model
    emb = np.load(d / f"{split}_emb.npy")
    zs = np.load(d / f"{split}_score_wtmarginal.npy")
    assert len(df) == len(emb) == len(zs), f"{model}/{split}: length mismatch"
    return df, emb, zs


def model_cost(model: str) -> dict:
    tj = json.loads((EMB / model / "timing_embed.json").read_text(encoding="utf-8"))
    splits = tj.get("splits", tj)  # 650M predates the "splits" nesting
    entries = {k: v for k, v in splits.items() if isinstance(v, dict)}
    secs = {k: v["seconds"] for k, v in entries.items()}
    return {"params_M": tj.get("params_M"),
            "embed_seconds_total": round(sum(secs.values()), 1),
            "peak_gpu_MiB": max((v["peak_gpu_MiB"] for v in entries.values()
                                 if v.get("peak_gpu_MiB")), default=None),
            "family": tj.get("family", "esm"),
            "upstream": tj.get("upstream", MODELS[model]["upstream"])}


def main() -> None:
    res: dict = {"protocol": {"ridge_alphas": list(RIDGE_ALPHAS), "mlp_seeds": MLP_SEEDS,
                              "ci": "bootstrap percentile 95%, 1000 resamples, rng seed 0"},
                 "models": {}}
    rows = []
    base = None
    for model in ORDER:
        tr, Xtr, zs_tr = load_split("train", model)
        va, Xva, zs_va = load_split("valid", model)
        te, Xte, zs_te = load_split("test", model)
        ytr, yva, yte = tr["log_fluor"].values, va["log_fluor"].values, te["log_fluor"].values
        m: dict = {"n_test": len(te), "emb_dim": int(Xtr.shape[1]), **model_cost(model)}

        r0 = spearmanr(zs_te, yte).statistic
        lo, hi = boot_spearman(zs_te, yte)
        m["zero_shot"] = {"spearman": round(float(r0), 4), "ci95": [round(lo, 4), round(hi, 4)]}

        best = None
        for alpha in RIDGE_ALPHAS:
            mm = Ridge(alpha=alpha).fit(Xtr, ytr)
            r = spearmanr(mm.predict(Xva), yva).statistic
            if best is None or r > best[1]:
                best = (alpha, r, mm)
        alpha, r_valid, head = best
        p = head.predict(Xte)
        r = spearmanr(p, yte).statistic
        lo, hi = boot_spearman(p, yte)
        m["ridge"] = {"alpha": alpha, "valid_spearman": round(float(r_valid), 4),
                      "spearman": round(float(r), 4), "ci95": [round(lo, 4), round(hi, 4)]}

        preds = []
        for s in MLP_SEEDS:
            pr, _ = mlp_head(Xtr, ytr, Xte, s)
            preds.append(pr)
        pm = np.mean(preds, 0)
        r = spearmanr(pm, yte).statistic
        lo, hi = boot_spearman(pm, yte)
        m["mlp"] = {"seeds": MLP_SEEDS,
                    "per_seed_spearman": [round(float(spearmanr(pr, yte).statistic), 4) for pr in preds],
                    "spearman": round(float(r), 4), "ci95": [round(lo, 4), round(hi, 4)]}

        res["models"][model] = m
        rows.append({"model": model, "upstream": m["upstream"], "family": m["family"],
                     "emb_dim": m["emb_dim"], "embed_seconds_total": m["embed_seconds_total"],
                     "zero_shot": m["zero_shot"]["spearman"],
                     "ridge": m["ridge"]["spearman"], "ridge_alpha": m["ridge"]["alpha"],
                     "mlp": m["mlp"]["spearman"]})
        print(f"{model:18s} zs={m['zero_shot']['spearman']:+.4f} "
              f"ridge={m['ridge']['spearman']:+.4f}(a={alpha}) mlp={m['mlp']['spearman']:+.4f}", flush=True)

    # drift assert: the 650M row must equal the published headline numbers
    main_metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    m6 = res["models"]["esm2_650M"]
    assert m6["zero_shot"]["spearman"] == main_metrics["zero_shot"]["spearman"]
    assert m6["ridge"]["spearman"] == main_metrics["ridge"]["spearman"]
    assert m6["ridge"]["alpha"] == main_metrics["ridge_alpha"]
    assert m6["mlp"]["spearman"] == main_metrics["mlp_mean5seed"]["spearman"]

    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False, lineterminator="\n")
    (OUT / "model_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    md = ["# E3 PLM 横评（同一头协议 × 四个冻结编码器）", "",
          f"- 任务/切分/头部/CI 协议与主报告完全一致（ridge 同 {len(RIDGE_ALPHAS)} 点 alpha 网格、"
          f"MLP 同 {len(MLP_SEEDS)} seeds 取均值、bootstrap 95% CI 1000 次 seed 0）；"
          "模型间唯一变量是冻结嵌入与 WT-marginal 分数本身。",
          "- 本表不含 masked-marginal 零样本口径（主报告 5,000 条子样本口径仅对 esm2_650M 计算过）。", "",
          "| 模型 | 参数量级 | 维度 | 嵌入总耗时(s) | zero-shot ρ [95%CI] | ridge ρ [95%CI] | MLP(5 seed) ρ [95%CI] |",
          "|---|---|---|---|---|---|---|"]
    # 650M predates the params_M field in timing_embed.json; its checkpoint name fixes it
    fallback = {"esm2_650M": 650.0}
    def params_txt(model):
        v = res["models"][model]["params_M"]
        return f"{v if v is not None else fallback[model]:.1f}M"
    for model in ORDER:
        m = res["models"][model]
        md.append(f"| `{model}` | {params_txt(model)} | {m['emb_dim']} | {m['embed_seconds_total']:.0f} | "
                  f"{m['zero_shot']['spearman']:+.4f} [{m['zero_shot']['ci95'][0]:+.4f}, {m['zero_shot']['ci95'][1]:+.4f}] | "
                  f"{m['ridge']['spearman']:+.4f} [{m['ridge']['ci95'][0]:+.4f}, {m['ridge']['ci95'][1]:+.4f}] | "
                  f"{m['mlp']['spearman']:+.4f} [{m['mlp']['ci95'][0]:+.4f}, {m['mlp']['ci95'][1]:+.4f}] |")
    ridge_sorted = sorted(ORDER, key=lambda k: -res["models"][k]["ridge"]["spearman"])
    zs_sorted = sorted(ORDER, key=lambda k: -res["models"][k]["zero_shot"]["spearman"])
    md += ["", "## 判读（数据驱动，措辞随重跑自动更新）", "",
           f"- 线性头排序（ridge）：{' > '.join(ridge_sorted)}。"
           "这是点估计排序；同一测试集上名次差异应做配对差值检验，不能用边际 CI 重叠代替。",
           f"- 零样本 WT-marginal 排序：{' > '.join(zs_sorted)}；所有模型同一口径（Σ log p(mt)−log p(wt)，无掩蔽），"
           "该口径自带自洽偏差，主报告已量化（masked-marginal 与其 ρ 仅 +0.309）。"]
    for model in ORDER:
        m = res["models"][model]
        if m["zero_shot"]["spearman"] < 0:
            lo, hi = m["zero_shot"]["ci95"]
            verdict = ("CI 完全在 0 以下 ⇒ 该口径下排序方向系统性反转（不是抽样噪声）"
                       if hi < 0 else
                       "CI 跨 0 ⇒ 方向未定，不能称其为反向信号")
            md.append(f"- `{model}` 零样本为负（{m['zero_shot']['spearman']:+.4f}，CI "
                      f"[{lo:+.4f}, {hi:+.4f}]）：{verdict}。")
    ci_overlap = [f"{a} vs {b}" for i, a in enumerate(ORDER) for b in ORDER[i + 1:]
                  if not (res["models"][a]["ridge"]["ci95"][1] < res["models"][b]["ridge"]["ci95"][0]
                          or res["models"][b]["ridge"]["ci95"][1] < res["models"][a]["ridge"]["ci95"][0])]
    md += ["", "## 口径与限制", "",
           "- SaProt 用的是作者发布的 seqOnly 变体（训练时 3Di 槽全为 `#`，模型卡：用于与 ESM-2 35M 对比）；"
           "普通 SaProt 35M/650M 的冻结嵌入按官方 README 需要真实 3Di 结构 token，本基准不提供结构输入。",
           "- ProtBERT 的输入 id 按发布 vocab.txt 行号手工构造（transformers 5 的 BertTokenizer 无 tokenizer.json 可复现路径，"
           "见 embed.py `ManualBertTok` 注释）；每残基一 token、[CLS]/[SEP] 位置与官方用法一致。",
           "- 单一蛋白（avGFP）、单一任务（FLU-S）、n=25,517：这是「同一协议下编码器快照的差别」，"
           "不是对模型通用强弱的一般结论。",
           f"- 以下模型对的 ridge 边际 CI 重叠（尚不能据此判定排序差异）：{'；'.join(ci_overlap)}" if ci_overlap
           else "- 各模型 ridge CI 两两不重叠。",
           "- 复跑：`python src/embed.py --model <tag>`（需 GPU）后 `python src/compare_models.py`（仅 CPU+sklearn/torch 头）。"]
    (OUT / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("\nwrote results/model_comparison.{csv,json,md}")


if __name__ == "__main__":
    main()
