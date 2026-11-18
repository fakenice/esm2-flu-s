"""Step 2b: masked-marginal zero-shot scoring (the unbiased variant).

For each mutation (i, wt, mt) of a variant, mask position i and score
    log p(mt | sequence with i masked) - log p(wt | sequence with i masked)
This removes the self-consistency bias of WT-marginal scoring, where the mutant
sequence is used as its own context and the score is almost always positive.

Cost is one forward pass per mutation, so the split is sampled:
    python src/zeroshot_masked.py --split test --variants 5000 --batch 32
The sample is drawn with a fixed seed and the indices are saved, so the number is
reproducible and the analysis can be re-run on the same subset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EMB = ROOT / "results" / "embed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "esm2_t33_650M_UR50D"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(PROC / f"flu_{args.split}.csv")
    wt = json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))["wt_sequence"]

    rng = np.random.default_rng(args.seed)
    n = min(args.variants, len(df))
    idx = np.sort(rng.choice(len(df), n, replace=False))
    sub = df.iloc[idx].reset_index(drop=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(dev).eval()
    mask_id = tok.mask_token_id
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"}

    # build one masked sequence per mutation
    seqs, tasks = [], []
    for row_i, s in enumerate(sub["seq"]):
        for pos, (a, b) in enumerate(zip(wt, s)):
            if a != b:
                seqs.append(s[:pos] + "<mask>" + s[pos + 1:])
                tasks.append((row_i, pos, a, b))
    print(f"{n} variants -> {len(seqs)} masked sequences")

    scores = np.zeros(n, dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(seqs), args.batch):
            chunk = seqs[start:start + args.batch]
            meta = tasks[start:start + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits
            # the mask token sits at position p+1 in the tokenised sequence
            bidx = torch.arange(len(chunk), device=dev)
            pidx = torch.tensor([m[1] + 1 for m in meta], device=dev)
            logp = torch.log_softmax(logits[bidx, pidx].float(), dim=-1)
            mt = torch.tensor([aa_ids[m[3]] for m in meta], device=dev)
            wtv = torch.tensor([aa_ids[m[2]] for m in meta], device=dev)
            d = (logp.gather(1, mt[:, None]) - logp.gather(1, wtv[:, None])).squeeze(1)
            for (row_i, *_), v in zip(meta, d.cpu().numpy()):
                scores[row_i] += float(v)
            if start % (args.batch * 100) == 0:
                el = time.time() - t0
                print(f"  {min(start + args.batch, len(seqs))}/{len(seqs)}  "
                      f"{min(start + args.batch, len(seqs)) / max(el, 1e-9):.1f} mut/s", flush=True)

    np.save(EMB / f"{args.split}_score_maskedmarginal_subset.npy", scores)
    np.save(EMB / f"{args.split}_masked_subset_idx.npy", idx)
    info = {
        "split": args.split, "variants": int(n), "masked_sequences": len(seqs),
        "seed": args.seed, "batch": args.batch,
        "seconds": round(time.time() - t0, 1),
        "mut_per_s": round(len(seqs) / (time.time() - t0), 1),
        "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 0) if dev == "cuda" else None,
    }
    (EMB / f"{args.split}_masked_timing.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
