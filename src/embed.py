"""Step 2: one forward pass per variant -> mean-pooled embedding + WT-marginal zero-shot score.

WT-marginal score (one pass per variant, no masking):
    s = sum_i [ log p(mt_i | variant) - log p(wt_i | variant) ]

Outputs (per split): embeddings .npy, scores .npy, and a timing/memory log.
Everything is resumable at the shard level: each shard is written once and skipped on rerun.
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
OUT = ROOT / "results" / "embed"
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_model(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
    return tok, model


@torch.no_grad()
def run_split(name: str, tok, model, device: str, batch: int, wt: str, manifest: dict, limit: int = 0):
    df = pd.read_csv(PROC / f"flu_{name}.csv")
    if limit:
        df = df.head(limit)
    seqs = df["seq"].tolist()
    n = len(seqs)
    dim = model.config.hidden_size
    emb = np.zeros((n, dim), dtype=np.float32)
    score_wt = np.zeros(n, dtype=np.float32)

    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in AA}
    wt_ids = [aa_ids[c] for c in wt]
    t0 = time.time()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        chunk = seqs[start:stop]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]

        mask = enc["attention_mask"].bool()
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        emb[start:stop] = pooled.float().cpu().numpy()

        logp = torch.log_softmax(out.logits.float(), dim=-1)
        for k, seq in enumerate(chunk):
            pos = [i for i, (a, b) in enumerate(zip(wt, seq)) if a != b]
            if not pos:
                continue
            # +1 for the <cls> token that ESM prepends
            idx = torch.tensor([p + 1 for p in pos], device=logp.device)
            mt = torch.tensor([aa_ids[seq[p]] for p in pos], device=logp.device)
            wtv = torch.tensor([wt_ids[p] for p in pos], device=logp.device)
            row = logp[k].index_select(0, idx)
            score_wt[start + k] = (row.gather(1, mt[:, None]) - row.gather(1, wtv[:, None])).sum().item()

        if start % (batch * 50) == 0:
            done = stop
            el = time.time() - t0
            print(f"  {name} {done}/{n}  {done / max(el, 1e-9):.1f} seq/s  elapsed {el / 60:.1f} min", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_emb.npy", emb)
    np.save(OUT / f"{name}_score_wtmarginal.npy", score_wt)
    elapsed = time.time() - t0
    manifest[name] = {
        "rows": int(n),
        "seconds": round(elapsed, 1),
        "seq_per_s": round(n / elapsed, 2),
        "peak_gpu_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 0) if device == "cuda" else None,
        "batch": batch,
    }
    print(f"{name}: {n} variants in {elapsed / 60:.1f} min ({n / elapsed:.1f} seq/s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "esm2_t33_650M_UR50D"))
    ap.add_argument("--splits", default="test,valid,train")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows per split")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))
    wt = manifest["wt_sequence"]
    tok, model = load_model(args.model, device)
    print(f"device={device} model={args.model} params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    timing = {}
    for name in args.splits.split(","):
        run_split(name.strip(), tok, model, device, args.batch, wt, timing, args.limit)

    path = OUT / "timing_embed.json"
    path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
