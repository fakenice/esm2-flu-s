"""Step 2: one forward pass per variant -> mean-pooled embedding + WT-marginal zero-shot score.

WT-marginal score (one pass per variant, no masking):
    s = sum_i [ log p(mt_i | variant) - log p(wt_i | variant) ]

Outputs (per split, per model): embeddings .npy, scores .npy under
results/embed/<model>/, and timing_embed.json in the same directory.
Everything is resumable at the shard level: each shard is written once and skipped on rerun.

E3 cross-model note: the four registry entries below differ in input
formatting and vocabulary handling, not in the scoring formula. Each entry
carries a per-residue token template and a joiner:
  esm2_*            template "{a}",   joined   - raw sequence, ESM2 vocab,
                    +1 offset for <cls>.
  saprot_35m_seqonly template "{a}#", joined   - the authors' sequence-only
                    checkpoint (model card: training marks every 3Di slot
                    with '#'); vocabulary tokens are the pairs "A#".."W#".
  protbert_bfd      template "{a}",   joined " " - Rostlab convention
                    space-separated residues, +1 offset for [CLS].
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
AA = "ACDEFGHIKLMNPQRSTVWY"

MODELS = {
    "esm2_650M": {"path": ROOT / "models" / "esm2_t33_650M_UR50D",
                  "family": "esm", "upstream": "facebook/esm2_t33_650M_UR50D",
                  "tmpl": "{a}", "join": ""},
    "esm2_35M": {"path": ROOT / "models" / "esm2_t12_35M_UR50D",
                 "family": "esm", "upstream": "facebook/esm2_t12_35M_UR50D",
                 "tmpl": "{a}", "join": ""},
    "saprot_35m_seqonly": {"path": ROOT / "models" / "SaProt_35M_AF2_seqOnly",
                           "family": "esm", "upstream": "westlake-repl/SaProt_35M_AF2_seqOnly",
                           "tmpl": "{a}#", "join": ""},
    "protbert_bfd": {"path": ROOT / "models" / "prot_bert_bfd",
                     "family": "bert", "upstream": "Rostlab/prot_bert_bfd",
                     "tmpl": "{a}", "join": " "},
}


class ManualBertTok:
    """Deterministic encoder for ProtBERT: input ids are the line numbers of
    the authors' published vocab.txt (the HF fast-tokenizer machinery is not
    reproducible for this repo: no tokenizer.json, and the slow BertTokenizer
    re-registers its special tokens with added-vocab indices that clash with
    the checkpoint's [CLS]=2/[SEP]=3). One residue = one token, uppercase."""

    def __init__(self, vocab_path: Path):
        self.ids = {line.strip(): k for k, line in enumerate(
            vocab_path.read_text(encoding="utf-8").splitlines())}
        for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            if t not in self.ids:
                raise RuntimeError(f"vocab.txt lacks {t}")
        self.pad_id, self.cls_id, self.sep_id = self.ids["[PAD]"], self.ids["[CLS]"], self.ids["[SEP]"]
        self.unk_token_id = self.ids["[UNK]"]

    def convert_tokens_to_ids(self, t: str):
        return self.ids.get(t, self.unk_token_id)

    def encode_batch(self, seqs: list[str], device: str):
        rows = [[self.cls_id] + [self.ids.get(c, self.unk_token_id) for c in s] + [self.sep_id]
                for s in seqs]
            # unk must not occur for amino-acid sequences; guard explicitly
        for s, r in zip(seqs, rows):
            if self.unk_token_id in r:
                bad = [c for c in s if c not in self.ids]
                raise RuntimeError(f"ProtBERT vocab lacks residue tokens {bad}")
        width = max(len(r) for r in rows)
        input_ids, attn = [], []
        for r in rows:
            input_ids.append(r + [self.pad_id] * (width - len(r)))
            attn.append([1] * len(r) + [0] * (width - len(r)))
        import torch as _t
        return {"input_ids": _t.tensor(input_ids, device=device),
                "attention_mask": _t.tensor(attn, device=device)}


def load_model(model_dir: Path, family: str, device: str):
    if family == "bert":
        # the published config.json lacks a model_type key (transformers 5
        # refuses to route it through AutoModelForMaskedLM); load explicitly
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(str(model_dir)).to(device).eval()
        return ManualBertTok(model_dir / "vocab.txt"), model
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir)).to(device).eval()
    return tok, model


def encode(chunk: list[str], spec: dict, tok, device: str):
    # both families tokenize as [CLS] + one token per residue + terminal token
    expected = torch.tensor([len(s) + 2 for s in chunk])
    if isinstance(tok, ManualBertTok):
        enc = tok.encode_batch(chunk, device)
    else:
        tmpl, join = spec["tmpl"], spec["join"]
        data = [join.join(tmpl.format(a=c) for c in s) for s in chunk]
        enc = tok(data, return_tensors="pt", padding=True, truncation=False).to(device)
    ntok = enc["attention_mask"].sum(1).cpu()
    if not bool((ntok == expected).all()):
        bad = (ntok != expected).nonzero().flatten()[:3].tolist()
        raise RuntimeError(f"{spec['family']}: tokenizer split/truncated sequences (rows {bad}: "
                           f"got {ntok[bad].tolist()} vs expected {(expected[bad]).tolist()})")
    return enc


@torch.no_grad()
def run_split(name: str, tok, model, spec: dict, device: str, batch: int, wt: str,
              manifest: dict, outdir: Path, limit: int = 0):
    df = pd.read_csv(PROC / f"flu_{name}.csv")
    if limit:
        df = df.head(limit)
    seqs = df["seq"].tolist()
    n = len(seqs)
    dim = model.config.hidden_size
    emb = np.zeros((n, dim), dtype=np.float32)
    score_wt = np.zeros(n, dtype=np.float32)

    unk = tok.unk_token_id
    aa_ids = {}
    for a in AA:
        key = spec["tmpl"].format(a=a)
        tid = tok.convert_tokens_to_ids(key)
        if tid is None or tid == unk:
            raise RuntimeError(f"{spec['family']}: token '{key}' "
                               "missing from the vocabulary")
        aa_ids[a] = tid
    wt_ids = [aa_ids[c] for c in wt]
    t0 = time.time()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        chunk = seqs[start:stop]
        enc = encode(chunk, spec, tok, device)
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
            # +1 for the <cls>/[CLS] token prepended by both families
            idx = torch.tensor([p + 1 for p in pos], device=logp.device)
            mt = torch.tensor([aa_ids[seq[p]] for p in pos], device=logp.device)
            wtv = torch.tensor([wt_ids[p] for p in pos], device=logp.device)
            row = logp[k].index_select(0, idx)
            score_wt[start + k] = (
                row.gather(1, mt[:, None]) - row.gather(1, wtv[:, None])
            ).sum().item()

        if start % (batch * 50) == 0:
            done = stop
            el = time.time() - t0
            print(f"  {name} {done}/{n}  {done / max(el, 1e-9):.1f} seq/s  elapsed {el / 60:.1f} min", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{name}_emb.npy", emb)
    np.save(outdir / f"{name}_score_wtmarginal.npy", score_wt)
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
    ap.add_argument("--model", default="esm2_650M", choices=sorted(MODELS),
                    help="registry tag; outputs go to results/embed/<tag>/")
    ap.add_argument("--splits", default="test,valid,train")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: 16 for esm2_650M, 32 for the smaller models")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows per split")
    args = ap.parse_args()

    spec = MODELS[args.model]
    batch = args.batch or (16 if args.model == "esm2_650M" else 32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))
    wt = manifest["wt_sequence"]
    tok, model = load_model(spec["path"], spec["family"], device)
    print(f"device={device} model={args.model} upstream={spec['upstream']} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M batch={batch}")

    aa_key = lambda a: spec["tmpl"].format(a=a)
    timing = {"model": args.model, "upstream": spec["upstream"], "family": spec["family"],
              "params_M": round(sum(pp.numel() for pp in model.parameters()) / 1e6, 1),
              "input_template": spec["tmpl"], "join": repr(spec["join"]),
              "vocab_aa_ids": {a: tok.convert_tokens_to_ids(aa_key(a)) for a in AA},
              "splits": {}}
    outdir = EMB / args.model
    for name in args.splits.split(","):
        run_split(name.strip(), tok, model, spec, device, batch, wt,
                  timing["splits"], outdir, args.limit)

    path = outdir / "timing_embed.json"
    path.write_text(json.dumps(timing, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
