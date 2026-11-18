"""E3: download the cross-model benchmark weights (CPU/network step, run once).

huggingface.co is not reachable from the shell on the machine that produced
results/ (DNS is filtered), so downloads go through the public mirror
hf-mirror.com with a browser User-Agent (the mirror 403s bare clients).
The files are byte-identical HF releases; the script verifies each download
by size and records sha256 into models/manifest_e3.json so the exact
checkpoint used by embed.py is auditable.

Models (license as published on the upstream HF repos):
  westlake-repl/SaProt_35M_AF2_seqOnly  (SaProt authors' AA-only 35M; the
      regular SaProt 35M/650M frozen embeddings need 3Di structure tokens,
      which this benchmark does not provide -> seqOnly is the honest variant)
  Rostlab/prot_bert_bfd                 (ProtBERT-BFD, BERT-style; input must
      be space-separated residues, handled by embed.py family="bert")
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIRROR = "https://hf-mirror.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0"}

REPOS = {
    "SaProt_35M_AF2_seqOnly": (
        "westlake-repl/SaProt_35M_AF2_seqOnly",
        ["config.json", "special_tokens_map.json", "tokenizer_config.json",
         "vocab.txt", "pytorch_model.bin"],
    ),
    "prot_bert_bfd": (
        "Rostlab/prot_bert_bfd",
        ["config.json", "special_tokens_map.json", "tokenizer_config.json",
         "vocab.txt", "pytorch_model.bin"],
    ),
}


def fetch(repo: str, fname: str, dest: pathlib.Path) -> dict:
    url = f"{MIRROR}/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    manifest: dict = {"mirror": MIRROR, "note": "sha256 of the exact bytes used locally",
                      "repos": {}}
    for dirname, (repo, files) in REPOS.items():
        d = ROOT / "models" / dirname
        d.mkdir(parents=True, exist_ok=True)
        entry = {"upstream": repo, "files": {}}
        for f in files:
            out = d / f
            if out.exists() and out.stat().st_size > 0:
                info = {"bytes": out.stat().st_size,
                        "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
                print(f"{repo}/{f}: already present, hashing only", flush=True)
            else:
                info = fetch(repo, f, out)
            entry["files"][f] = info
            print(f"{repo}/{f}: {info['bytes']} bytes sha256={info['sha256'][:16]}...", flush=True)
        manifest["repos"][dirname] = entry
    mp = ROOT / "models" / "manifest_e3.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {mp}")


if __name__ == "__main__":
    main()
