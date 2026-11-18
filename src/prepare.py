"""Step 1: prepare TAPE fluorescence (FLU-S) data.

- loads the three official splits
- derives the wild-type (avGFP) sequence by per-position consensus
- validates the derived WT against the `num_mutations` field shipped with TAPE
- writes normalised parquet-free CSV + a sha256 manifest

No numbers are invented here: everything printed comes from the files.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fluorescence"
OUT = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(name: str) -> pd.DataFrame:
    with open(DATA / f"fluorescence_{name}.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    df = pd.DataFrame(
        {
            "seq": [r["primary"] for r in rows],
            "log_fluor": [float(r["log_fluorescence"][0]) for r in rows],
            "num_mutations": [int(r["num_mutations"]) for r in rows],
            "protein_length": [int(r["protein_length"]) for r in rows],
            "split": name,
        }
    )
    return df


def consensus_wt(seqs) -> str:
    """Most frequent residue per position across all variants."""
    length = len(seqs[0])
    wt = []
    for i in range(length):
        counts = Counter(s[i] for s in seqs)
        wt.append(counts.most_common(1)[0][0])
    return "".join(wt)


def mutations(seq: str, wt: str) -> list[tuple[int, str, str]]:
    return [(i, wt[i], seq[i]) for i in range(len(wt)) if seq[i] != wt[i]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {n: load_split(n) for n in ("train", "valid", "test")}

    all_seqs = pd.concat(splits.values())["seq"].tolist()
    length_counts = Counter(len(s) for s in all_seqs)
    modal_len = length_counts.most_common(1)[0][0]
    dropped = {L: n for L, n in sorted(length_counts.items()) if L != modal_len}

    # keep the modal length only: shorter variants are internal deletions and cannot be
    # scored position-wise against a fixed wild type. Counts are recorded, not hidden.
    for name, df in splits.items():
        before = len(df)
        splits[name] = df[df["seq"].str.len() == modal_len].reset_index(drop=True)
        print(f"{name:5s} kept {len(splits[name])}/{before} (length == {modal_len})")

    all_seqs = pd.concat(splits.values())["seq"].tolist()
    wt = consensus_wt(all_seqs)

    print(f"\nvariants used   = {len(all_seqs)}")
    print(f"sequence length = {modal_len}")
    print(f"dropped lengths = {dropped}")
    print(f"consensus WT    = {wt}")

    # --- validation: does the derived WT reproduce TAPE's own num_mutations? ---
    report = {}
    for name, df in splits.items():
        diffs = [len(mutations(s, wt)) for s in df["seq"]]
        match = sum(int(a == b) for a, b in zip(diffs, df["num_mutations"]))
        report[name] = {
            "rows": int(len(df)),
            "num_mutations_match": int(match),
            "num_mutations_match_rate": round(match / len(df), 4),
            "mut_mean": float(df["num_mutations"].mean()),
            "mut_max": int(df["num_mutations"].max()),
            "mut_hist": {int(k): int(v) for k, v in Counter(df["num_mutations"]).most_common()},
            "log_fluor_mean": round(float(df["log_fluor"].mean()), 4),
            "log_fluor_std": round(float(df["log_fluor"].std(ddof=1)), 4),
            "log_fluor_min": round(float(df["log_fluor"].min()), 4),
            "log_fluor_max": round(float(df["log_fluor"].max()), 4),
        }
        df["n_mut_derived"] = diffs
        print(
            f"{name:5s} rows={len(df):6d} mut_mean={df['num_mutations'].mean():5.2f} "
            f"max={df['num_mutations'].max():3d} wt_reproduces_num_mutations={match}/{len(df)}"
        )

    for name, df in splits.items():
        df.to_csv(OUT / f"flu_{name}.csv", index=False)

    manifest = {
        "source": "TAPE fluorescence (S3: songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz)",
        "wt_sequence": wt,
        "wt_derivation": f"per-position consensus over {len(all_seqs)} kept variants (length {modal_len})",
        "dropped_length_counts": {str(k): v for k, v in dropped.items()},
        "files": {
            p.name: {"sha256": sha256(p), "bytes": os.path.getsize(p)}
            for p in sorted(DATA.glob("*.json"))
        },
        "splits": report,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT}/manifest.json")


if __name__ == "__main__":
    main()
