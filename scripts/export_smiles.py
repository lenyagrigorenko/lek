"""Export the unique standardised SMILES of every molecule of the project (train + test + external) for GPU embedding.

    python scripts/export_smiles.py                  # -> colab/smiles_std.csv   (about 3 000 lines, one column: smiles_std)

Upload this ONE small file to Colab (see notebooks/colab_embeddings.ipynb); the competition files themselves never leave
your machine. Run it after `python -m vacv_si run-all` (or at least `python -m vacv_si prepare -c configs/exp/final.yaml`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vacv_si.config import load_config
from vacv_si.data.prepare import load_all_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/exp/final.yaml")
    ap.add_argument("--out", default="colab/smiles_std.csv")
    args = ap.parse_args()
    df = load_all_table(load_config(args.config))
    smiles = sorted(set(df["smiles_std"].dropna().astype(str)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles_std": smiles}).to_csv(args.out, index=False)
    print(f"{len(smiles)} unique molecules -> {args.out}")


if __name__ == "__main__":
    main()
