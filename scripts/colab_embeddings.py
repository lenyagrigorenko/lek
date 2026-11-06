"""Embeddings of every molecule from a pretrained chemical language model (run on a GPU, e.g. Google Colab).

Colab recipe
    !git clone https://github.com/lenyagrigorenko/lek && cd lek
    !pip install -q -r requirements.txt transformers torch && pip install -q -e . --no-deps
    # upload train.csv / test.csv / sample_submit.csv into data/raw/, then:
    !python -m vacv_si prepare -c configs/exp/final.yaml          # writes data/processed/all_ext.csv
    !python scripts/colab_embeddings.py --model DeepChem/ChemBERTa-77M-MLM --out data/embeddings/chemberta77m.npz
Download the .npz, put it into data/embeddings/ of the local checkout and run configs/exp/embeddings.yaml.

Other models worth trying: seyonec/ChemBERTa-zinc-base-v1, ibm/MoLFormer-XL-both-10pct (needs --trust-remote-code).
NOTE: this script needs torch + transformers and was not executed in the development environment (no GPU stack there).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/processed/all_ext.csv", help="table written by `vacv_si prepare` (column smiles_std)")
    ap.add_argument("--model", default="DeepChem/ChemBERTa-77M-MLM")
    ap.add_argument("--out", default="data/embeddings/chemberta77m.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    smiles = sorted(set(pd.read_csv(args.table)["smiles_std"].dropna().astype(str)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to(dev).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(smiles), args.batch):
            enc = tok(smiles[i:i + args.batch], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(dev)
            hid = model(**enc).last_hidden_state                          # (batch, tokens, dim)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            chunks.append(((hid * mask).sum(1) / mask.sum(1)).cpu().numpy())   # mean pooling over real tokens
    emb = np.vstack(chunks).astype(np.float32)
    np.savez_compressed(args.out, smiles=np.array(smiles, dtype=object), emb=emb)
    print(f"{len(smiles)} molecules -> {args.out}  embedding dim {emb.shape[1]}  device {dev}")


if __name__ == "__main__":
    main()
