"""Embeddings of every molecule from several pretrained chemical models (GPU recommended, CPU/MPS work for the small ones).

    python scripts/colab_embeddings.py --smiles colab/smiles_std.csv --out-dir data/embeddings
    python scripts/colab_embeddings.py --models chemberta77m_mlm chemeleon          # a subset
    python scripts/colab_embeddings.py --models all --trust-remote-code             # + MoLFormer (runs code shipped in its HF repo) and the large models

Models (name -> what it is)
    chemberta77m_mlm  DeepChem/ChemBERTa-77M-MLM        RoBERTa on 77M PubChem SMILES (masked-LM)
    chemberta77m_mtr  DeepChem/ChemBERTa-77M-MTR        same encoder trained to regress 200 RDKit descriptors (often better for properties)
    chemberta_zinc    seyonec/ChemBERTa-zinc-base-v1    RoBERTa on ZINC
    molformer         ibm-research/MoLFormer-XL-both-10pct   linear-attention transformer, 1.1B molecules (needs trust_remote_code)
    chemeleon         CheMeleon (chemprop D-MPNN)       graph foundation model pretrained on 1M molecules to predict Mordred descriptors
    gpt2_zinc87m, chemgpt1p2b                          optional causal LMs (`--models all`); ChemGPT reads SELFIES (needs `pip install selfies`)

Output: one <out-dir>/<name>.npz per model with `smiles` (unique standardised SMILES) and `emb` (n x d float32), plus
manifest.json. The project matches embeddings to rows by standardised SMILES (block type `npz_embedding`).

CheMeleon loading follows the MIT-licensed reference implementation of J. Burns et al.
(https://github.com/JacksonBurns/chemeleon, class CheMeleonFingerprint).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "chemberta77m_mlm": dict(kind="hf", id="DeepChem/ChemBERTa-77M-MLM"),
    "chemberta77m_mtr": dict(kind="hf", id="DeepChem/ChemBERTa-77M-MTR"),
    "chemberta_zinc": dict(kind="hf", id="seyonec/ChemBERTa-zinc-base-v1"),
    "molformer": dict(kind="hf", id="ibm-research/MoLFormer-XL-both-10pct", trust=True, kwargs={"deterministic_eval": True}),
    "chemeleon": dict(kind="chemeleon"),
    "gpt2_zinc87m": dict(kind="hf", id="entropy/gpt2_zinc_87m"),
    "chemgpt1p2b": dict(kind="hf", id="ncfrey/ChemGPT-1.2B", dtype="bf16", input="selfies"),   # trained on SELFIES, not SMILES
}
DEFAULT = ["chemberta77m_mlm", "chemberta77m_mtr", "molformer", "chemeleon"]   # molformer is skipped unless --trust-remote-code
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_selfies(smiles: list[str]) -> list[str]:
    """ChemGPT was pretrained on SELFIES strings; feeding it SMILES would give meaningless embeddings."""
    import selfies as sf

    out, bad = [], 0
    for s in smiles:
        try:
            enc = sf.encoder(s)
        except Exception:  # noqa: BLE001 - selfies raises its own EncoderError for SMILES it cannot represent
            enc = None
        if not enc:
            enc, bad = "", bad + 1
        out.append(enc)
    if bad:
        print(f"    {bad} molecules could not be converted to SELFIES (they get the embedding of an empty string)")
    return out


def embed_hf(smiles: list[str], spec: dict, device, batch: int, max_len: int, dtype_name: str) -> np.ndarray:
    """Mean-pooled last hidden state over the real (non-padding) tokens."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = to_selfies(smiles) if spec.get("input") == "selfies" else smiles
    trust = bool(spec.get("trust", False))
    tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=trust)
    if tok.pad_token is None:                                    # causal LMs (GPT-style) have no pad token
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModel.from_pretrained(spec["id"], trust_remote_code=trust, **spec.get("kwargs", {})).eval()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    model = model.to(device=device, dtype=dtype)
    order = np.argsort([len(s) for s in texts])                  # sort by length: far less padding, much faster
    out: list[np.ndarray | None] = [None] * len(texts)
    i, bs = 0, batch
    while i < len(order):
        idx = order[i:i + bs]
        try:
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.inference_mode():
                hid = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
            pooled = ((hid * m).sum(1) / m.sum(1)).float().cpu().numpy()
        except RuntimeError as e:                                 # out of memory -> halve the batch and retry
            if "out of memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"    out of memory, batch -> {bs}")
                continue
            raise
        for k, j in enumerate(idx):
            out[j] = pooled[k]
        i += len(idx)
    return np.vstack(out).astype(np.float32)


def embed_chemeleon(smiles: list[str], device, batch: int) -> np.ndarray:
    import torch
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN
    from rdkit import Chem
    from urllib.request import urlretrieve

    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / "chemeleon_mp.pt"
    if not path.exists():
        urlretrieve(CHEMELEON_URL, path)
    ck = torch.load(path, weights_only=True)
    mp = nn.BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    model = MPNN(message_passing=mp, agg=nn.MeanAggregation(), predictor=RegressionFFN(input_dim=mp.output_dim)).eval().to(device)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    chunks, dim = [], mp.output_dim
    for i in range(0, len(smiles), batch):
        mols = [Chem.MolFromSmiles(s) for s in smiles[i:i + batch]]
        good = [m for m in mols if m is not None]
        emb = np.zeros((len(mols), dim), dtype=np.float32)
        if good:
            bmg = BatchMolGraph([feat(m) for m in good])
            bmg.to(device=model.device)
            with torch.no_grad():
                e = model.fingerprint(bmg).numpy(force=True)
            emb[[k for k, m in enumerate(mols) if m is not None]] = e
        chunks.append(emb)
    return np.vstack(chunks)


def read_smiles(path: str) -> list[str]:
    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        col = "smiles_std" if "smiles_std" in df.columns else df.columns[0]
        return sorted(set(df[col].dropna().astype(str)))
    return sorted({ln.strip() for ln in p.read_text().splitlines() if ln.strip()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", default="colab/smiles_std.csv", help="csv with a smiles_std column (scripts/export_smiles.py) or a text file")
    ap.add_argument("--out-dir", default="data/embeddings")
    ap.add_argument("--models", nargs="*", default=DEFAULT, help=f"names from {sorted(MODELS)}, or `all`")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N molecules")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow models whose HuggingFace repository ships its own Python code (MoLFormer). That code is downloaded "
                         "and EXECUTED, so it is off by default: use it on Colab (an isolated VM), not on your own machine")
    args = ap.parse_args()

    names = sorted(MODELS) if args.models == ["all"] else args.models
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {sorted(MODELS)}")
    smiles = read_smiles(args.smiles)
    if args.limit:
        smiles = smiles[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    import torch

    print(f"{len(smiles)} molecules | device {device} | torch {torch.__version__}")
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        spec = MODELS[name]
        t0 = time.time()
        if spec.get("trust") and not args.trust_remote_code:
            print(f"[{name}] skipped: the model repository ships its own Python code; pass --trust-remote-code to run it (Colab)")
            manifest[name] = {"status": "skipped (needs --trust-remote-code)"}
            continue
        print(f"[{name}] ...", flush=True)
        try:
            if spec["kind"] == "chemeleon":
                emb = embed_chemeleon(smiles, device, args.batch)
            else:
                dtype = spec.get("dtype", "fp32") if args.dtype == "auto" else args.dtype
                emb = embed_hf(smiles, spec, device, args.batch, args.max_len, dtype)
        except Exception as e:  # noqa: BLE001 - keep going with the other models
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            manifest[name] = {"status": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            manifest_path.write_text(json.dumps(manifest, indent=1))
            continue
        ok = bool(np.isfinite(emb).all())
        centred = emb - emb.mean(0)
        z = centred / np.clip(np.linalg.norm(centred, axis=1, keepdims=True), 1e-9, None)
        sample = np.random.default_rng(0).choice(len(z), size=min(300, len(z)), replace=False)
        cos = z[sample] @ z[sample].T
        spread = float((cos.sum() - len(sample)) / (len(sample) * (len(sample) - 1)))
        print(f"[{name}] dim={emb.shape[1]} finite={ok} mean per-dim std={emb.std(0).mean():.4f} "
              f"mean pairwise cosine (centred)={spread:.3f}  {time.time() - t0:.0f}s")
        if not ok:
            emb = np.nan_to_num(emb)
        np.savez_compressed(out_dir / f"{name}.npz", smiles=np.array(smiles, dtype=object), emb=emb)
        manifest[name] = {"status": "ok", "dim": int(emb.shape[1]), "n": len(smiles), "seconds": round(time.time() - t0, 1),
                          "model": spec.get("id", "CheMeleon"), "device": str(device)}
        manifest_path.write_text(json.dumps(manifest, indent=1))
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
