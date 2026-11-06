"""RDKit 2D descriptors (+ a few cheap custom ones), cleaned for use by any model family."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors

_DESC_NAMES = [n for n, _ in Descriptors._descList]


def _one(smi: Optional[str]) -> list[float]:
    if smi is None:
        return [np.nan] * (len(_DESC_NAMES) + 4)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return [np.nan] * (len(_DESC_NAMES) + 4)
    vals = []
    for _, fn in Descriptors._descList:
        try:
            vals.append(float(fn(m)))
        except Exception:
            vals.append(np.nan)
    # a few extras that are not in the standard list
    n_heavy = m.GetNumHeavyAtoms() or 1
    n_stereo = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    charge_pos = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0)
    charge_neg = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() < 0)
    vals += [n_stereo, n_stereo / n_heavy, charge_pos, charge_neg]
    return vals


def rdkit_descriptors(smiles_std: Sequence[Optional[str]], n_jobs: int = 8, log_compress: bool = True):
    """Return (matrix, names). NaN/inf are median-imputed, constant columns dropped and heavy tails log-compressed."""
    rows = Parallel(n_jobs=n_jobs, batch_size=32)(delayed(_one)(s) for s in smiles_std)
    X = np.asarray(rows, dtype=np.float64)
    names = _DESC_NAMES + ["n_stereo", "frac_stereo", "n_pos_atoms", "n_neg_atoms"]
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    r, c = np.where(np.isnan(X))
    X[r, c] = med[c]
    keep = X.std(axis=0) > 1e-12
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    if log_compress:
        big = np.abs(X).max(axis=0) > 1e3
        X[:, big] = np.sign(X[:, big]) * np.log1p(np.abs(X[:, big]))
    return X.astype(np.float32), names
