"""SMILES standardisation: one canonical, neutral, single-fragment representation per molecule.

Used for (a) model inputs, (b) de-duplication and (c) matching external records against the
competition molecules (via InChIKey).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StdOptions:
    keep_largest_fragment: bool = True
    neutralize: bool = True
    canonical_tautomer: bool = False   # slow and can erase stereo; off by default

    @classmethod
    def from_cfg(cls, cfg: Optional[dict]) -> "StdOptions":
        return cls(**(cfg or {}))


def parse_smiles(smi: str) -> Optional[Chem.Mol]:
    """Parse a SMILES; fall back to a lenient sanitisation for odd inputs (hypervalent dyes etc.)."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    smi = smi.strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY,
            catchErrors=True,
        )
        return mol
    except Exception:
        return None


def standardize_mol(mol: Chem.Mol, opts: StdOptions = StdOptions()) -> Optional[Chem.Mol]:
    try:
        mol = rdMolStandardize.Cleanup(mol)
        if opts.keep_largest_fragment:
            mol = rdMolStandardize.FragmentParent(mol)
        if opts.neutralize:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if opts.canonical_tautomer:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol
    except Exception:
        return None


def _one(smi: str, opts: StdOptions) -> dict:
    out = {"smiles_std": None, "smiles_flat": None, "inchikey": None, "ik14": None, "n_frag_raw": 0, "std_ok": False}
    mol = parse_smiles(smi)
    if mol is None:
        return out
    out["n_frag_raw"] = len(Chem.GetMolFrags(mol))
    std = standardize_mol(mol, opts)
    if std is None:
        return out
    try:
        out["smiles_std"] = Chem.MolToSmiles(std, isomericSmiles=True)
        out["smiles_flat"] = Chem.MolToSmiles(std, isomericSmiles=False)
        key = Chem.MolToInchiKey(std)
        out["inchikey"] = key or None
        out["ik14"] = key[:14] if key else None
        out["std_ok"] = True
    except Exception:
        pass
    return out


def standardize_series(smiles: pd.Series, opts: StdOptions = StdOptions(), n_jobs: int = 8) -> pd.DataFrame:
    """Standardise a Series of SMILES (unique values are processed once, in parallel)."""
    uniq = pd.Series(smiles.dropna().astype(str).unique())
    if len(uniq) == 0:
        return pd.DataFrame(index=smiles.index)
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_one)(s, opts) for s in uniq)
    table = pd.DataFrame(res, index=uniq.values)
    return table.reindex(smiles.astype(str).values).set_axis(smiles.index)


def mols_from_std(smiles_std: pd.Series) -> list[Optional[Chem.Mol]]:
    return [parse_smiles(s) if isinstance(s, str) else None for s in smiles_std]
