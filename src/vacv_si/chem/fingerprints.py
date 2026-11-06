"""Molecular fingerprints as dense float32 matrices (count-based wherever RDKit supports it)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as rfg, rdMolDescriptors, rdReducedGraphs


def _fill(gen, mols: Sequence[Optional[Chem.Mol]], n_bits: int) -> np.ndarray:
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = gen.GetCountFingerprintAsNumPy(m)
    return X


def morgan_counts(mols, radius: int = 2, n_bits: int = 2048, chirality: bool = False, feature: bool = False,
                  min_radius: int = 0):
    kw = dict(radius=radius, fpSize=n_bits, includeChirality=chirality)
    if feature:
        kw["atomInvariantsGenerator"] = rfg.GetMorganFeatureAtomInvGen()
    gen = rfg.GetMorganGenerator(**kw)
    X = _fill(gen, mols, n_bits)
    return X, [f"morgan{radius}{'f' if feature else ''}_{i}" for i in range(n_bits)]


def atom_pair_counts(mols, n_bits: int = 2048, chirality: bool = False, max_distance: int = 30):
    gen = rfg.GetAtomPairGenerator(fpSize=n_bits, includeChirality=chirality, maxDistance=max_distance)
    return _fill(gen, mols, n_bits), [f"ap_{i}" for i in range(n_bits)]


def torsion_counts(mols, n_bits: int = 2048, chirality: bool = False):
    gen = rfg.GetTopologicalTorsionGenerator(fpSize=n_bits, includeChirality=chirality)
    return _fill(gen, mols, n_bits), [f"tt_{i}" for i in range(n_bits)]


def rdkit_path_counts(mols, n_bits: int = 2048, max_path: int = 6):
    gen = rfg.GetRDKitFPGenerator(maxPath=max_path, fpSize=n_bits)
    return _fill(gen, mols, n_bits), [f"rdk{max_path}_{i}" for i in range(n_bits)]


def maccs(mols):
    X = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr)
        X[i] = arr
    return X, [f"maccs_{i}" for i in range(167)]


def avalon_counts(mols, n_bits: int = 1024):
    from rdkit.Avalon import pyAvalonTools

    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        for k, v in pyAvalonTools.GetAvalonCountFP(m, nBits=n_bits).GetNonzeroElements().items():
            X[i, k % n_bits] += v
    return X, [f"avalon_{i}" for i in range(n_bits)]


def erg(mols):
    n = 315
    X = np.zeros((len(mols), n), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = rdReducedGraphs.GetErGFingerprint(m)
    return X, [f"erg_{i}" for i in range(n)]


def mqn(mols):
    X = np.zeros((len(mols), 42), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is not None:
            X[i] = rdMolDescriptors.MQNs_(m)
    return X, [f"mqn_{i}" for i in range(42)]
