"""FeatureStore: named feature blocks (+ kernels) computed once for *all* rows and cached on disk.

A block is declared in the config and referred to by name from model specs:

    features:
      blocks:
        morgan2c: {type: morgan_counts, radius: 2, n_bits: 2048}
        desc:     {type: rdkit_desc}
        meta_num: {type: meta_num}

To add a new kind of block, write a function `(store, params) -> (matrix, names)` and decorate it with
`@register_block("my_type")`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..chem import fingerprints as fp
from ..chem.descriptors import rdkit_descriptors
from ..chem.kernels import minmax_kernel
from ..chem.standardize import mols_from_std
from ..data.schema import assert_no_forbidden
from ..paths import FEATURE_CACHE
from .meta import build_meta

log = logging.getLogger(__name__)

# Bump when the *code* that builds a block changes (the on-disk cache and every stored model hash include it).
FEATURE_CODE_VERSION = "3-smiles-style"

BLOCK_BUILDERS: dict[str, Callable] = {}


def register_block(name: str):
    def deco(fn):
        BLOCK_BUILDERS[name] = fn
        return fn

    return deco


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:14]


class FeatureStore:
    def __init__(self, df: pd.DataFrame, features_cfg: dict, cache_dir: Path = FEATURE_CACHE, n_jobs: int = 8):
        self.df = df.reset_index(drop=True)
        self.cfg = features_cfg or {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self._mols = None
        self._meta = None
        self._blocks: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._kernels: dict[str, np.ndarray] = {}
        self._smiles_key = _digest(self.df["smiles_std"].fillna("").tolist())
        self._text_key = _digest(self.df[["title", "strain", "culture", "kind", "smiles_raw"]].fillna("").astype(str).values.tolist())

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.df)

    def mols(self):
        if self._mols is None:
            self._mols = mols_from_std(self.df["smiles_std"])
        return self._mols

    def smiles_std(self) -> list:
        return [s if isinstance(s, str) else None for s in self.df["smiles_std"]]

    def meta(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._meta is None:
            self._meta = build_meta(self.df)
        return self._meta

    # ------------------------------------------------------------------ blocks
    def _block_spec(self, name: str) -> dict:
        spec = (self.cfg.get("blocks") or {}).get(name)
        if spec is None:
            raise KeyError(f"feature block {name!r} is not declared in features.blocks")
        return spec

    def block(self, name: str) -> np.ndarray:
        return self._get(name)[0]

    def block_names(self, name: str) -> list[str]:
        return self._get(name)[1]

    def _get(self, name: str):
        if name in self._blocks:
            return self._blocks[name]
        spec = dict(self._block_spec(name))
        typ = spec.pop("type")
        if typ not in BLOCK_BUILDERS:
            raise KeyError(f"unknown block type {typ!r}; known: {sorted(BLOCK_BUILDERS)}")
        key_src = self._text_key if typ.startswith("meta") else self._smiles_key
        key = _digest(key_src, typ, spec, FEATURE_CODE_VERSION)
        npy, js = self.cache_dir / f"{name}-{key}.npy", self.cache_dir / f"{name}-{key}.json"
        if npy.exists() and js.exists():
            X, names = np.load(npy), json.loads(js.read_text())
        else:
            log.info("building feature block %s (%s)", name, typ)
            binarize, log1p = spec.pop("binarize", False), spec.pop("log1p", False)
            X, names = BLOCK_BUILDERS[typ](self, spec)
            X = np.asarray(X, dtype=np.float32)
            if binarize:
                X = (X > 0).astype(np.float32)
            elif log1p:
                X = np.log1p(np.clip(X, 0, None))
            np.save(npy, X)
            js.write_text(json.dumps(names))
        assert_no_forbidden(names)
        assert X.shape[0] == len(self.df), (name, X.shape)
        self._blocks[name] = (X, names)
        return self._blocks[name]

    def expand(self, names: list[str]) -> list[str]:
        """Expand `@group` references (features.groups) into block names."""
        groups = self.cfg.get("groups") or {}
        out: list[str] = []
        for n in names or []:
            if isinstance(n, str) and n.startswith("@"):
                if n[1:] not in groups:
                    raise KeyError(f"feature group {n!r} is not declared in features.groups")
                out.extend(self.expand(groups[n[1:]]))
            else:
                out.append(n)
        return out

    def expand_cat(self, cols: list[str]) -> list[str]:
        groups = self.cfg.get("cat_groups") or {}
        out: list[str] = []
        for c in cols or []:
            if isinstance(c, str) and c.startswith("@"):
                out.extend(groups.get(c[1:], []))
            else:
                out.append(c)
        return out

    def matrix(self, names: list[str], rows: Optional[np.ndarray] = None) -> np.ndarray:
        names = self.expand(names)
        if not names:
            return np.zeros((len(self.df) if rows is None else len(rows), 0), dtype=np.float32)
        parts = [self.block(n) for n in names]
        X = parts[0] if len(parts) == 1 else np.hstack(parts)
        return X if rows is None else X[rows]

    def cat_frame(self, cols: Optional[list[str]] = None) -> pd.DataFrame:
        cat, _ = self.meta()
        return cat if cols is None else cat[cols]

    def molecule_codes(self) -> np.ndarray:
        """Integer id of the molecule (InChIKey connectivity layer); -1 where standardisation failed."""
        return pd.factorize(self.df["ik14"])[0].astype(np.int64)

    def group_codes(self, col: str) -> np.ndarray:
        cat, _ = self.meta()
        return pd.factorize(cat[col])[0].astype(np.int64)

    # ------------------------------------------------------------------ kernels
    def kernel(self, name: str) -> np.ndarray:
        """Full n x n MinMax/Tanimoto kernel of a feature block (cached)."""
        if name in self._kernels:
            return self._kernels[name]
        spec = (self.cfg.get("kernels") or {}).get(name)
        if spec is None:
            raise KeyError(f"kernel {name!r} is not declared in features.kernels")
        bspec = self._block_spec(spec["block"])
        key = _digest(self._smiles_key, bspec, spec, FEATURE_CODE_VERSION)
        path = self.cache_dir / f"K-{name}-{key}.npy"
        if path.exists():
            K = np.load(path)
        else:
            log.info("building kernel %s from block %s", name, spec["block"])
            K = minmax_kernel(self.block(spec["block"]))
            np.save(path, K)
        self._kernels[name] = K
        return K


# =====================================================================================================  block types
def _fp_builder(fn):
    def build(store: FeatureStore, params: dict):
        return fn(store.mols(), **params)

    return build


register_block("morgan_counts")(_fp_builder(fp.morgan_counts))
register_block("atom_pair_counts")(_fp_builder(fp.atom_pair_counts))
register_block("torsion_counts")(_fp_builder(fp.torsion_counts))
register_block("rdkit_path_counts")(_fp_builder(fp.rdkit_path_counts))
register_block("maccs")(_fp_builder(fp.maccs))
register_block("avalon_counts")(_fp_builder(fp.avalon_counts))
register_block("erg")(_fp_builder(fp.erg))
register_block("mqn")(_fp_builder(fp.mqn))


@register_block("rdkit_desc")
def _rdkit_desc(store: FeatureStore, params: dict):
    return rdkit_descriptors(store.smiles_std(), n_jobs=store.n_jobs, **params)


@register_block("meta_num")
def _meta_num(store: FeatureStore, params: dict):
    _, num = store.meta()
    inc, exc = params.get("prefixes"), params.get("exclude_prefixes") or []
    cols = [c for c in num.columns
            if (not inc or any(c.startswith(p) for p in inc)) and not any(c.startswith(p) for p in exc)]
    return num[cols].values, cols


@register_block("meta_onehot")
def _meta_onehot(store: FeatureStore, params: dict):
    cat, _ = store.meta()
    cols = [c for c in (params.get("cols") or list(cat.columns)) if c not in (params.get("exclude_cols") or [])]
    min_count = int(params.get("min_count", 2))
    oh = pd.get_dummies(cat[cols], dtype=np.float32)
    oh = oh.loc[:, oh.sum(axis=0) >= min_count]
    return oh.values, list(oh.columns)


@register_block("mordred_desc")
def _mordred(store: FeatureStore, params: dict):  # optional heavy block (pip install mordredcommunity)
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    mols = store.mols()
    df = calc.pandas([m for m in mols], nproc=store.n_jobs, quiet=True)
    X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.isna().mean() < 0.1]
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 1e-9]
    big = X.abs().max() > 1e3
    X.loc[:, big] = np.sign(X.loc[:, big]) * np.log1p(X.loc[:, big].abs())
    return X.values, [f"mordred_{c}" for c in X.columns]
