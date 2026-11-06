"""Build the single, standardised table (`all`) that every later stage works on.

Rows: competition train, competition test and (optionally) external records.
The forbidden columns (CC50_uM, IC50_uM_VACV) are removed here and never reappear.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..chem.standardize import StdOptions, standardize_series
from ..paths import PROCESSED, ensure_dirs, resolve
from .schema import FORBIDDEN_COLUMNS, SENTINEL_VALUE, SI_FLOOR, TARGET

log = logging.getLogger(__name__)

ALL_COLUMNS = [
    "row_id", "split", "kind", "test_index", "title", "strain", "culture",
    "smiles_raw", "smiles_std", "smiles_flat", "inchikey", "ik14", "std_ok", "n_frag_raw",
    "si", "is_sentinel", "log_si", "weight", "doi",
]


def log_si_from_si(si: pd.Series | np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(si, dtype=float), SI_FLOOR, None))


def load_kaggle(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the competition files and drop the forbidden columns immediately."""
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    train = train.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in train.columns])
    test = test.drop(columns=[c for c in FORBIDDEN_COLUMNS if c in test.columns])
    return train, test


def _kaggle_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_id": [f"{split}_{i}" for i in range(len(df))],
            "split": split,
            "kind": "kaggle",
            "test_index": df["index"].astype(float) if "index" in df.columns else np.nan,
            "title": df["Title"].astype(str),
            "strain": df["strain"].fillna("").astype(str),       # RAW strings on purpose: whitespace quirks identify sources
            "culture": df["Culture"].fillna("").astype(str),
            "smiles_raw": df["SMILES"].astype(str),
            "doi": "",
        }
    )
    if TARGET in df.columns:
        out["si"] = df[TARGET].astype(float).values
    else:
        out["si"] = np.nan
    return out


def _external_frame(path: Path) -> pd.DataFrame:
    ext = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "row_id": [f"ext_{i}" for i in range(len(ext))],
            "split": "ext",
            "kind": ext.get("source", pd.Series("external", index=ext.index)).astype(str).values,
            "test_index": np.nan,
            "title": ext.get("record_id", pd.Series("", index=ext.index)).fillna("").astype(str).values,
            "strain": ext.get("virus_strain", pd.Series("", index=ext.index)).fillna("").astype(str).values,
            "culture": ext.get("cell_line", pd.Series("", index=ext.index)).fillna("").astype(str).values,
            "smiles_raw": ext["smiles"].astype(str).values,
            "si": ext["SI"].astype(float).values,
            "doi": ext.get("doi", pd.Series("", index=ext.index)).fillna("").astype(str).values,
        }
    )
    # a record with a CC50 but no antiviral IC50 is the analogue of the competition sentinel (SI := 10000)
    out["_ic50_missing"] = ext["ic50_missing"].fillna(False).astype(bool).values if "ic50_missing" in ext.columns else False
    return out


def build_all_table(cfg: dict, use_external: Optional[bool] = None) -> pd.DataFrame:
    ensure_dirs()
    raw_dir = resolve(cfg.get("paths", {}).get("raw_dir", "data/raw"))
    n_jobs = int(cfg.get("n_jobs", 8))
    tr, te = load_kaggle(raw_dir)
    parts = [_kaggle_frame(tr, "train"), _kaggle_frame(te, "test")]

    use_external = cfg.get("external", {}).get("use", False) if use_external is None else use_external
    ext_path = resolve(cfg.get("paths", {}).get("external_csv", "data/external/vacv_external.csv"))
    if use_external:
        if not ext_path.exists():
            raise FileNotFoundError(f"external.use=true but {ext_path} does not exist (run `vacv-si collect` first)")
        parts.append(_external_frame(ext_path))

    df = pd.concat(parts, ignore_index=True)
    std = standardize_series(df["smiles_raw"], StdOptions.from_cfg(cfg.get("data", {}).get("standardize")), n_jobs=n_jobs)
    for c in ["smiles_std", "smiles_flat", "inchikey", "ik14", "std_ok", "n_frag_raw"]:
        df[c] = std[c].values
    bad = ~df["std_ok"].fillna(False).astype(bool)
    if bad.any():
        log.warning("%d molecules failed standardisation (%s)", bad.sum(), df.loc[bad, "split"].value_counts().to_dict())

    # ---- targets ----------------------------------------------------------------------------------------------
    ext_mask = df["split"] == "ext"
    if "_ic50_missing" in df.columns:
        miss = df["_ic50_missing"].fillna(False).astype(bool) & ext_mask
        if cfg.get("external", {}).get("sentinel_from_missing_ic50", True):
            df.loc[miss, "si"] = SENTINEL_VALUE          # same convention as the organisers' table
        else:
            df = df[~miss]                                # alternative: drop value-less records altogether
            ext_mask = df["split"] == "ext"
        df = df.drop(columns=["_ic50_missing"]).reset_index(drop=True)
        ext_mask = df["split"] == "ext"
    df["is_sentinel"] = np.where(df["si"].isna(), np.nan, (df["si"] == SENTINEL_VALUE).astype(float))
    df["log_si"] = np.where(df["is_sentinel"] == 1.0, np.nan, log_si_from_si(df["si"]))
    df.loc[df["si"].isna(), "log_si"] = np.nan
    df["weight"] = 1.0
    df.loc[ext_mask, "weight"] = float(cfg.get("external", {}).get("weight", 0.3))
    return df[ALL_COLUMNS].reset_index(drop=True)


def processed_path(tag: str = "all") -> Path:
    return PROCESSED / f"{tag}.pkl"


def load_all_table(cfg: dict, rebuild: bool = False) -> pd.DataFrame:
    use_ext = bool(cfg.get("external", {}).get("use", False))
    tag = "all_ext" if use_ext else "all"
    path = processed_path(tag)
    ext_path = resolve(cfg.get("paths", {}).get("external_csv", "data/external/vacv_external.csv"))
    stale = path.exists() and use_ext and ext_path.exists() and ext_path.stat().st_mtime > path.stat().st_mtime
    if path.exists() and not rebuild and not stale:
        return pd.read_pickle(path)
    df = build_all_table(cfg, use_external=use_ext)
    df.to_pickle(path)
    df.to_csv(path.with_suffix(".csv"), index=False)
    return df
