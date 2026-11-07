"""Synthetic competition-like files, so that the whole test-suite runs without the real Kaggle data (e.g. in CI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SMILES_POOL = [
    "CCO", "CCN", "CCC(=O)O", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "OC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O", "O=C1NC(=O)C=CN1", "CC1=CNC(=O)NC1=O",
    "C1CCCCC1", "C1CCNCC1", "CC(C)(C)c1ccccc1", "Clc1ccccc1", "Brc1ccc(cc1)C(=O)O", "COc1ccccc1", "CCOC(=O)C",
    "NC(=O)c1ccccn1", "O=C(O)CCC(=O)O", "CC(N)C(=O)O", "OCC(O)CO", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "CS(=O)(=O)N",
    "FC(F)(F)c1ccccc1", "O=[N+]([O-])c1ccccc1", "CC1(C)C2CCC1(C)C(=O)C2", "C1C2CC3CC1CC(C2)C3", "NC1CCCC1", "OC1CCCCC1",
    "CC(C)O", "c1ccncc1", "c1ccsc1", "c1ccoc1", "CC(=O)N", "CCCCCC", "CC(C)=O",
]
_SOURCES = [  # (strain, culture, P(sentinel))
    ("Vaccinia virus, HELA cells", "HELA cells", 0.95),
    ("Vaccinia virus (strain Copenhagen). culture Vero", "Vero", 0.30),
    ("Vaccinia virus, HFF cells", "Human foreskin fibroblast", 0.0),
    ("Vaccinia virus Tiantan strain", "Vero", 0.0),
]


def _make(n: int, split: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        strain, culture, p = _SOURCES[i % len(_SOURCES)]
        sent = bool(rng.random() < p)
        si = 0.0 if sent else float(np.round(10 ** rng.normal(1.0, 0.8), 3))      # release 2: inactive compounds have SI == 0
        cc50 = float(np.round(rng.uniform(10, 500), 2))
        rows.append({
            "Title": f"sid{103000000 + i}" if i % 2 == 0 else str(i),
            "CC50_uM": cc50, "IC50_uM_VACV": np.nan if sent else float(np.round(cc50 / si, 4)), "SI VACV": si,
            "SMILES": SMILES_POOL[(i * 7 + (0 if split == "train" else 3)) % len(SMILES_POOL)],
            "strain": strain, "Culture": culture,
        })
    df = pd.DataFrame(rows)
    if split == "test":
        df = df.drop(columns=["SI VACV"])
        df.insert(0, "index", np.arange(len(df)))
    return df


@pytest.fixture(scope="session")
def mini_raw(tmp_path_factory):
    """Directory with train_v2.csv / test.csv / sample_submit.csv in the real competition format (synthetic content)."""
    d = tmp_path_factory.mktemp("raw")
    _make(120, "train", 0).to_csv(d / "train_v2.csv", index=False)
    _make(40, "test", 1).to_csv(d / "test.csv", index=False)
    pd.DataFrame({"index": np.arange(40), "SI VACV": 1.0}).to_csv(d / "sample_submit.csv", index=False)
    return d


@pytest.fixture(scope="session")
def mini_cfg(mini_raw):
    return {
        "seed": 0, "n_jobs": 2,
        "paths": {"raw_dir": str(mini_raw)},
        "data": {"standardize": {}},
        "external": {"use": False},
        "cv": {"n_splits": 3, "n_repeats": 1, "seed": 0, "parallel": "threads", "full_fit_seeds": 0},
        "features": {
            "blocks": {
                "morgan2c": {"type": "morgan_counts", "radius": 2, "n_bits": 256},
                "desc": {"type": "rdkit_desc"},
                "meta_num": {"type": "meta_num", "exclude_prefixes": ["s_"]},
                "meta_oh": {"type": "meta_onehot", "min_count": 2, "exclude_cols": ["c_smiles_start2", "c_style_can", "c_source_style", "c_const"]},
                "style_num": {"type": "meta_num", "prefixes": ["s_"]},
            },
            "kernels": {"k_m2": {"block": "morgan2c"}},
            "groups": {"meta": ["meta_num", "meta_oh"], "style": ["style_num"], "meta_n": ["meta_num"], "style_n": ["style_num"]},
            "cat_groups": {"meta": ["c_source_raw", "c_title_pat"], "style": ["c_style_can"]},
            "default_source_col": "c_source_raw",
        },
    }
