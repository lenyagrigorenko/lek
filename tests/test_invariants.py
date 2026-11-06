"""Invariants the whole solution relies on. Run:  pytest"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacv_si.chem.kernels import minmax_kernel, same_group_matrix, sharpen
from vacv_si.chem.standardize import StdOptions, standardize_series
from vacv_si.cv.folds import make_fold_plan
from vacv_si.cv.metrics import combine
from vacv_si.data.schema import FORBIDDEN_COLUMNS, SENTINEL_VALUE, assert_no_forbidden
from vacv_si.models.base import neighbor_features
from vacv_si.paths import RAW

needs_data = pytest.mark.skipif(not (RAW / "train.csv").exists(), reason="competition files not in data/raw")


def test_forbidden_guard_raises():
    with pytest.raises(AssertionError):
        assert_no_forbidden(["morgan_1", "CC50_uM"])
    with pytest.raises(AssertionError):
        assert_no_forbidden(["ic50_uM_vacv_log"])
    assert_no_forbidden(["morgan_1", "desc_MolWt", "q_strain_len"])


@needs_data
def test_processed_table_has_no_forbidden_columns():
    from vacv_si.config import load_config
    from vacv_si.data.prepare import build_all_table

    df = build_all_table(load_config("configs/base.yaml"))
    for c in FORBIDDEN_COLUMNS:
        assert c not in df.columns
    assert set(df["split"]) == {"train", "test"}
    assert df.loc[df.split == "test", "si"].isna().all()
    # sentinel is exactly the value 10000
    tr = df[df.split == "train"]
    assert ((tr["si"] == SENTINEL_VALUE) == (tr["is_sentinel"] == 1.0)).all()


@needs_data
def test_feature_blocks_never_contain_forbidden_names():
    from vacv_si.config import load_config
    from vacv_si.data.prepare import build_all_table
    from vacv_si.features.store import FeatureStore

    cfg = load_config("configs/base.yaml")
    store = FeatureStore(build_all_table(cfg), cfg["features"])
    for b in ["meta_num", "meta_oh", "maccs", "mqn"]:
        assert_no_forbidden(store.block_names(b))


@needs_data
def test_fold_plan_is_a_partition_without_leakage():
    from vacv_si.config import load_config
    from vacv_si.data.prepare import build_all_table
    from vacv_si.features.meta import build_meta

    cfg = load_config("configs/base.yaml")
    df = build_all_table(cfg)
    cat, _ = build_meta(df)
    plan = make_fold_plan(df, cat["c_source_norm"], {"n_splits": 5, "n_repeats": 3, "seed": 1})
    train_rows = set(np.flatnonzero((df.split == "train").values))
    for r in range(3):
        folds = [f for f in plan if f.repeat == r]
        val = np.concatenate([f.val for f in folds])
        assert len(val) == len(set(val)) == len(train_rows)          # every row validated exactly once per repeat
        for f in folds:
            assert not set(f.train) & set(f.val)
            assert set(f.train) | set(f.val) == train_rows
            assert not set(f.val) & set(np.flatnonzero((df.split != "train").values))   # never touches test rows


def test_minmax_kernel_properties():
    rng = np.random.default_rng(0)
    A = rng.poisson(1.0, size=(40, 64)).astype(float)
    K = minmax_kernel(A)
    assert np.allclose(K, K.T, atol=1e-6) and np.allclose(np.diag(K), 1.0, atol=1e-6)
    assert np.linalg.eigvalsh(sharpen(K, 3.0)).min() > -1e-6        # PSD after sharpening
    # equals Tanimoto on binary vectors
    B = (A > 1).astype(float)
    inter, union = B @ B.T, B.sum(1)[:, None] + B.sum(1)[None, :] - B @ B.T
    with np.errstate(invalid="ignore"):
        T = np.where(union > 0, inter / union, 1.0)
    assert np.allclose(minmax_kernel(B), T, atol=1e-5)


def test_same_group_matrix_ignores_negative_codes():
    g = np.array([0, 0, 1, -1, -1])
    M = same_group_matrix(g)
    assert M[0, 1] == 1 and M[0, 2] == 0 and M[3, 4] == 0 and M[3, 3] == 0


def test_standardization_strips_salts_and_keeps_stereo():
    s = pd.Series(["CC(=O)O.[Na+]", "CC(=O)[O-].[Na+]", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "not a smiles"])
    out = standardize_series(s, StdOptions(), n_jobs=1)
    assert out.loc[0, "smiles_std"] == out.loc[1, "smiles_std"] == "CC(=O)O"          # acetate salt -> acetic acid
    assert out.loc[2, "smiles_std"] != out.loc[3, "smiles_std"]                      # enantiomers stay different ...
    assert out.loc[2, "ik14"] == out.loc[3, "ik14"]                                  # ... but share the connectivity key
    assert not bool(out.loc[4, "std_ok"])


def test_combine_is_the_conditional_mean():
    assert combine(np.array([0.0]), np.array([123.0]))[0] == 123.0
    assert combine(np.array([1.0]), np.array([123.0]))[0] == SENTINEL_VALUE
    assert combine(np.array([0.25]), np.array([100.0]))[0] == pytest.approx(0.25 * 10000 + 0.75 * 100)


def test_neighbor_features_are_leave_one_out():
    S = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    y = np.array([1.0, 0.0, 0.0])
    src = np.array([0, 0, 1])
    mask = np.eye(3, dtype=bool)
    F, names = neighbor_features(S, y, src, src, ks=(1,), self_mask=mask)
    i = names.index("nn1_y")
    assert F[0, i] == 0.0 and F[1, i] == 1.0           # nearest *other* molecule's label, never the row itself
    assert names.index("nn1_sim") >= 0 and F[0, names.index("nn1_sim")] == pytest.approx(0.9)
