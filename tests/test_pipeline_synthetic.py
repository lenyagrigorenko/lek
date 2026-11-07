"""End-to-end checks on synthetic competition-format files (see conftest.py): they run anywhere, including CI."""
from __future__ import annotations

import numpy as np
import pytest

from vacv_si.cv.folds import make_fold_plan
from vacv_si.cv.runner import run_cv_one
from vacv_si.data.prepare import build_all_table
from vacv_si.data.schema import FORBIDDEN_COLUMNS, SENTINEL_VALUE, assert_no_forbidden
from vacv_si.features.store import FeatureStore


def _store(cfg, tmp_path):
    df = build_all_table(cfg)
    return df, FeatureStore(df, cfg["features"], cache_dir=tmp_path, n_jobs=1)


def test_forbidden_columns_never_enter_the_table(mini_cfg):
    df = build_all_table(mini_cfg)
    for c in FORBIDDEN_COLUMNS:
        assert c not in df.columns
    assert (df.split == "train").sum() == 120 and (df.split == "test").sum() == 40
    tr = df[df.split == "train"]
    assert ((tr["si"] == SENTINEL_VALUE) == (tr["is_sentinel"] == 1.0)).all()
    assert df.loc[df.split == "test", "si"].isna().all()
    assert sorted(df.loc[df.split == "test", "test_index"].astype(int)) == list(range(40))


def test_feature_blocks_kernel_and_groups(mini_cfg, tmp_path):
    df, store = _store(mini_cfg, tmp_path)
    for b in ["morgan2c", "desc", "meta_num", "meta_oh", "style_num"]:
        X = store.block(b)
        assert X.shape[0] == len(df) and np.isfinite(X).all()
        assert_no_forbidden(store.block_names(b))
    K = store.kernel("k_m2")
    assert K.shape == (len(df), len(df)) and np.allclose(K, K.T, atol=1e-6) and np.allclose(np.diag(K), 1.0, atol=1e-5)
    assert store.expand(["@meta", "@style"]) == ["meta_num", "meta_oh", "style_num"]
    assert store.source_col("@src") == "c_source_raw" and store.source_col(None) == "c_source_raw"
    # tier switch: chemistry-only configs point "same source" at a constant column
    store.cfg = {**store.cfg, "default_source_col": "c_const"}
    assert store.source_col("@src") == "c_const"


def test_fold_plan_is_a_partition(mini_cfg, tmp_path):
    df, store = _store(mini_cfg, tmp_path)
    plan = make_fold_plan(store.df, store.meta()[0]["c_source_norm"], {"n_splits": 4, "n_repeats": 2, "seed": 1})
    train_rows = set(np.flatnonzero((df.split == "train").values))
    for r in range(2):
        folds = [f for f in plan if f.repeat == r]
        val = np.concatenate([f.val for f in folds])
        assert len(val) == len(set(val)) == len(train_rows)
        for f in folds:
            assert not set(f.train) & set(f.val) and set(f.train) | set(f.val) == train_rows


@pytest.mark.parametrize("task", ["clf", "reg", "rawns", "raw"])
def test_cv_runner_smoke(mini_cfg, tmp_path, task):
    df, store = _store(mini_cfg, tmp_path)
    plan = make_fold_plan(store.df, store.meta()[0]["c_source_norm"], mini_cfg["cv"])
    spec = {"type": "extra_trees", "blocks": ["morgan2c", "@meta", "@style"], "params": {"n_estimators": 40, "n_jobs": 1}}
    res = run_cv_one(task, "et", spec, store, plan, mini_cfg)
    assert np.isfinite(res["oof"]).all() and res["test"].shape == (len(plan), 40)
    if task == "clf":                                   # the synthetic label is a function of the source strings
        assert res["summary"]["metrics"]["auc"] > 0.7
    assert res["summary"]["task"] == task


def test_knn_models_and_neighbour_pools(mini_cfg, tmp_path):
    df, store = _store(mini_cfg, tmp_path)
    plan = make_fold_plan(store.df, store.meta()[0]["c_source_norm"], mini_cfg["cv"])
    spec = {"type": "extra_trees", "blocks": ["@meta"], "knn": {"kernel": "k_m2", "ks": [1, 3]}, "params": {"n_estimators": 30, "n_jobs": 1}}
    res = run_cv_one("clf", "et_knn", spec, store, plan, mini_cfg)
    assert np.isfinite(res["oof"]).all()


def test_npz_embedding_block_aligns_by_smiles(mini_cfg, tmp_path):
    df = build_all_table(mini_cfg)
    uniq = sorted(set(df["smiles_std"].dropna()))
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(uniq), 12)).astype(np.float32)
    path = tmp_path / "emb.npz"
    np.savez_compressed(path, smiles=np.array(uniq, dtype=object), emb=emb)
    cfg = {**mini_cfg["features"], "blocks": {**mini_cfg["features"]["blocks"], "emb": {"type": "npz_embedding", "path": str(path)},
                                               "emb_pca": {"type": "npz_embedding", "path": str(path), "pca": 4}}}
    store = FeatureStore(df, cfg, cache_dir=tmp_path / "cache", n_jobs=1)
    X = store.block("emb")
    i = 5
    assert np.allclose(X[i], emb[uniq.index(df["smiles_std"].iloc[i])])
    assert store.block("emb_pca").shape == (len(df), 4)


def test_aux_pool_never_sees_the_molecule_itself(mini_cfg, tmp_path):
    import pandas as pd

    from conftest import SMILES_POOL

    df = build_all_table(mini_cfg)
    pool = pd.DataFrame({"SMILES": SMILES_POOL[:20], "CC50_uM": np.linspace(1, 500, 20)})   # contains molecules that are also in the table
    path = tmp_path / "pool.csv"
    pool.to_csv(path, index=False)
    cfg = {**mini_cfg["features"], "blocks": {**mini_cfg["features"]["blocks"],
           "aux": {"type": "aux_pool", "path": str(path), "kernel_block": "morgan2c", "targets": {"cc50": {"col": "CC50_uM"}}}}}
    store = FeatureStore(df, cfg, cache_dir=tmp_path / "cache", n_jobs=1)
    X, names = store.block("aux"), store.block_names("aux")
    assert X.shape == (len(df), 4) and np.isfinite(X).all() and names[0] == "aux_cc50_maxsim"
    inside = df["smiles_std"].isin(set(store.df.loc[store.df["smiles_std"].isin(SMILES_POOL[:20]), "smiles_std"]))
    assert inside.any() and (X[inside.values, 0] < 1.0 - 1e-6).all()      # an identical pool molecule is masked, so maxsim < 1
