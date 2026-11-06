"""Models that consume a dense feature matrix (tree ensembles, linear models, CatBoost, ...).

All of them can optionally append *neighbour features* (`knn:` section of the spec): similarity-weighted labels
of the nearest labelled training molecules, overall and inside the same source batch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..data.schema import assert_no_forbidden
from .base import BaseModel, neighbor_features

CLIP = 1e-4


class MatrixModel(BaseModel):
    """Base for models on `store.matrix(blocks)` (+ optional neighbour features)."""

    scale_default = False

    def __init__(self, store, task, spec, seed=0):
        super().__init__(store, task, spec, seed)
        self.blocks: list[str] = list(spec.get("blocks") or [])
        self.knn_spec: Optional[dict] = spec.get("knn")
        self.scale = bool(spec.get("scale", self.scale_default))
        self.scaler: Optional[StandardScaler] = None
        self._ref = None

    # ---- feature assembly ------------------------------------------------------------------------------------------
    def _set_ref(self, idx, y, w, ref):
        idx, y = np.asarray(idx), np.asarray(y, dtype=np.float64)
        r_idx = np.asarray(ref[0]) if ref is not None else np.zeros(0, dtype=np.int64)
        r_y = np.asarray(ref[1], dtype=np.float64) if ref is not None else np.zeros(0)
        prior = float(np.average(y, weights=w)) if w is not None else float(y.mean())
        self._ref = {"idx": idx, "y": y, "prior": prior,
                     "all_idx": np.concatenate([idx, r_idx]), "all_y": np.concatenate([y, r_y])}

    def _knn_block(self, idx: np.ndarray, training: bool) -> np.ndarray:
        """Neighbour-label features, computed separately for every reference pool (competition rows / external rows)."""
        ks, store = self.knn_spec, self.store
        codes = store.group_codes(store.source_col(ks.get("source_col")))
        ik = store.molecule_codes()
        split = store.df["split"].values
        ref_idx, ref_y = self._ref["all_idx"], self._ref["all_y"]
        feats = []
        for pool in (ks.get("pools") or [{"name": "all"}]):
            sel = np.ones(len(ref_idx), bool) if pool.get("rows") is None else (split[ref_idx] == pool["rows"])
            r_idx, r_y = ref_idx[sel], ref_y[sel]
            prior = float(r_y.mean()) if len(r_y) else self._ref["prior"]
            mask = idx[:, None] == r_idx[None, :]                      # never see yourself
            if pool.get("hide_same_molecule") and len(r_idx):
                mask = mask | ((ik[idx][:, None] == ik[r_idx][None, :]) & (ik[idx][:, None] >= 0))
            for kname in (ks.get("kernels") or [ks["kernel"]]):
                S = store.kernel(kname)[np.ix_(idx, r_idx)]
                F, _ = neighbor_features(S, r_y, codes[idx], codes[r_idx], ks=tuple(ks.get("ks", (1, 3, 8))),
                                         power=float(ks.get("power", 4.0)), prior=prior, self_mask=mask)
                feats.append(F)
        return np.hstack(feats) if len(feats) > 1 else feats[0]

    def _features(self, idx: np.ndarray, training: bool = False) -> np.ndarray:
        parts = []
        if self.blocks:
            parts.append(self.store.matrix(self.blocks, idx))
        if self.knn_spec:
            parts.append(self._knn_block(idx, training))
        X = np.hstack(parts) if len(parts) > 1 else parts[0]
        assert_no_forbidden(self.store.expand(self.blocks))
        return X

    def _prepare_fit(self, idx, y, w, ref=None):
        idx = np.asarray(idx)
        y = np.asarray(y, dtype=np.float64)
        if self.knn_spec:
            self._set_ref(idx, y, w, ref)
        X = self._features(idx, training=True)
        if self.scale:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        return X, y

    def _X_pred(self, idx):
        X = self._features(np.asarray(idx), training=False)
        return self.scaler.transform(X) if self.scaler is not None else X


class SklearnMatrixModel(MatrixModel):
    """Wraps (classifier, regressor) sklearn class pairs; `type` picks the pair."""

    FACTORY: dict[str, tuple] = {}
    supports_weight = True

    def _make(self):
        clf_cls, reg_cls = self.FACTORY[self.spec["type"]]
        cls = clf_cls if self.task == "clf" else reg_cls
        p = dict(self.params)
        if "random_state" in cls().get_params():
            p.setdefault("random_state", self.seed)
        return cls(**p)

    def fit(self, idx, y, w=None, ref=None):
        X, y = self._prepare_fit(idx, y, w, ref)
        self.est = self._make()
        try:
            self.est.fit(X, y.astype(int) if self.task == "clf" else y, sample_weight=w)
        except TypeError:
            self.est.fit(X, y.astype(int) if self.task == "clf" else y)
        return self

    def predict(self, idx):
        X = self._X_pred(idx)
        if self.task == "clf":
            p = self.est.predict_proba(X)[:, list(self.est.classes_).index(1)]
            return np.clip(p, CLIP, 1 - CLIP)
        return self.est.predict(X)


def _tree_factory(name):
    return {
        "extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor),
        "random_forest": (RandomForestClassifier, RandomForestRegressor),
        "hgb": (HistGradientBoostingClassifier, HistGradientBoostingRegressor),
        "logreg": (LogisticRegression, Ridge),
    }[name]


class TreeLike(SklearnMatrixModel):
    FACTORY = {k: _tree_factory(k) for k in ("extra_trees", "random_forest", "hgb", "logreg")}


class LinearLike(TreeLike):
    scale_default = True


# ------------------------------------------------------------------------------------------------------------------
class CatBoostModel(MatrixModel):
    """CatBoost with native handling of the categorical source columns (`cat_cols` of the meta frame)."""

    def __init__(self, store, task, spec, seed=0):
        super().__init__(store, task, spec, seed)
        self.cat_cols: list[str] = self.store.expand_cat(list(spec.get("cat_cols") or []))

    def _frame(self, idx, training=False) -> pd.DataFrame:
        parts = []
        if self.blocks or self.knn_spec:
            X = self._features(idx, training)
            parts.append(pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]))
        if self.cat_cols:
            parts.append(self.store.cat_frame(self.cat_cols).iloc[idx].reset_index(drop=True).astype(str))
        return pd.concat(parts, axis=1)

    def fit(self, idx, y, w=None, ref=None):
        from catboost import CatBoostClassifier, CatBoostRegressor

        idx = np.asarray(idx)
        y = np.asarray(y, dtype=np.float64)
        if self.knn_spec:
            self._set_ref(idx, y, w, ref)
        X = self._frame(idx, training=True)
        p = dict(self.params)
        p.setdefault("random_seed", self.seed)
        p.setdefault("thread_count", 1)
        p.setdefault("verbose", 0)
        p.setdefault("allow_writing_files", False)
        cls = CatBoostClassifier if self.task == "clf" else CatBoostRegressor
        if self.task == "clf":
            p.setdefault("loss_function", "Logloss")
        else:
            p.setdefault("loss_function", "RMSE")
        self.est = cls(cat_features=self.cat_cols or None, **p)
        self.est.fit(X, y.astype(int) if self.task == "clf" else y, sample_weight=w)
        return self

    def predict(self, idx):
        X = self._frame(np.asarray(idx), training=False)
        if self.task == "clf":
            return np.clip(self.est.predict_proba(X)[:, 1], CLIP, 1 - CLIP)
        return self.est.predict(X)


# ------------------------------------------------------------------------------------------------------------------
class _LGBMModel(MatrixModel):
    def fit(self, idx, y, w=None, ref=None):
        import lightgbm as lgb

        X, y = self._prepare_fit(idx, y, w, ref)
        p = dict(self.params)
        p.setdefault("random_state", self.seed)
        p.setdefault("n_jobs", 1)
        p.setdefault("verbose", -1)
        cls = lgb.LGBMClassifier if self.task == "clf" else lgb.LGBMRegressor
        self.est = cls(**p)
        self.est.fit(X, y.astype(int) if self.task == "clf" else y, sample_weight=w)
        return self

    def predict(self, idx):
        X = self._X_pred(idx)
        if self.task == "clf":
            return np.clip(self.est.predict_proba(X)[:, 1], CLIP, 1 - CLIP)
        return self.est.predict(X)


class _XGBModel(MatrixModel):
    def fit(self, idx, y, w=None, ref=None):
        import xgboost as xgb

        X, y = self._prepare_fit(idx, y, w, ref)
        p = dict(self.params)
        p.setdefault("random_state", self.seed)
        p.setdefault("n_jobs", 1)
        p.setdefault("verbosity", 0)
        cls = xgb.XGBClassifier if self.task == "clf" else xgb.XGBRegressor
        self.est = cls(**p)
        self.est.fit(X, y.astype(int) if self.task == "clf" else y, sample_weight=w)
        return self

    def predict(self, idx):
        X = self._X_pred(idx)
        if self.task == "clf":
            return np.clip(self.est.predict_proba(X)[:, 1], CLIP, 1 - CLIP)
        return self.est.predict(X)


MATRIX_MODELS = {
    "extra_trees": TreeLike,
    "random_forest": TreeLike,
    "hgb": TreeLike,
    "logreg": LinearLike,       # LogisticRegression for clf, Ridge for reg (scaled inputs)
    "catboost": CatBoostModel,
    "lightgbm": _LGBMModel,
    "xgboost": _XGBModel,
}
