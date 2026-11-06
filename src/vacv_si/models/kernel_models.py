"""Kernel methods on Tanimoto/MinMax similarities (very strong on small, series-structured chemistry data).

The kernel of a model is composite and declared in its spec:

    kernel:
      base: k_m2                 # a kernel from features.kernels
      gamma: 3.0                 # K <- exp(gamma * (K - 1))   (stays positive semi-definite)
      mult: {c_source_raw: 1.0}  # K <- K * (1 + a * same_group)  (molecule similarity matters most inside a source)
      add:  {c_source_raw: 0.3}  # K <- K + b * same_group        (shared source prior)
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from scipy.linalg import solve
from sklearn.svm import SVC, SVR

from ..chem.kernels import same_group_matrix, sharpen
from .base import BaseModel

CLIP = 1e-4
_KCACHE: dict[str, np.ndarray] = {}


def composite_kernel(store, kspec: dict) -> np.ndarray:
    key = f"{id(store)}:{store.cfg.get('default_source_col')}:{json.dumps(kspec, sort_keys=True)}"
    if key in _KCACHE:
        return _KCACHE[key]
    K = store.kernel(kspec["base"]).astype(np.float64)
    g = float(kspec.get("gamma", 0.0))
    if g > 0:
        K = sharpen(K, g)
    for col, a in (kspec.get("mult") or {}).items():
        K = K * (1.0 + float(a) * same_group_matrix(store.group_codes(store.source_col(col) if col == "@src" else col)))
    for col, b in (kspec.get("add") or {}).items():
        K = K + float(b) * same_group_matrix(store.group_codes(store.source_col(col) if col == "@src" else col))
    if len(_KCACHE) > 64:
        _KCACHE.clear()
    _KCACHE[key] = K
    return K


class KernelModel(BaseModel):
    def __init__(self, store, task, spec, seed=0):
        super().__init__(store, task, spec, seed)
        self.K = composite_kernel(store, spec["kernel"])

    def _sub(self, a, b):
        return self.K[np.ix_(a, b)]


class KernelRidgeModel(KernelModel):
    """Kernel ridge regression on 0/1 labels (clf) or log-SI (reg); closed form, no tuning of iterations."""

    def fit(self, idx, y, w=None, ref=None):
        self.idx = np.asarray(idx)
        y = np.asarray(y, dtype=np.float64)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
        self.mean = float(np.average(y, weights=w))
        lam = float(self.params.get("alpha", 0.3))
        A = self._sub(self.idx, self.idx) + lam * np.diag(1.0 / np.clip(w, 1e-3, None))
        self.coef = solve(A, y - self.mean, assume_a="pos")
        return self

    def predict(self, idx):
        f = self.mean + self._sub(np.asarray(idx), self.idx) @ self.coef
        if self.task == "clf":
            return np.clip(f, 0.005, 0.995)
        return f


class KernelSVMModel(KernelModel):
    def fit(self, idx, y, w=None, ref=None):
        self.idx = np.asarray(idx)
        y = np.asarray(y, dtype=np.float64)
        Ktt = self._sub(self.idx, self.idx)
        if self.task == "clf":
            self.est = SVC(kernel="precomputed", C=float(self.params.get("C", 3.0)), class_weight=self.params.get("class_weight"))
            self.est.fit(Ktt, y.astype(int), sample_weight=w)
        else:
            self.est = SVR(kernel="precomputed", C=float(self.params.get("C", 3.0)), epsilon=float(self.params.get("epsilon", 0.1)))
            self.est.fit(Ktt, y, sample_weight=w)
        return self

    def predict(self, idx):
        d = self.est.decision_function(self._sub(np.asarray(idx), self.idx)) if self.task == "clf" else None
        if self.task == "clf":
            return np.clip(1.0 / (1.0 + np.exp(-float(self.params.get("slope", 2.0)) * d)), CLIP, 1 - CLIP)
        return self.est.predict(self._sub(np.asarray(idx), self.idx))


class KernelNWModel(KernelModel):
    """Nadaraya-Watson / similarity-weighted neighbours with a shrinkage prior."""

    def fit(self, idx, y, w=None, ref=None):
        self.idx = np.asarray(idx)
        self.y = np.asarray(y, dtype=np.float64)
        self.w = np.ones_like(self.y) if w is None else np.asarray(w, dtype=np.float64)
        self.prior = float(np.average(self.y, weights=self.w))
        return self

    def predict(self, idx):
        S = self._sub(np.asarray(idx), self.idx)
        lam = float(self.params.get("prior_strength", 0.3))
        Sw = S * self.w[None, :]
        f = (Sw @ self.y + lam * self.prior) / (Sw.sum(1) + lam)
        return np.clip(f, 0.005, 0.995) if self.task == "clf" else f


KERNEL_MODELS = {
    "kernel_ridge": KernelRidgeModel,
    "kernel_svm": KernelSVMModel,
    "kernel_nw": KernelNWModel,
}
