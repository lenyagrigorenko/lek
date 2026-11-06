"""Model interface used by the CV runner.

Every model works on *row indices* into the global FeatureStore (not on matrices):

    model.fit(idx, y, w)      # idx: global row indices of the training rows, y: labels, w: sample weights
    model.predict(idx)        # clf -> P(sentinel);  reg -> predicted log10(SI) of a non-sentinel record

This keeps the runner generic: tree models slice feature blocks, kernel models slice a pre-computed kernel
matrix, and "neighbour-feature" models query the kernel for labelled reference rows.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..features.store import FeatureStore


class BaseModel:
    def __init__(self, store: FeatureStore, task: str, spec: dict, seed: int = 0):
        assert task in ("clf", "reg", "rawns", "raw"), task
        self.store, self.task, self.spec, self.seed = store, task, spec, seed
        self.params = dict(spec.get("params") or {})

    def fit(self, idx: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
            ref: Optional[tuple] = None) -> "BaseModel":
        """`ref` = (ref_idx, ref_y): extra labelled rows usable as neighbours/reference points but NOT as training rows."""
        raise NotImplementedError

    def predict(self, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------------------------
def neighbor_features(S: np.ndarray, y_ref: np.ndarray, src_q: np.ndarray, src_ref: np.ndarray,
                      ks=(1, 3, 8), power: float = 4.0, prior: float = 0.0, self_mask: Optional[np.ndarray] = None
                      ) -> tuple[np.ndarray, list[str]]:
    """Label-aware similarity features for query rows against labelled reference rows.

    S: (q, r) similarities in [0, 1]. `self_mask` marks (query, reference) pairs that are the same row and must be
    ignored, which makes the features of training rows leave-one-out and hence consistent with test-time features.
    """
    S = np.array(S, dtype=np.float64)
    if S.shape[1] == 0:                      # empty reference pool -> neutral features (same columns as usual)
        _, names = neighbor_features(np.zeros((1, 2)), np.zeros(2), np.zeros(1, int), np.zeros(2, int), ks=ks, power=power)
        F = np.zeros((S.shape[0], len(names)), dtype=np.float32)
        F[:, names.index("src_mean_y")] = prior
        F[:, names.index("kr_y")] = prior
        return F, names
    if self_mask is not None:
        S[self_mask] = -1.0
    valid = S >= 0.0
    kmax = min(max(ks), S.shape[1])
    # top-kmax by similarity without sorting the whole row (O(r) instead of O(r log r))
    part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax] if kmax < S.shape[1] else np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
    srt = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
    top = np.take_along_axis(part, srt, 1)
    s_top = np.take_along_axis(S, top, 1)
    y_top = y_ref[top]
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, dtype=np.float64)); names.append(name)

    add("nn1_sim", s_top[:, 0])
    add("nn1_y", y_top[:, 0])
    for k in ks:
        k = min(k, kmax)
        w = np.clip(s_top[:, :k], 0, 1) ** power + 1e-9
        add(f"knn{k}_y", (w * y_top[:, :k]).sum(1) / w.sum(1))
        add(f"knn{k}_sim", s_top[:, :k].mean(1))
    W = np.where(valid, np.clip(S, 0, 1) ** power, 0.0)
    eps = 1e-3
    add("kr_y", (W @ y_ref + prior * eps) / (W.sum(1) + eps))
    same = (src_q[:, None] == src_ref[None, :]) & valid & (src_q[:, None] >= 0)
    cnt = same.sum(1)
    add("src_n", np.log1p(cnt))
    add("src_mean_y", np.where(cnt > 0, (same * y_ref[None, :]).sum(1) / np.maximum(cnt, 1), prior))
    Ws = W * same
    add("src_kr_y", (Ws @ y_ref + prior * 0.05) / (Ws.sum(1) + 0.05))
    return np.column_stack(cols).astype(np.float32), names
