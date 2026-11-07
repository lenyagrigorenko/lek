"""Metrics. The competition metric is MSE on the raw SI. SI is 0 for inactive compounds ("sentinel") and a heavy-tailed positive
number otherwise, so the MSE-optimal prediction is E[SI | x] = (1 - p_sentinel(x)) * E[SI | x, active] (see `combine`).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..data.schema import SENTINEL_VALUE


def mse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((y - p) ** 2))


def rmse(y, p) -> float:
    return float(np.sqrt(mse(y, p)))


def clf_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "rmse_log": rmse(y, p),
        "mae_log": float(np.mean(np.abs(y - p))),
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 else float("nan"),
    }


def proxy_raw_mse(si_true: np.ndarray, p_sentinel: np.ndarray, nonsent_mean: float) -> float:
    """Raw-scale MSE of  p*0 + (1-p)*c  (c = mean SI of non-sentinel rows): a fast ranking proxy for clf models."""
    pred = p_sentinel * SENTINEL_VALUE + (1.0 - p_sentinel) * nonsent_mean
    return mse(si_true, pred)


def combine(p_sentinel: np.ndarray, m_nonsent: np.ndarray) -> np.ndarray:
    """E[SI | x] = p * SENTINEL_VALUE (= 0) + (1 - p) * E[SI | x, non-sentinel]."""
    return p_sentinel * SENTINEL_VALUE + (1.0 - p_sentinel) * m_nonsent
