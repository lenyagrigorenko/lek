"""Tanimoto / MinMax kernels on non-negative count vectors and helpers for composite kernels.

For non-negative vectors a, b:   sum(min) = (sa + sb - L1)/2,  sum(max) = (sa + sb + L1)/2
so the MinMax (a.k.a. count Tanimoto) kernel is  K = (sa + sb - L1) / (sa + sb + L1)  and it reduces to the
ordinary Tanimoto kernel for binary fingerprints. It is positive semi-definite, K(x, x) = 1.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def minmax_kernel(A: np.ndarray, B: np.ndarray | None = None, chunk: int = 512) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    symmetric = B is None
    B = A if symmetric else np.asarray(B, dtype=np.float64)
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)), dtype=np.float32)
    for i in range(0, len(A), chunk):
        sl = slice(i, i + chunk)
        l1 = cdist(A[sl], B, metric="cityblock")
        tot = sa[sl, None] + sb[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            k = (tot - l1) / (tot + l1)
        k[(tot + l1) == 0] = 1.0           # two empty fingerprints
        K[sl] = k
    return K


def sharpen(K: np.ndarray, gamma: float) -> np.ndarray:
    """exp(gamma * (K - 1)) keeps positive semi-definiteness for gamma >= 0 and sharpens neighbourhoods."""
    return np.exp(gamma * (np.asarray(K, dtype=np.float64) - 1.0))


def same_group_matrix(codes_a: np.ndarray, codes_b: np.ndarray | None = None) -> np.ndarray:
    """Indicator kernel 1[g_i == g_j] (negative codes never match anything, not even themselves)."""
    codes_b = codes_a if codes_b is None else codes_b
    M = (codes_a[:, None] == codes_b[None, :]) & (codes_a[:, None] >= 0)
    return M.astype(np.float64)
