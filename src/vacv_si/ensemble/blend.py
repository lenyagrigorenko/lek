"""Level-2 blending of stored OOF predictions.

 clf : logit-space non-negative stack with intercept, minimising Brier (== the MSE the competition cares about)
 reg : non-negative linear stack of log10(SI) predictions on non-sentinel rows
 M(x): retransformation of log10(SI) to the *mean* SI of non-sentinel molecules (constant / smearing / isotonic)
All estimators here have very few parameters and are evaluated by repeated nested K-fold on the OOF predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def logit(p, eps: float = 1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------------------------------------------ classification
@dataclass
class ClfStack:
    w: np.ndarray
    b: float
    names: list = field(default_factory=list)

    def predict(self, P: np.ndarray) -> np.ndarray:           # P: (n, m) probabilities
        return sigmoid(self.b + logit(P) @ self.w)


def fit_clf_stack(P: np.ndarray, y: np.ndarray, l2: float = 0.0, loss: str = "brier") -> ClfStack:
    Z = logit(P)
    n, m = Z.shape

    def obj(theta):
        b, w = theta[0], theta[1:]
        p = sigmoid(b + Z @ w)
        if loss == "brier":
            L = np.mean((p - y) ** 2)
        else:
            L = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return L + l2 * np.sum((w - 1.0 / m) ** 2)

    x0 = np.r_[0.0, np.full(m, 1.0 / m)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=[(None, None)] + [(0.0, None)] * m)
    return ClfStack(w=res.x[1:], b=float(res.x[0]))


def nested_scores_clf(P: np.ndarray, y: np.ndarray, fit_fn, n_splits: int = 5, n_repeats: int = 5, seed: int = 0) -> np.ndarray:
    """Brier of `fit_fn(P_train, y_train).predict(P_val)` under repeated stratified K-fold -> array (n_repeats,)."""
    out = []
    for r in range(n_repeats):
        pred = np.zeros(len(y))
        for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(P, y):
            pred[b] = fit_fn(P[a], y[a]).predict(P[b])
        out.append(np.mean((pred - y) ** 2))
    return np.asarray(out)


class IsoMeanBlend:
    """Mean of selected models followed by a monotone (isotonic) recalibration, clipped away from 0 and 1."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, P, y):
        self.iso = IsotonicRegression(y_min=0.002, y_max=0.998, out_of_bounds="clip").fit(P[:, self.cols].mean(1), y)
        return self

    def predict(self, P):
        return self.iso.predict(P[:, self.cols].mean(1))


class TopKMeanBlend:
    """Rank models by Brier on the data it is fitted on, average the k best (selection is part of the fit)."""

    def __init__(self, k: int):
        self.k = k

    def fit(self, P, y):
        self.cols = list(np.argsort(((P - np.asarray(y)[:, None]) ** 2).mean(0))[: self.k])
        return self

    def predict(self, P):
        return P[:, self.cols].mean(1)


class TopKIsoBlend(TopKMeanBlend):
    """Top-k mean followed by a monotone (isotonic) recalibration, clipped away from 0 and 1."""

    def fit(self, P, y):
        super().fit(P, y)
        self.iso = IsotonicRegression(y_min=0.002, y_max=0.998, out_of_bounds="clip").fit(P[:, self.cols].mean(1), y)
        return self

    def predict(self, P):
        return self.iso.predict(P[:, self.cols].mean(1))


class TopKRegMean:
    """Regression analogue: the k models with the lowest squared error on the fitting data, averaged."""

    def __init__(self, k: int):
        self.k = k

    def fit(self, R, y):
        self.cols = list(np.argsort(((R - np.asarray(y)[:, None]) ** 2).mean(0))[: self.k])
        return self

    def predict(self, R):
        return R[:, self.cols].mean(1)


class MeanBlend:
    """Plain average of probabilities of a fixed model subset (no fitting)."""

    def __init__(self, cols=None):
        self.cols = cols

    def predict(self, P):
        return P[:, self.cols].mean(1) if self.cols is not None else P.mean(1)


# ------------------------------------------------------------------------------------------------------ regression
@dataclass
class RegStack:
    w: np.ndarray
    b: float

    def predict(self, R: np.ndarray) -> np.ndarray:
        return self.b + R @ self.w


def fit_reg_stack(R: np.ndarray, y: np.ndarray, l2: float = 0.0) -> RegStack:
    n, m = R.shape

    def obj(theta):
        b, w = theta[0], theta[1:]
        return np.mean((y - b - R @ w) ** 2) + l2 * np.sum((w - 1.0 / m) ** 2)

    x0 = np.r_[0.0, np.full(m, 1.0 / m)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=[(None, None)] + [(0.0, None)] * m)
    return RegStack(w=res.x[1:], b=float(res.x[0]))


# ------------------------------------------------------------------------------------------------------ retransformation
class ConstMean:
    """M(x) = mean SI of non-sentinel training rows."""

    def fit(self, mu, si, groups=None):
        self.c = float(np.mean(si))
        return self

    def __call__(self, mu, groups=None):
        return np.full(len(mu), self.c)


class Smearing:
    """Duan's smearing: M(x) = 10^mu(x) * mean(10^residual), residuals from OOF predictions."""

    def fit(self, mu, si, groups=None):
        res = np.log10(np.clip(si, 1e-2, None)) - mu
        self.s = float(np.mean(10 ** res))
        return self

    def __call__(self, mu, groups=None):
        return (10 ** np.asarray(mu)) * self.s


class RobustSmearing:
    """Duan smearing with a winsorised residual factor.

    S = mean(min(10^res, q-quantile of 10^res)): one extreme residual (a single SI = 36 066 record) must not
    inflate the retransformation factor of every other molecule. `cap_si` additionally caps SI before the residual
    is computed (consistent with training the raw-scale models on a capped target).
    """

    def __init__(self, q: float = 0.98, cap_si: float | None = None):
        self.q, self.cap_si = q, cap_si

    def fit(self, mu, si, groups=None):
        y = np.minimum(si, self.cap_si) if self.cap_si else si
        w = 10 ** (np.log10(np.clip(y, 1e-2, None)) - np.asarray(mu))
        self.s = float(np.minimum(w, np.quantile(w, self.q)).mean()) if self.q < 1 else float(w.mean())
        return self

    def __call__(self, mu, groups=None):
        return (10 ** np.asarray(mu)) * self.s


class IsoMean:
    """Monotone calibration of raw SI against the predicted log10(SI) (isotonic regression of the *mean*)."""

    def __init__(self, cap: float | None = None):
        self.cap = cap

    def fit(self, mu, si, groups=None):
        y = np.minimum(si, self.cap) if self.cap else si
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(mu, y)
        return self

    def __call__(self, mu, groups=None):
        return self.iso.predict(np.asarray(mu))


class PowerMean:
    """M(x) = a * 10^(b * mu): two-parameter log-link fit of the raw conditional mean (Poisson-like quasi-likelihood)."""

    def __init__(self, cap: float | None = None):
        self.cap = cap

    def fit(self, mu, si, groups=None):
        y = np.minimum(si, self.cap) if self.cap else si

        def obj(t):
            m = np.exp(t[0] + t[1] * np.log(10) * mu)
            return np.mean(m - y * np.log(m + 1e-12))     # Poisson deviance (robust for heavy right tails)

        r = minimize(obj, x0=[np.log(np.mean(y)), 1.0], method="L-BFGS-B", bounds=[(None, None), (0.0, 3.0)])
        self.t = r.x
        return self

    def __call__(self, mu, groups=None):
        return np.exp(self.t[0] + self.t[1] * np.log(10) * np.asarray(mu))


class GroupSmearing:
    """Duan smearing with a *group-specific* factor, shrunk towards the global one.

    M(x) = 10^mu(x) * S_g(x),   S_g = (sum_g 10^res + k * S_global) / (n_g + k).
    Heavy-tailed source series (a few records with SI in the thousands) need a much larger retransformation factor
    than light-tailed ones; with MSE on the raw SI this is where the mean prediction matters.
    """

    def __init__(self, k: float = 10.0):
        self.k = k

    def fit(self, mu, si, groups=None):
        w = 10 ** (np.log10(np.clip(si, 1e-2, None)) - np.asarray(mu))
        self.glob = float(w.mean())
        g = np.asarray(groups)
        self.S = {u: float((w[g == u].sum() + self.k * self.glob) / ((g == u).sum() + self.k)) for u in np.unique(g)}
        return self

    def __call__(self, mu, groups=None):
        g = np.asarray(groups)
        return (10 ** np.asarray(mu)) * np.array([self.S.get(u, self.glob) for u in g])


RETRANSFORMS = {
    "const": lambda: ConstMean(),
    "smear": lambda: Smearing(),
    "iso": lambda: IsoMean(),
    "iso_cap": lambda: IsoMean(cap=5000.0),
    "power": lambda: PowerMean(),
    "power_cap": lambda: PowerMean(cap=5000.0),
    "smear_w98": lambda: RobustSmearing(0.98),
    "smear_w95": lambda: RobustSmearing(0.95),
    "smear_cap5000": lambda: RobustSmearing(1.0, cap_si=5000.0),
    "smear_cap10000": lambda: RobustSmearing(1.0, cap_si=10000.0),
    "gsmear4": lambda: GroupSmearing(4.0),
    "gsmear10": lambda: GroupSmearing(10.0),
    "gsmear25": lambda: GroupSmearing(25.0),
}
