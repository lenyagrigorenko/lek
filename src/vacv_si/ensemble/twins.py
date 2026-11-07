"""Twin correction: use the labels of the *same molecule* (or its stereoisomer) in the training pool.

A "twin" of a row is another labelled training row with the same InChIKey connectivity block (`ik14`): a duplicate record of
the molecule, or one of its stereoisomers. Tree/kernel models see the twin only as one neighbour among many, so its (possibly
extreme) label is diluted. Data say the twin label is worth much more than that:

  * external records (ChEMBL; only molecules that are NOT in the competition, so no eval molecule is involved), stereoisomer
    pairs of the same paper: log10 SI of the partner = 0.12 + 0.89 * own log10 SI, residual sd 0.63 (n = 30, correlation 0.90);
    the activity status agrees in 22 of 23 pairs (both active 15, both inactive 7, one inactive 1);
  * competition train, pairs of identical molecules: 6 of 9 agree within a factor of 2 (`experiments/analysis/23_twin_pairs.py`).

The model's own log10 error is 0.68 (sd), the twin's 0.63, so the two are about equally informative; the correction is the
precision-weighted average with a cautious weight  w = 0.5:

    mu'  = (1 - w) * mu + w * (a + b * mean log10 SI of the active twins)          (only if some twin is active)
    p'   = (1 - w) * p  + w * q                                                    q = P(inactive | twins' status)

Twins are looked up in the training pool only (never in external data), with the row itself excluded, so the same function gives
leave-one-out corrections for train rows (honest out-of-fold scores) and corrections for test rows.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {"use": True, "w": 0.5, "a": 0.12, "b": 0.89, "q_active": 0.06, "q_zero": 0.875, "q_mixed": 0.5}


def twin_lookup(keys_query, keys_pool, exclude_self: bool) -> list[np.ndarray]:
    """For every query row the indices (into the pool) of pool rows with the same non-empty key; the row itself is skipped
    when query and pool are the same table."""
    groups: dict[str, list[int]] = {}
    for i, k in enumerate(keys_pool):
        if isinstance(k, str) and k:
            groups.setdefault(k, []).append(i)
    out = []
    for i, k in enumerate(keys_query):
        idx = [j for j in groups.get(k, []) if not (exclude_self and j == i)] if isinstance(k, str) and k else []
        out.append(np.array(idx, dtype=int))
    return out


def adjust(mu: np.ndarray, p: np.ndarray, twins: list[np.ndarray], pool_log_si: np.ndarray, pool_is_zero: np.ndarray, params: dict | None = None):
    """Apply the correction; returns (mu', p', n_changed). `pool_log_si` is NaN for inactive rows."""
    prm = {**DEFAULTS, **(params or {})}
    mu2, p2 = np.array(mu, dtype=float), np.array(p, dtype=float)
    changed = 0
    for i, idx in enumerate(twins):
        if len(idx) == 0:
            continue
        zero = pool_is_zero[idx].astype(bool)
        q = prm["q_zero"] if zero.all() else (prm["q_active"] if not zero.any() else prm["q_mixed"])
        p2[i] = (1.0 - prm["w"]) * p2[i] + prm["w"] * q
        act = idx[~zero & np.isfinite(pool_log_si[idx])]
        if len(act):
            mu2[i] = (1.0 - prm["w"]) * mu2[i] + prm["w"] * (prm["a"] + prm["b"] * float(np.mean(pool_log_si[act])))
        changed += 1
    return mu2, p2, changed
