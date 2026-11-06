"""Repeated stratified K-fold over the *competition training rows*.

The public test split is (as the EDA shows) a random split of the same pool: nearest-neighbour similarity
test->train matches train->train leave-one-out. Plain random K-fold is therefore the right validation scheme, and we
stratify by (sentinel x source) so that every fold sees every source batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class Fold:
    repeat: int
    fold: int
    train: np.ndarray    # global row indices (competition train rows)
    val: np.ndarray


def strata(is_sentinel: np.ndarray, source: pd.Series, n_splits: int) -> np.ndarray:
    y = pd.Series(is_sentinel.astype(int).astype(str), index=source.index)
    counts = source.map(source.value_counts())
    src = source.where(counts >= 2 * n_splits, "rare")
    lab = y + "|" + src
    small = lab.map(lab.value_counts()) < n_splits
    lab = lab.where(~small, y + "|rare")
    small = lab.map(lab.value_counts()) < n_splits
    lab = lab.where(~small, y)
    return lab.values


def make_fold_plan(df: pd.DataFrame, source: pd.Series, cv_cfg: dict) -> list[Fold]:
    train_rows = np.flatnonzero((df["split"] == "train").values)
    y = df.loc[train_rows, "is_sentinel"].values
    n_splits, n_repeats, seed = int(cv_cfg["n_splits"]), int(cv_cfg["n_repeats"]), int(cv_cfg.get("seed", 42))
    if cv_cfg.get("stratify", "sentinel_source") == "sentinel":
        lab = y.astype(int)
    else:
        lab = strata(y, source.iloc[train_rows].reset_index(drop=True), n_splits)
    plan = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (a, b) in enumerate(skf.split(np.zeros(len(train_rows)), lab)):
            plan.append(Fold(r, k, train_rows[a], train_rows[b]))
    return plan
