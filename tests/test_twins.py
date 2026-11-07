"""The twin correction (labels of the same molecule / stereoisomer in the training pool) behaves as documented."""
from __future__ import annotations

import numpy as np
import pytest

from vacv_si.ensemble.twins import adjust, twin_lookup


def test_twin_lookup_excludes_self_and_empty_keys():
    keys = ["A", "A", "B", "", None]
    tw = twin_lookup(keys, keys, exclude_self=True)
    assert tw[0].tolist() == [1] and tw[1].tolist() == [0]
    assert len(tw[2]) == 0 and len(tw[3]) == 0 and len(tw[4]) == 0       # unique molecule / missing key: no twin
    q = twin_lookup(["A", "C"], keys, exclude_self=False)                 # test rows look into the training pool only
    assert q[0].tolist() == [0, 1] and len(q[1]) == 0


def test_twin_adjust_moves_towards_the_twin_label():
    mu, p = np.array([1.0, 1.0, 1.0]), np.array([0.1, 0.1, 0.1])
    twins = [np.array([0]), np.array([1]), np.array([], dtype=int)]
    pool_log, pool_zero = np.array([4.0, np.nan]), np.array([False, True])
    mu2, p2, n = adjust(mu, p, twins, pool_log, pool_zero, {"w": 0.5})
    assert n == 2
    assert mu2[0] == pytest.approx(0.5 * 1.0 + 0.5 * (0.12 + 0.89 * 4.0))        # active twin: log10 SI is pulled up
    assert p2[0] == pytest.approx(0.5 * 0.1 + 0.5 * 0.06)                        # ... and "inactive" becomes less likely
    assert mu2[1] == 1.0 and p2[1] == pytest.approx(0.5 * 0.1 + 0.5 * 0.875)     # inactive twin: p goes up, mu is left alone
    assert mu2[2] == 1.0 and p2[2] == 0.1                                        # no twin: unchanged
    mu3, p3, _ = adjust(mu, p, twins, pool_log, pool_zero, {"w": 0.0})
    assert np.allclose(mu3, mu) and np.allclose(p3, p)                           # weight 0 is the identity


def test_no_config_looks_up_evaluation_molecules():
    """The organisers forbid looking up the evaluation molecules in external databases: every config must drop their records."""
    from pathlib import Path

    from vacv_si.config import load_config

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs").rglob("*.yaml")):
        cfg = load_config(path)
        assert (cfg.get("external", {}) or {}).get("drop_overlap_with_eval", True) is True, path.name
