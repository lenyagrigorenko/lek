"""Orchestrates the external-data collection:  raw downloads -> tidy per-source tables -> one validated CSV.

    python -m vacv_si collect                 # all sources
    python -m vacv_si collect --sources chembl
"""
from __future__ import annotations

import logging

import pandas as pd

from ..paths import EXTERNAL, EXTERNAL_RAW, ensure_dirs, resolve

log = logging.getLogger(__name__)


def collect_all(cfg: dict, sources=None, refresh: bool = False) -> None:
    ensure_dirs()
    sources = sources or ["chembl"]
    tables = {}
    if "chembl" in sources:
        from .chembl import collect_chembl_raw

        raw = collect_chembl_raw(refresh)
        for k, v in raw.items():
            v.to_csv(EXTERNAL_RAW / f"chembl_{k}.csv", index=False)
            log.info("chembl_%s: %d rows", k, len(v))
        tables["chembl"] = raw
    if "pubchem" in sources:
        from .pubchem import collect_pubchem_raw

        raw = collect_pubchem_raw(refresh)
        for k, v in raw.items():
            v.to_csv(EXTERNAL_RAW / f"pubchem_{k}.csv", index=False)
        tables["pubchem"] = raw

    from .assemble import assemble_external

    out = assemble_external(cfg, tables)
    path = resolve(cfg.get("paths", {}).get("external_csv", "data/external/vacv_external.csv"))
    out.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(out))
