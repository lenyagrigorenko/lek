"""Validation of the assembled external dataset (a competition deliverable: "script ... and its validation").

Hard checks raise; soft findings are reported. A JSON report is written next to the dataset.
Cross-checks with the competition TRAIN labels use only the target column (`SI VACV`).
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from ..data.schema import EXTERNAL_REQUIRED, SENTINEL_VALUE
from ..paths import EXTERNAL

log = logging.getLogger(__name__)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def validate_external(ext: pd.DataFrame, cfg: dict, report_path=None, crosscheck: bool = True) -> dict:
    report: dict = {"n_rows": int(len(ext))}
    if ext.empty:
        raise ValueError("external dataset is empty")

    # --- hard checks ----------------------------------------------------------------------------------------------
    missing = [c for c in ("smiles", "IC50_uM", "CC50_uM", "SI", "doi") if c not in ext.columns]
    assert not missing, f"missing required columns {missing}"
    assert ext["smiles"].notna().all(), "records without SMILES"
    for c in ("IC50_uM", "CC50_uM"):
        v = ext[c].dropna()
        assert (v > 0).all(), f"{c} must be positive"
    ok = ext[["IC50_uM", "CC50_uM", "SI"]].notna().all(axis=1)
    ratio = ext.loc[ok, "CC50_uM"] / ext.loc[ok, "IC50_uM"]
    assert np.allclose(ratio, ext.loc[ok, "SI"], rtol=1e-6), "SI != CC50/IC50 for computed rows"

    # --- soft checks ----------------------------------------------------------------------------------------------
    doi = ext["doi"].fillna("")
    report["doi_present"] = float((doi != "").mean())
    report["doi_malformed"] = int(((doi != "") & ~doi.map(lambda d: bool(DOI_RE.match(d)))).sum())
    report["ic50_missing_rows"] = int(ext["ic50_missing"].sum()) if "ic50_missing" in ext else 0
    report["unique_molecules"] = int(ext["ik14"].nunique())
    report["unique_documents"] = int(ext["document_id"].nunique()) if "document_id" in ext else 0
    for c in ("IC50_uM", "CC50_uM", "SI"):
        q = ext[c].dropna().quantile([0, .01, .5, .99, 1]).round(4).tolist()
        report[f"{c}_quantiles(0,1,50,99,100%)"] = q
    report["si_gt_1e4"] = int((ext["SI"] > 1e4).sum())
    report["implausible_ic50(<1e-6 or >1e6 uM)"] = int(((ext["IC50_uM"] < 1e-6) | (ext["IC50_uM"] > 1e6)).sum())
    report["duplicated_keys"] = int(ext.duplicated(["ik14", "document_id", "assay_id", "cell_line"]).sum())

    # --- cross-check against the competition rows ------------------------------------------------------------------
    try:
        if not crosscheck:
            raise RuntimeError("crosscheck disabled")
        from ..data.prepare import load_all_table

        base_cfg = {**cfg, "external": {**cfg.get("external", {}), "use": False}}
        df = load_all_table(base_cfg)
        tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
        report["molecules_also_in_train"] = int(tr["ik14"].isin(set(ext["ik14"])).sum())
        report["molecules_also_in_test"] = int(te["ik14"].isin(set(ext["ik14"])).sum())
        report["frac_train_covered"] = round(report["molecules_also_in_train"] / len(tr), 3)
        report["frac_test_covered"] = round(report["molecules_also_in_test"] / len(te), 3)
        # agreement of the SI definition on train molecules (the only labelled rows we may look at)
        by_ik = {k: g for k, g in ext.groupby("ik14")}
        exact = close2 = n = sent_ok = sent_n = 0
        for _, r in tr.iterrows():
            g = by_ik.get(r["ik14"])
            if g is None:
                continue
            n += 1
            if r["is_sentinel"] == 1.0:
                sent_n += 1
                sent_ok += int(bool(g["ic50_missing"].any()))
                continue
            cand = g["SI"].dropna()
            if cand.empty:
                continue
            rel = np.abs(np.log10(np.clip(cand.values, 1e-3, None)) - np.log10(max(r["si"], 1e-3)))
            exact += int(rel.min() < np.log10(1.02))
            close2 += int(rel.min() < np.log10(2.0))
        report["train_overlap_si_agreement"] = {"n_train_molecules_matched": n, "non_sentinel_within_2pct": exact,
                                                "non_sentinel_within_x2": close2, "sentinel_matched": sent_n,
                                                "sentinel_with_missing_ic50_record": sent_ok}
    except Exception as e:  # noqa: BLE001
        report["crosscheck_error"] = str(e)[:200]

    path = report_path or (EXTERNAL / "validation_report.json")
    path.write_text(json.dumps(report, indent=1, default=float))
    log.info("validation report -> %s\n%s", path, json.dumps(report, indent=1, default=float))
    return report
