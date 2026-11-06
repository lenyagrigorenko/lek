"""Turn raw database dumps into ONE tidy external dataset:

    record_id, source, smiles, IC50_uM, CC50_uM, SI, doi, virus_strain, cell_line, ...

Conventions mirror the competition data (checked on molecules present in both, see `validate.py`):
  * SI = CC50 / IC50 computed from the numbers as reported (the sign of '>' / '<' relations is ignored),
  * IC50 / CC50 are in micromolar (ug/mL values are converted with the molecular weight),
  * a record whose antiviral activity is reported without a number ("inactive") but has a CC50 gets IC50 = NaN
    (`ic50_missing = True`) - in the competition table this is exactly the "SI = 10000" sentinel.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..chem.standardize import StdOptions, mols_from_std, standardize_series
from ..features.meta import norm_culture

log = logging.getLogger(__name__)

EXTERNAL_COLUMNS = [
    "record_id", "source", "smiles", "IC50_uM", "CC50_uM", "SI", "doi", "virus_strain", "cell_line",
    "ic50_type", "ic50_relation", "cc50_type", "cc50_relation", "ic50_missing", "si_reported",
    "document_id", "assay_id", "year", "journal", "ik14", "note",
]

UM = {"nM": 1e-3, "uM": 1.0, "µM": 1.0, "um": 1.0, "UM": 1.0, "mM": 1e3, "M": 1.0e6, "pM": 1e-6}
MASS_CONC = {"ug.mL-1", "ug/mL", "ug ml-1", "ug.ml-1", "mg/L", "mg.L-1", "ug mL-1"}

CYTO_RE = re.compile(r"cytotox|toxicit|viability|cytostatic|cell growth|growth inhibition|proliferation|cc50|mock[- ]infected|"
                     r"uninfected|minimum cytotoxic|toxic conc|cell morphology|cell alteration", re.I)
VIRAL_RE = re.compile(r"vaccinia|vacv|poxvirus|cowpox|virus[- ]induced|cytopath|antiviral|plaque|viral replication|infected", re.I)
INACTIVE_RE = re.compile(r"inactive|not active|no activity|not determined|nd\b|n\.d\.", re.I)


def _mw_table(smiles_std: pd.Series) -> dict:
    out = {}
    for s in smiles_std.dropna().unique():
        m = Chem.MolFromSmiles(s)
        out[s] = Descriptors.MolWt(m) if m is not None else np.nan
    return out


def to_uM(value, units, mw) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v):
        return np.nan
    if units in UM:
        return v * UM[units]
    if units in MASS_CONC and mw and np.isfinite(mw):
        return v / mw * 1000.0
    return np.nan


def _clean_smiles(df: pd.DataFrame, col: str) -> pd.DataFrame:
    std = standardize_series(df[col].fillna("").astype(str), StdOptions(), n_jobs=8)
    out = df.copy()
    out["smiles_std"] = std["smiles_std"].values
    out["ik14"] = std["ik14"].values
    return out[out["smiles_std"].notna()]


def _cell(desc: str, cell_type: Optional[str]) -> str:
    return norm_culture(f"{cell_type or ''} {desc or ''}")


def assemble_chembl(raw: dict) -> pd.DataFrame:
    assays, act, tox = raw["assays"], raw["activities"], raw["cytotox"]
    tox_assays, docs, mols = raw["cytotox_assays"], raw["documents"], raw["molecules"]
    if act.empty:
        return pd.DataFrame(columns=EXTERNAL_COLUMNS)

    # ---- molecules -------------------------------------------------------------------------------------------------
    smi = pd.concat([
        act[["molecule_chembl_id", "canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"}),
        tox[["molecule_chembl_id", "canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"}) if len(tox) else None,
        mols[["molecule_chembl_id", "molecule_smiles"]].rename(columns={"molecule_smiles": "smiles"}) if len(mols) else None,
    ], ignore_index=True).dropna().drop_duplicates("molecule_chembl_id")
    smi = _clean_smiles(smi, "smiles")
    mw = _mw_table(smi["smiles_std"])
    smi["mw"] = smi["smiles_std"].map(mw)
    smi = smi.set_index("molecule_chembl_id")

    # ---- antiviral potency records (vaccinia assays) -----------------------------------------------------------
    a_info = assays.set_index("assay_chembl_id")
    pot = act.copy()
    pot = pot[pot["molecule_chembl_id"].isin(smi.index)]
    pot["mw"] = pot["molecule_chembl_id"].map(smi["mw"])
    pot["val_uM"] = [to_uM(v, u, m) for v, u, m in zip(pot["standard_value"], pot["standard_units"], pot["mw"])]
    pot["assay_desc"] = pot["assay_chembl_id"].map(a_info["description"]).fillna(pot.get("assay_description", ""))
    pot["is_cyto_assay"] = pot["assay_desc"].fillna("").map(lambda d: bool(CYTO_RE.search(d)) and not re.search(r"vaccinia|vacv", d, re.I))
    pot = pot[~pot["is_cyto_assay"]]                       # drop cytotoxicity assays that merely mention the virus
    # A vaccinia record WITHOUT any number ("Activity", "Not Determined", "Not Active", ...) is what the competition
    # encodes as IC50 = NaN -> SI = 10000. Records whose number is in unusable units (e.g. % inhibition) are dropped.
    pot["has_value"] = pot["standard_value"].notna()
    pot["numeric"] = pot["val_uM"].notna()
    pot = pot[pot["numeric"] | ~pot["has_value"]]

    # ---- cytotoxicity records of the same documents ---------------------------------------------------------------
    if len(tox):
        t_info = tox_assays.set_index("assay_chembl_id") if len(tox_assays) else pd.DataFrame()
        tx = tox[tox["molecule_chembl_id"].isin(smi.index)].copy()
        tx["assay_desc"] = tx["assay_chembl_id"].map(t_info["description"]) if len(t_info) else tx["assay_description"]
        tx["assay_desc"] = tx["assay_desc"].fillna(tx["assay_description"]).fillna("")
        strong = tx["standard_type"].isin(["CC50", "TC50", "CTC50", "CD50", "TD50", "MCC", "CC90"])
        weak = tx["standard_type"].isin(["IC50", "EC50", "GI50", "LC50"]) & tx["assay_desc"].map(
            lambda d: bool(CYTO_RE.search(d)) and not re.search(r"vaccinia|vacv|virus[- ]induced|antiviral|infected", d, re.I))
        tx = tx[strong | weak]
        tx["mw"] = tx["molecule_chembl_id"].map(smi["mw"])
        tx["val_uM"] = [to_uM(v, u, m) for v, u, m in zip(tx["standard_value"], tx["standard_units"], tx["mw"])]
        tx = tx[tx["val_uM"].notna()]
        ctype = t_info["assay_cell_type"] if len(t_info) and "assay_cell_type" in t_info else pd.Series(dtype=object)
        tx["cell"] = [_cell(d, ctype.get(a)) for d, a in zip(tx["assay_desc"], tx["assay_chembl_id"])]
    else:
        tx = pd.DataFrame(columns=["document_chembl_id", "molecule_chembl_id", "val_uM", "cell", "standard_type", "standard_relation"])

    pot["cell"] = [_cell(d, a_info["assay_cell_type"].get(a) if "assay_cell_type" in a_info else None)
                   for d, a in zip(pot["assay_desc"], pot["assay_chembl_id"])]

    # ---- pair: per (document, molecule) pick the cytotoxicity of the same cell line, else the document median ----------
    tox_by = {k: g for k, g in tx.groupby(["document_chembl_id", "molecule_chembl_id"])}
    rows = []
    for (doc, mol, assay), g in pot.groupby(["document_chembl_id", "molecule_chembl_id", "assay_chembl_id"]):
        num = g[g["numeric"]]
        if len(num):
            # several values in one assay: geometric mean of exact values, else of whatever is there
            exact = num[num["standard_relation"].fillna("=") == "="]
            use = exact if len(exact) else num
            ic50 = float(np.exp(np.log(use["val_uM"].clip(lower=1e-9)).mean()))
            rel = "=" if len(exact) else str(num["standard_relation"].iloc[0])
            ic_type = ",".join(sorted(set(use["standard_type"].astype(str))))
            missing = False
        else:
            ic50, rel, ic_type, missing = np.nan, "", "", True
        cc = tox_by.get((doc, mol))
        cc50 = cc_type = cc_rel = np.nan
        if cc is not None and len(cc):
            same = cc[cc["cell"] == g["cell"].iloc[0]]
            use = same if len(same) else cc
            order = use.assign(_p=use["standard_type"].map({"CC50": 0, "TC50": 1, "CTC50": 1, "CD50": 1}).fillna(2)).sort_values("_p")
            use = order[order["_p"] == order["_p"].min()]
            cc50 = float(np.exp(np.log(use["val_uM"].clip(lower=1e-9)).mean()))
            cc_type = ",".join(sorted(set(use["standard_type"].astype(str))))
            cc_rel = str(use["standard_relation"].iloc[0])
        rows.append({"document_id": doc, "molecule_chembl_id": mol, "assay_id": assay, "IC50_uM": ic50, "CC50_uM": cc50,
                     "ic50_type": ic_type, "ic50_relation": rel, "cc50_type": cc_type, "cc50_relation": cc_rel,
                     "ic50_missing": missing, "cell_line": g["cell"].iloc[0], "assay_desc": g["assay_desc"].iloc[0],
                     "year": g["document_year"].iloc[0], "journal": g["document_journal"].iloc[0]})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=EXTERNAL_COLUMNS)
    out["SI"] = np.where(out["IC50_uM"].notna() & out["CC50_uM"].notna(), out["CC50_uM"] / out["IC50_uM"], np.nan)
    out["si_reported"] = False
    out["record_id"] = out["molecule_chembl_id"]
    out["source"] = "chembl"
    out["smiles"] = out["molecule_chembl_id"].map(smi["smiles_std"])
    out["ik14"] = out["molecule_chembl_id"].map(smi["ik14"])
    out["doi"] = out["document_id"].map(docs.set_index("document_chembl_id")["doi"]) if len(docs) else ""
    out["virus_strain"] = out["assay_id"].map(
        lambda a: " ".join(str(x) for x in (a_info["assay_organism"].get(a), a_info["assay_strain"].get(a)) if isinstance(x, str)))
    out["note"] = out["assay_desc"].fillna("").str.slice(0, 160)
    # keep rows that carry information about SI: both numbers, or a missing IC50 with a known CC50 (competition sentinel)
    keep = out["SI"].notna() | (out["ic50_missing"] & out["CC50_uM"].notna())
    # when the same molecule has a numeric IC50 in the same document the competition uses the number, so the
    # value-less record would only add label noise (checked on molecules present in both: 0/14 are sentinels)
    has_num = out.groupby(["document_id", "molecule_chembl_id"])["IC50_uM"].transform(lambda s: s.notna().any())
    keep &= ~(out["ic50_missing"] & has_num)
    return out.loc[keep, EXTERNAL_COLUMNS].reset_index(drop=True)


def assemble_external(cfg: dict, tables: dict) -> pd.DataFrame:
    frames = []
    if "chembl" in tables:
        frames.append(assemble_chembl(tables["chembl"]))
    if "pubchem" in tables:
        from .pubchem import assemble_pubchem

        frames.append(assemble_pubchem(tables["pubchem"]))
    ext = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame(columns=EXTERNAL_COLUMNS)
    ext = ext.drop_duplicates(subset=["ik14", "document_id", "assay_id", "cell_line", "IC50_uM", "CC50_uM"]).reset_index(drop=True)
    for c in ("IC50_uM", "CC50_uM", "SI"):                   # 8 significant digits: no floating-point noise in the csv
        ext[c] = ext[c].map(lambda v: float(f"{v:.8g}") if pd.notna(v) else v)
    from .validate import validate_external

    validate_external(ext, cfg)
    return ext
