"""ChEMBL collector: antiviral (vaccinia) potency records + cytotoxicity (CC50) records of the same publications.

Source: ChEMBL REST API  https://www.ebi.ac.uk/chembl/api/data  (open, CC BY-SA 3.0)

Pipeline
  1. find assays against vaccinia virus (organism / target / description match)
  2. download all their activities (IC50, EC50, ED50, ... + inactive/qualitative records)
  3. for the *documents* (papers) of those assays download cytotoxicity activities (CC50, TC50, cytotoxic IC50, ...)
  4. download document metadata (DOI, year, journal)
Everything is cached on disk (data/external/raw/chembl), the tidy tables are returned as DataFrames.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .http import cached_get

log = logging.getLogger(__name__)
BASE = "https://www.ebi.ac.uk/chembl/api/data"
VACV_TARGETS = ["CHEMBL613493", "CHEMBL613168"]          # Vaccinia virus (organism), Vaccinia virus WR
# Types that can serve as the antiviral "IC50". (qHTS "Potency" records, % inhibition and MIC are deliberately excluded:
# they are not paired with CC50 and one qHTS deposit alone has > 50 000 rows.)
POTENCY_TYPES = ["IC50", "EC50", "ED50", "IC90", "EC90", "Activity"]
CYTOTOX_TYPES = ["CC50", "TC50", "CTC50", "CC90", "LC50", "CD50", "IC50", "EC50", "GI50", "TD50", "MTC", "MCC", "Cytotoxicity"]


def _chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _paged(endpoint: str, key: str, params: dict, refresh: bool = False, page: int = 1000,
           max_records: int = 60000) -> list[dict]:
    out, offset = [], 0
    while True:
        if len(out) >= max_records:
            log.warning("%s %s: stopped after %d records (cap)", endpoint, {k: str(v)[:60] for k, v in params.items()}, len(out))
            break
        p = {**params, "format": "json", "limit": page, "offset": offset}
        d = cached_get(f"{BASE}/{endpoint}", p, refresh=refresh)
        items = d.get(key, [])
        out.extend(items)
        if not d.get("page_meta", {}).get("next"):
            break
        offset += page
    return out


def fetch_vacv_assays(refresh: bool = False) -> pd.DataFrame:
    frames = []
    queries = [
        {"assay_organism__icontains": "vaccinia"},
        {"target_chembl_id__in": ",".join(VACV_TARGETS)},
        {"description__icontains": "vaccinia"},
        {"description__icontains": "VACV"},
        {"assay_strain__icontains": "vaccinia"},
    ]
    for q in queries:
        items = _paged("assay", "assays", q, refresh)
        log.info("assays %s -> %d", q, len(items))
        frames.append(pd.DataFrame(items))
    a = pd.concat(frames, ignore_index=True).drop_duplicates("assay_chembl_id")
    keep = ["assay_chembl_id", "document_chembl_id", "assay_type", "assay_organism", "assay_strain", "assay_cell_type",
            "assay_tissue", "assay_tax_id", "description", "target_chembl_id", "cell_chembl_id", "confidence_score"]
    return a[[c for c in keep if c in a.columns]].reset_index(drop=True)


ACT_FIELDS = ["activity_id", "assay_chembl_id", "document_chembl_id", "molecule_chembl_id", "canonical_smiles",
              "standard_type", "standard_relation", "standard_value", "standard_units", "type", "relation", "value", "units",
              "activity_comment", "data_validity_comment", "potential_duplicate", "assay_description", "assay_type",
              "target_organism", "document_year", "document_journal", "pchembl_value"]


def _activity_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=ACT_FIELDS)
    df = pd.DataFrame(items)
    return df[[c for c in ACT_FIELDS if c in df.columns]]


def fetch_activities_for_assays(assay_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    frames = []
    for chunk in _chunks(sorted(set(assay_ids)), 20):
        items = _paged("activity", "activities", {"assay_chembl_id__in": ",".join(chunk),
                                                  "standard_type__in": ",".join(POTENCY_TYPES)}, refresh)
        frames.append(_activity_frame(items))
        log.info("activities for %d assays -> %d", len(chunk), len(items))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ACT_FIELDS)


def fetch_cytotox_for_documents(doc_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    frames = []
    for chunk in _chunks(sorted(set(doc_ids)), 5):
        items = _paged("activity", "activities", {"document_chembl_id__in": ",".join(chunk),
                                                  "standard_type__in": ",".join(CYTOTOX_TYPES)}, refresh, max_records=15000)
        frames.append(_activity_frame(items))
        log.info("cytotox candidates for %d documents -> %d", len(chunk), len(items))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ACT_FIELDS)


def fetch_documents(doc_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    rows = []
    for chunk in _chunks(sorted(set(doc_ids)), 50):
        rows.extend(_paged("document", "documents", {"document_chembl_id__in": ",".join(chunk)}, refresh))
    d = pd.DataFrame(rows)
    keep = ["document_chembl_id", "doi", "pubmed_id", "journal", "year", "title", "authors"]
    return d[[c for c in keep if c in d.columns]] if len(d) else pd.DataFrame(columns=keep)


def fetch_assays_by_id(assay_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    rows = []
    for chunk in _chunks(sorted(set(assay_ids)), 50):
        rows.extend(_paged("assay", "assays", {"assay_chembl_id__in": ",".join(chunk)}, refresh))
    a = pd.DataFrame(rows)
    keep = ["assay_chembl_id", "document_chembl_id", "assay_type", "assay_organism", "assay_strain", "assay_cell_type",
            "assay_tissue", "description", "target_chembl_id", "cell_chembl_id"]
    return a[[c for c in keep if c in a.columns]] if len(a) else pd.DataFrame(columns=keep)


def fetch_molecules(mol_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    """Canonical SMILES + names for molecules (the activity records only carry canonical_smiles)."""
    rows = []
    for chunk in _chunks(sorted(set(mol_ids)), 50):
        items = _paged("molecule", "molecules", {"molecule_chembl_id__in": ",".join(chunk)}, refresh)
        for m in items:
            st = m.get("molecule_structures") or {}
            rows.append({"molecule_chembl_id": m["molecule_chembl_id"], "pref_name": m.get("pref_name"),
                         "molecule_smiles": st.get("canonical_smiles")})
    return pd.DataFrame(rows)


def collect_chembl_raw(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download everything needed; returns raw tidy tables (not yet joined)."""
    assays = fetch_vacv_assays(refresh)
    log.info("vaccinia assays: %d (documents: %d)", len(assays), assays["document_chembl_id"].nunique())
    act = fetch_activities_for_assays(assays["assay_chembl_id"].tolist(), refresh)
    docs = sorted(set(assays["document_chembl_id"].dropna()))
    tox = fetch_cytotox_for_documents(docs, refresh)
    tox_assays = fetch_assays_by_id(tox["assay_chembl_id"].dropna().unique().tolist(), refresh) if len(tox) else pd.DataFrame()
    documents = fetch_documents(docs, refresh)
    mols = fetch_molecules(sorted(set(act["molecule_chembl_id"]) | set(tox.get("molecule_chembl_id", []))), refresh)
    return {"assays": assays, "activities": act, "cytotox": tox, "cytotox_assays": tox_assays, "documents": documents,
            "molecules": mols}
