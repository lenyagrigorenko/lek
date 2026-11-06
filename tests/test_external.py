"""External-data assembly on tiny synthetic ChEMBL-like tables (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacv_si.external.assemble import EXTERNAL_COLUMNS, assemble_chembl, to_uM
from vacv_si.external.validate import validate_external


def test_unit_conversion():
    assert to_uM(2000, "nM", None) == pytest.approx(2.0)
    assert to_uM(5, "uM", None) == 5.0
    assert to_uM(1e-4, "M", None) == pytest.approx(100.0)
    assert to_uM(10, "ug.mL-1", 200.0) == pytest.approx(50.0)       # 10 ug/mL of a 200 g/mol compound = 50 uM
    assert np.isnan(to_uM(10, "ug.mL-1", None))                      # needs a molecular weight
    assert np.isnan(to_uM(55, "%", 200.0))                           # % inhibition is not a concentration
    assert np.isnan(to_uM(None, "nM", 100.0))


def _tables():
    act_cols = ["activity_id", "assay_chembl_id", "document_chembl_id", "molecule_chembl_id", "canonical_smiles", "standard_type",
                "standard_relation", "standard_value", "standard_units", "activity_comment", "assay_description",
                "document_year", "document_journal"]

    def row(i, assay, mol, smi, typ, rel, val, units, comment=None, desc=""):
        return dict(zip(act_cols, [i, assay, "CHEMBLDOC1", mol, smi, typ, rel, val, units, comment, desc, 2016, "J Med Chem"]))

    assays = pd.DataFrame({
        "assay_chembl_id": ["A_VACV"], "document_chembl_id": ["CHEMBLDOC1"], "assay_type": ["F"],
        "assay_organism": ["Orthopoxvirus vaccinia"], "assay_strain": [None], "assay_cell_type": ["Vero"],
        "assay_tissue": [None], "description": ["Antiviral activity against Vaccinia virus infected in Vero cells"],
        "target_chembl_id": ["CHEMBL613493"], "cell_chembl_id": [None]})
    desc = "Antiviral activity against Vaccinia virus infected in Vero cells"
    act = pd.DataFrame([
        row(1, "A_VACV", "M_OK", "CC(=O)Oc1ccccc1C(=O)O", "EC50", "=", 2000.0, "nM", desc=desc),                 # numeric
        row(2, "A_VACV", "M_MISSING", "c1ccccc1O", "Activity", None, np.nan, None, desc=desc),                      # no number
        row(3, "A_VACV", "M_BOTH", "c1ccccc1N", "EC50", "=", 500.0, "nM", desc=desc),                               # numeric ...
        row(4, "A_VACV", "M_BOTH", "c1ccccc1N", "Activity", None, np.nan, None, desc=desc),                         # ... + value-less
    ])
    tox_assays = pd.DataFrame({"assay_chembl_id": ["A_TOX"], "description": ["Cytotoxicity against Vero cells by MTT assay"],
                               "assay_cell_type": ["Vero"]})
    tdesc = "Cytotoxicity against Vero cells by MTT assay"
    tox = pd.DataFrame([row(10 + i, "A_TOX", m, smi, "CC50", "=", 100000.0, "nM", desc=tdesc)
                        for i, (m, smi) in enumerate([("M_OK", "CC(=O)Oc1ccccc1C(=O)O"), ("M_MISSING", "c1ccccc1O"), ("M_BOTH", "c1ccccc1N")])])
    docs = pd.DataFrame({"document_chembl_id": ["CHEMBLDOC1"], "doi": ["10.1021/jm000000x"], "pubmed_id": [1], "journal": ["J"], "year": [2016], "title": ["t"], "authors": ["a"]})
    mols = pd.DataFrame({"molecule_chembl_id": ["M_OK", "M_MISSING", "M_BOTH"], "pref_name": [None] * 3,
                         "molecule_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O", "c1ccccc1N"]})
    return {"assays": assays, "activities": act, "cytotox": tox, "cytotox_assays": tox_assays, "documents": docs, "molecules": mols}


def test_assemble_follows_the_competition_conventions():
    out = assemble_chembl(_tables())
    assert list(out.columns) == EXTERNAL_COLUMNS
    by = out.set_index("record_id")
    ok = by.loc["M_OK"]
    assert ok["IC50_uM"] == pytest.approx(2.0) and ok["CC50_uM"] == pytest.approx(100.0) and ok["SI"] == pytest.approx(50.0)
    assert ok["doi"] == "10.1021/jm000000x" and not ok["ic50_missing"]
    miss = by.loc["M_MISSING"]
    assert bool(miss["ic50_missing"]) and np.isnan(miss["IC50_uM"]) and miss["CC50_uM"] == pytest.approx(100.0)   # sentinel analogue
    both = out[out["record_id"] == "M_BOTH"]
    assert len(both) == 1 and not both["ic50_missing"].iloc[0]          # the value-less duplicate is dropped
    assert both["SI"].iloc[0] == pytest.approx(200.0)


def test_validation_catches_inconsistent_si(tmp_path):
    out = assemble_chembl(_tables())
    rep = validate_external(out, {}, report_path=tmp_path / "r.json", crosscheck=False)
    assert rep["n_rows"] == len(out) and rep["doi_malformed"] == 0
    bad = out.copy()
    bad.loc[bad["record_id"] == "M_OK", "SI"] = 999.0
    with pytest.raises(AssertionError):
        validate_external(bad, {}, report_path=tmp_path / "r2.json", crosscheck=False)
