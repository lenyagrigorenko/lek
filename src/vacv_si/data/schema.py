"""Column names and competition rules that the rest of the code must respect."""
from __future__ import annotations

TARGET = "SI VACV"
# Data release 2 (train_v2.csv, 7 Nov 2026): SI is exactly 0 for compounds without an antiviral IC50 (inactive against VACV)
# and for the two records without a CC50. The first release used 10000 for the inactive ones (an organisers' mistake: the
# value should have been in IC50, not in SI); `load_kaggle` refuses a file in that convention.
SENTINEL_VALUE = 0.0
OLD_SENTINEL_VALUE = 10000.0
SI_FLOOR = 1e-2               # SI is clipped to this before taking log10 (only non-zero SI is ever log-transformed)

# The organisers forbid using these columns to build models / predict the target
# ("the target is a function of these two columns"). They are dropped at load time and
# `assert_no_forbidden()` guards every feature matrix.
FORBIDDEN_COLUMNS = ("CC50_uM", "IC50_uM_VACV")

# Columns of the raw kaggle files that are legitimate model inputs.
INPUT_TEXT_COLUMNS = ("Title", "strain", "Culture")
SMILES_COLUMN = "SMILES"

# Required columns of the external dataset produced by `vacv_si.external` (competition requirement).
EXTERNAL_REQUIRED = ("smiles", "IC50_uM", "CC50_uM", "SI", "doi")


def assert_no_forbidden(columns) -> None:
    bad = [c for c in columns if any(str(c).strip().lower().startswith(f.lower()) for f in FORBIDDEN_COLUMNS)]
    if bad:
        raise AssertionError(f"forbidden columns reached the model inputs: {bad}")
