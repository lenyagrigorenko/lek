"""Как «потолочные» молекулы train выглядят в ChEMBL: запись об активности без числа.

Запуск из корня репозитория: python experiments/analysis/10_sentinel_in_chembl.py   [после `python -m vacv_si collect`]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np
from vacv_si.chem.standardize import standardize_series
pd.set_option("display.width",260); pd.set_option("display.max_colwidth",70); pd.set_option("display.max_rows",200)
R="data/external/raw/"
act=pd.read_csv(R+"chembl_activities.csv",low_memory=False); mol=pd.read_csv(R+"chembl_molecules.csv")
tox=pd.read_csv(R+"chembl_cytotox.csv",low_memory=False)
st=standardize_series(mol.molecule_smiles.fillna(""),n_jobs=8); mol["ik14"]=st["ik14"].values
ik=dict(zip(mol.molecule_chembl_id,mol.ik14))
act["ik14"]=act.molecule_chembl_id.map(ik); tox["ik14"]=tox.molecule_chembl_id.map(ik)
df=pd.read_pickle("data/processed/all_ext.pkl"); tr=df[df.split=="train"]
sent=tr[(tr.is_sentinel==1)&tr.ik14.isin(set(act.ik14)|set(tox.ik14))]
print("sentinel train molecules found in ChEMBL activity/tox tables:", len(sent))
cols=["standard_type","standard_relation","standard_value","standard_units","activity_comment","assay_chembl_id"]
for _,r in sent.head(8).iterrows():
    print(f"\n=== {r.title} | {r.culture.strip()} | si={r.si} | {r.strain[:60]}")
    a=act[act.ik14==r.ik14]; t=tox[tox.ik14==r.ik14]
    print("  vaccinia potency records:", len(a)); 
    if len(a): print(a[cols].head(4).to_string(index=False))
    print("  cytotox-type records:", len(t))
    if len(t): print(t[["standard_type","standard_relation","standard_value","standard_units","assay_description"]].head(3).to_string(index=False))
# how are sentinel molecules distributed over doc/assays? and how many vaccinia potency records have no numeric value?
print("\npotency records without standard_value:", act.standard_value.isna().sum(), "of", len(act))
print(act[act.standard_value.isna()].activity_comment.value_counts().head(8).to_string())
print("\nrelation counts among potency records:\n", act.standard_relation.value_counts(dropna=False).to_string())
