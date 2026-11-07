"""Покрытие сложной группы внешними пулами (наш ChEMBL и чужой CSV).

Запуск из корня репозитория: python experiments/analysis/22_... <путь к csv>
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import sys, numpy as np, pandas as pd
from vacv_si.features.meta import smiles_style_features
from vacv_si.chem.fingerprints import morgan_counts
from vacv_si.chem.standardize import mols_from_std, standardize_series
from vacv_si.chem.kernels import minmax_kernel
df = pd.read_pickle("data/processed/all_ext.pkl")
sty = smiles_style_features(df.smiles_raw); df["can"] = sty.s_is_rdkit_can.values
kag = df[df.split.isin(["train","test"])].reset_index(drop=True)
hard = kag[(kag.strain.str.strip().str.startswith("Vaccinia virus (strain Copenhagen). culture Vero")) & (kag.can==0)]
print("hard group rows (non-canonical Copenhagen/Vero): train %d  test %d"%((hard.split=="train").sum(), (hard.split=="test").sum()))
ext = df[df.split=="ext"].drop_duplicates("ik14")
f = pd.read_csv(sys.argv[1]); st = standardize_series(f.SMILES.fillna(""), n_jobs=8); f["ik14"]=st["ik14"].values; f["smi"]=st["smiles_std"].values
fr = f.drop_duplicates("ik14")
Xh,_ = morgan_counts(mols_from_std(hard.smiles_std),2,2048)
for name, pool in [("OUR external pool", ext.smiles_std), ("friend's pool", fr.smi)]:
    X,_ = morgan_counts(mols_from_std(pool),2,2048); K = minmax_kernel(Xh, X)
    # exclude the very same molecule (strict policy): ik14 equality
    same = (hard.ik14.to_numpy(dtype=object)[:,None] == (ext.ik14 if name.startswith("OUR") else fr.ik14).to_numpy(dtype=object)[None,:]); K[same] = 0
    m = K.max(1)
    print(f"{name:18s} max-sim of hard-group molecules: quantiles 10/25/50/75/90% = {np.quantile(m,[.1,.25,.5,.75,.9]).round(2)} | share >=0.6: {(m>=0.6).mean():.2f}  >=0.8: {(m>=0.8).mean():.2f}")
# within Kaggle itself (train neighbours of the hard-group rows)
tr = kag[kag.split=="train"]; Xt,_ = morgan_counts(mols_from_std(tr.smiles_std),2,2048); Kt = minmax_kernel(Xh, Xt)
for i,(idx,r) in enumerate(hard.iterrows()):
    if r.split=="train": Kt[i, tr.index.get_loc(idx)] = 0
mt = Kt.max(1); print("Kaggle-train neighbours of hard-group rows: max-sim quantiles =", np.quantile(mt,[.1,.25,.5,.75,.9]).round(2))
