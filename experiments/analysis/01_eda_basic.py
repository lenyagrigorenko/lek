"""Первичный EDA: размеры, пропуски, распределение SI, доля «потолка» 10000, пересечения train/test.

Запуск из корня репозитория: python experiments/analysis/01_eda_basic.py data/raw
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import sys, pandas as pd, numpy as np
d = sys.argv[1]
tr = pd.read_csv(f"{d}/train.csv"); te = pd.read_csv(f"{d}/test.csv"); ss = pd.read_csv(f"{d}/sample_submit.csv")
print("train", tr.shape, "test", te.shape, "sub", ss.shape)
print("\ntrain dtypes/nulls:\n", pd.concat([tr.dtypes, tr.isna().sum()], axis=1))
print("\ntest nulls:\n", te.isna().sum())
y = tr["SI VACV"]
print("\nSI describe:\n", y.describe(percentiles=[.05,.25,.5,.75,.9,.95,.99]))
print("\nSI == 10000:", (y==10000).sum(), " SI>=1000:", (y>=1000).sum(), " SI<1:", (y<1).sum(), " SI==1:", (y==1).sum())
print("log10(SI) describe:\n", np.log10(y.clip(lower=1e-3)).describe())
print("\nTop value counts SI:\n", y.value_counts().head(8))
print("\nsample_submit stats:\n", ss["SI VACV"].describe())
print("\nunique SMILES train:", tr.SMILES.nunique(), "/", len(tr), " test:", te.SMILES.nunique(), "/", len(te))
ov = set(tr.SMILES) & set(te.SMILES)
print("SMILES overlap train/test:", len(ov))
print("\nCulture train:\n", tr.Culture.str.strip().value_counts().head(10))
print("\nCulture test:\n", te.Culture.str.strip().value_counts().head(10))
# consistency: SI = CC50/IC50 ?
m = tr.IC50_uM_VACV.notna()
calc = tr.CC50_uM[m]/tr.IC50_uM_VACV[m]
print("\nIC50 present:", m.sum(), " SI ~ CC50/IC50 match:", np.isclose(calc, y[m], rtol=0.01).mean())
print("IC50 missing -> SI values:\n", y[~m].value_counts().head(5))
print("\nTest IC50 missing:", te.IC50_uM_VACV.isna().sum(), "of", len(te))
print("\nlen SMILES train/test:", tr.SMILES.str.len().describe()[["mean","50%","max"]].to_dict(), te.SMILES.str.len().describe()[["mean","50%","max"]].to_dict())
