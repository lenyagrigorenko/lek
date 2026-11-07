"""Внешний CSV (чужой набор): пересечение с train/test по InChIKey и сходство с нашими молекулами.

Запуск из корня репозитория: python experiments/analysis/21_... <путь к csv>
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import sys, numpy as np, pandas as pd
from vacv_si.chem.standardize import standardize_series
pd.set_option("display.width",220); pd.set_option("display.max_colwidth",60)
f = pd.read_csv(sys.argv[1])
print("rows", len(f), "| unique Title", f.Title.nunique(), "| label_quality:", f.label_quality.value_counts().to_dict())
st = standardize_series(f.SMILES.fillna(""), n_jobs=8); f["ik14"] = st["ik14"].values; f["smi_std"] = st["smiles_std"].values
print("standardisation failures:", int(f.ik14.isna().sum()), "| unique molecules (ik14):", f.ik14.nunique())
df = pd.read_pickle("data/processed/all_ext.pkl")
kag = set(df[df.split.isin(["train","test"])].ik14); ours = set(df[df.split=="ext"].ik14)
print("overlap with Kaggle train/test molecules:", len(set(f.ik14) & kag), "| overlap with OUR external set:", len(set(f.ik14) & ours), "of", len(ours), "| new molecules vs ours:", len(set(f.ik14) - ours - kag))
for c in ["CC50_uM","IC50_uM_VACV","EC50_uM_VACV","Potency_uM_VACV","SI VACV"]:
    v = f[c].dropna(); print(f"{c:16s} n={len(v):5d}  molecules={f.loc[v.index,'ik14'].nunique():5d}  median={v.median():.3g}  min={v.min():.3g}  max={v.max():.3g}")
print("\nCulture:", f.Culture.fillna("(none)").value_counts().head(8).to_dict())
print("strain notna:", int(f.strain.notna().sum()))
# similarity of the friend's molecules to Kaggle molecules (could the pool help neighbours?)
from vacv_si.chem.fingerprints import morgan_counts
from vacv_si.chem.standardize import mols_from_std
from vacv_si.chem.kernels import minmax_kernel
fm = mols_from_std(f.drop_duplicates("ik14").smi_std); Xf,_ = morgan_counts(fm, 2, 2048)
km = df[df.split.isin(["train","test"])]; Xk,_ = morgan_counts(mols_from_std(km.smiles_std), 2, 2048)
K = minmax_kernel(Xk, Xf)
ms = K.max(1); print("\nmax Tanimoto (Morgan2 counts) of Kaggle molecules to the friend's pool: quantiles 10/25/50/75/90% =", np.quantile(ms,[.1,.25,.5,.75,.9]).round(3), "| share >=0.7: %.2f"%(ms>=0.7).mean(), "| >=0.5: %.2f"%(ms>=0.5).mean())
Ko = minmax_kernel(Xk, morgan_counts(mols_from_std(df[df.split=='ext'].drop_duplicates('ik14').smiles_std),2,2048)[0])
mo = Ko.max(1); print("same for OUR external pool: quantiles =", np.quantile(mo,[.1,.25,.5,.75,.9]).round(3), "| share >=0.7: %.2f"%(mo>=0.7).mean())
