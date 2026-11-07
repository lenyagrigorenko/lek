"""Тестовые молекулы, точно совпадающие со train: сколько и из одного ли источника.

Запуск из корня репозитория: python experiments/analysis/17_duplicates_test_vs_train.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
pd.set_option("display.width",230); pd.set_option("display.max_colwidth",50)
df = pd.read_pickle("data/processed/all_ext.pkl"); k = df[df.split!="ext"].reset_index(drop=True)
tr = k[k.split=="train"]; te = k[k.split=="test"]
sub = pd.read_csv("submissions/submission.csv").set_index("index")["SI VACV"]
te = te.assign(pred=sub.loc[te.test_index.astype(int)].values)
# 1) exact standardized-SMILES matches test <-> train
m = te.merge(tr[["smiles_std","title","strain","culture","si","is_sentinel"]], on="smiles_std", suffixes=("","_tr"))
m["same_source"] = (m.strain.str.strip()==m.strain_tr.str.strip()) & (m.culture.str.strip()==m.culture_tr.str.strip())
m["same_strain"] = (m.strain.str.strip()==m.strain_tr.str.strip())
print("test rows with an exact (standardised) SMILES match in train:", m.test_index.nunique(), "| pairs:", len(m), "| same source (strain+culture): %d"%m.same_source.sum())
print("same strain string:", int(m.same_strain.sum()), "of", len(m))
x = m[m.same_strain]
print(x[["title","title_tr","same_source","is_sentinel_tr","si_tr","pred"]].round(1).to_string(index=False))
# 2) how consistent are labels for the same molecule (ik14) inside train?
g = tr.groupby("ik14")
pairs = []
for ik, d in g:
    if len(d) > 1:
        for i in range(len(d)):
            for j in range(i+1, len(d)):
                a, b = d.iloc[i], d.iloc[j]
                pairs.append((a.is_sentinel==b.is_sentinel, a.strain.strip()==b.strain.strip(), np.log10(max(a.si,1e-2))-np.log10(max(b.si,1e-2))))
P = pd.DataFrame(pairs, columns=["same_sentinel","same_strain","dlog"])
print("\ntrain pairs with the same molecule connectivity (ik14):", len(P), "| same sentinel flag: %.2f"%P.same_sentinel.mean(), "| of which same strain string: %d, sentinel agree %.2f"%(P.same_strain.sum(), P[P.same_strain].same_sentinel.mean() if P.same_strain.any() else float('nan')))
