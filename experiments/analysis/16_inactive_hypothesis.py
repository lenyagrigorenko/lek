"""Гипотеза «потолок = неактивное соединение»; метки соседей из внешнего пула: «потолок» против «неактивное».

Запуск из корня репозитория: python experiments/analysis/16_inactive_hypothesis.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from vacv_si.features.meta import smiles_style_features
from vacv_si.chem.fingerprints import morgan_counts
from vacv_si.chem.standardize import mols_from_std
from vacv_si.chem.kernels import minmax_kernel
df = pd.read_pickle("data/processed/all_ext.pkl")
df["can"] = smiles_style_features(df.smiles_raw).s_is_rdkit_can.values
tr = df[df.split=="train"]; ext = df[df.split=="ext"].reset_index(drop=True)
hard = tr[(tr.strain.str.strip().str.startswith("Vaccinia virus (strain Copenhagen). culture Vero")) & (tr.can==0)]
inact = lambda d: ((d.is_sentinel==1) | (d.si<=1.5)).astype(float)
hard = hard.assign(inactive=inact(hard).values)
print("hard-group train rows:", len(hard), "| sentinel %.2f | inactive(sentinel or SI<=1.5) %.2f | numeric-inactive (SI<=1.5, not sentinel): %d"%(hard.is_sentinel.mean(), hard.inactive.mean(), int(((hard.si<=1.5)&(hard.is_sentinel==0)).sum())))
print("P(sentinel | inactive) in the hard group: %.2f"%(hard.loc[hard.inactive==1,"is_sentinel"].mean()))
allin = tr.assign(inactive=inact(tr).values)
print("whole train: inactive share %.2f | P(sentinel|inactive) = %.2f | by canonical style: can=1 -> %.2f, can=0 -> %.2f"%(allin.inactive.mean(), allin.loc[allin.inactive==1,"is_sentinel"].mean(), allin.loc[(allin.inactive==1)&(allin.can==1),"is_sentinel"].mean(), allin.loc[(allin.inactive==1)&(allin.can==0),"is_sentinel"].mean()))
# neighbour label from the external pool: sentinel proxy vs inactive proxy
Xh,_ = morgan_counts(mols_from_std(hard.smiles_std),2,2048); Xe,_ = morgan_counts(mols_from_std(ext.smiles_std),2,2048)
K = minmax_kernel(Xh, Xe).astype(float)
K[hard.ik14.to_numpy(dtype=object)[:,None] == ext.ik14.to_numpy(dtype=object)[None,:]] = 0     # strict: never the same molecule
ext_sent = ext.is_sentinel.values.astype(float); ext_inact = ((ext.is_sentinel==1) | (ext.si<=1.5)).astype(float).values
def score(y):
    W = K**4; return (W@y)/(W.sum(1)+1e-9)
for tgt_name, tgt in [("Kaggle sentinel", hard.is_sentinel.values), ("Kaggle inactive", hard.inactive.values)]:
    for lab_name, y in [("ext sentinel-proxy", ext_sent), ("ext inactive-proxy", ext_inact)]:
        print(f"AUC of pool neighbour score [{lab_name}] for target [{tgt_name}] in the hard group: {roc_auc_score(tgt, score(y)):.3f}")
# same but using only well-covered rows (max-sim >= 0.6)
m = K.max(1) >= 0.6
print("rows with an external neighbour >=0.6:", int(m.sum()), "of", len(m))
for tgt_name, tgt in [("Kaggle sentinel", hard.is_sentinel.values), ("Kaggle inactive", hard.inactive.values)]:
    if len(set(tgt[m]))>1:
        print(f"  restricted AUC  [ext inactive-proxy] -> [{tgt_name}]: {roc_auc_score(tgt[m], score(ext_inact)[m]):.3f}  | [ext sentinel-proxy]: {roc_auc_score(tgt[m], score(ext_sent)[m]):.3f}")
