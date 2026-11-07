"""Стиль записи SMILES (канонический RDKit или ChemDraw) и доля «потолка» внутри источника.

Запуск из корня репозитория: python experiments/analysis/06_smiles_style_provenance.py
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np, re
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
df=pd.read_pickle("data/processed/all_ext.pkl")
def feats(s):
    m=Chem.MolFromSmiles(s)
    can=Chem.MolToSmiles(m) if m is not None else ""
    can_noiso=Chem.MolToSmiles(m,isomericSmiles=False) if m is not None else ""
    return dict(
        is_rdkit_can = float(can==s),
        has_lower_arom = float(bool(re.search(r"[cnos]",re.sub(r"\[[^\]]*\]","",s)))),
        starts_C1 = float(s.startswith("C1")),
        starts_CC1 = float(s.startswith("CC1")),
        n_stereo_at = s.count("@"),
        has_slash = float("/" in s or "\\" in s),
        has_ring_digit_gt1 = float(bool(re.search(r"[%0-9]",s)) and bool(re.search(r"[3-9]",s))),
        explicit_H = float("[H]" in s or "[nH]" in s),
        len = len(s))
F=pd.DataFrame([feats(s) for s in df.smiles_raw]); D=pd.concat([df[["split","strain","is_sentinel","title"]].reset_index(drop=True),F],axis=1)
tr=D[D.split=="train"]
print("overall sentinel vs is_rdkit_canonical:"); print(tr.groupby("is_rdkit_can").is_sentinel.agg(["size","mean"]).round(3))
print("\ntest share is_rdkit_can:", D[D.split=="test"].is_rdkit_can.mean().round(3), " train:", tr.is_rdkit_can.mean().round(3))
cop=tr[tr.strain.str.strip()=="Vaccinia virus (strain Copenhagen). culture Vero"]
print("\nCopenhagen/Vero group (n=%d):"%len(cop))
for c in ["is_rdkit_can","has_lower_arom","starts_C1","starts_CC1","has_slash","explicit_H"]:
    g=cop.groupby(c).is_sentinel.agg(["size","mean"]).round(2); print(c, g.to_dict("index"))
print("\nstyle by source group (train): share RDKit-canonical / sentinel rate when canonical vs not")
g=tr.groupby("strain").apply(lambda x: pd.Series(dict(n=len(x), can=x.is_rdkit_can.mean(), sent_can=x.loc[x.is_rdkit_can==1,"is_sentinel"].mean(), sent_non=x.loc[x.is_rdkit_can==0,"is_sentinel"].mean())))
print(g.sort_values("n",ascending=False).head(8).round(2).to_string())
