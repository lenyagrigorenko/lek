"""Отдельная калибровка вероятности «потолка» внутри сложной группы (Platt, сжатие): не помогает.

Запуск из корня репозитория: python experiments/analysis/15_regime_calibration.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from vacv_si.features.meta import smiles_style_features
df = pd.read_pickle("data/processed/all_ext.pkl"); df["can"] = smiles_style_features(df.smiles_raw).s_is_rdkit_can.values
tr = df[df.split=="train"].reset_index(drop=True); y = tr.is_sentinel.values.astype(int)
hard = ((tr.strain.str.strip().str.startswith("Vaccinia virus (strain Copenhagen). culture Vero")) & (tr.can==0)).values
p = pd.read_csv("reports/oof_final_final.csv").p.values
lg = lambda q: np.log(np.clip(q,1e-3,1-1e-3)/(1-np.clip(q,1e-3,1-1e-3)))
ph, yh = p[hard], y[hard]
print("in-group Brier of the current blend: %.4f"%np.mean((ph-yh)**2))
res = {"platt":[], "shrink":[]}
for r in range(30):
    pp = np.zeros(len(yh)); ps = np.zeros(len(yh))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=r).split(ph,yh):
        m = LogisticRegression(C=10).fit(lg(ph[a]).reshape(-1,1), yh[a]); pp[b] = m.predict_proba(lg(ph[b]).reshape(-1,1))[:,1]
        best = min(np.linspace(0,0.6,13), key=lambda l: np.mean(((1-l)*ph[a]+l*yh[a].mean()-yh[a])**2)); ps[b] = (1-best)*ph[b]+best*yh[a].mean()
    res["platt"].append(np.mean((pp-yh)**2)); res["shrink"].append(np.mean((ps-yh)**2))
for k,v in res.items(): print(f"nested-CV in-group Brier with {k:7s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
# which rows carry the in-group Brier?
e = (ph-yh)**2; o = np.argsort(-e)
print("top-6 rows carry %.0f%% of the in-group Brier; 4 worst: %s"%(100*e[o[:6]].sum()/e.sum(), [(tr.title[hard].iloc[i], int(yh[i]), round(float(ph[i]),2)) for i in o[:6]]))
