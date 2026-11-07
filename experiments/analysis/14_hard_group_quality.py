"""Качество каждой модели внутри сложной группы (Brier, AUC) и доля общей ошибки.

Запуск из корня репозитория: python experiments/analysis/14_hard_group_quality.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from vacv_si.features.meta import smiles_style_features
df = pd.read_pickle("data/processed/all_ext.pkl"); df["can"] = smiles_style_features(df.smiles_raw).s_is_rdkit_can.values
tr = df[df.split=="train"].reset_index(drop=True); y = tr.is_sentinel.values
hard = ((tr.strain.str.strip().str.startswith("Vaccinia virus (strain Copenhagen). culture Vero")) & (tr.can==0)).values
print("hard rows %d (sentinel %.2f) | rest rows %d (sentinel %.2f)"%(hard.sum(), y[hard].mean(), (~hard).sum(), y[~hard].mean()))
o = pd.read_csv("reports/oof_final_final.csv"); p = o.p.values
b = lambda m: np.mean((p[m]-y[m])**2)
tot = np.sum((p-y)**2)
print("ENSEMBLE  in-group Brier %.4f  AUC %.3f | rest Brier %.4f | share of total Brier mass from the hard group: %.0f%%"%(b(hard), roc_auc_score(y[hard],p[hard]), b(~hard), 100*np.sum((p[hard]-y[hard])**2)/tot))
rows=[]
for f in sorted(glob.glob("artifacts/oof/final/clf/*.npz")):
    z=np.load(f); q=np.nanmean(z["oof"],0); n=os.path.basename(f)[:-4]
    rows.append((n, np.mean((q[hard]-y[hard])**2), roc_auc_score(y[hard],q[hard]), np.mean((q[~hard]-y[~hard])**2)))
t=pd.DataFrame(rows,columns=["model","brier_hard","auc_hard","brier_rest"]).sort_values("brier_hard")
print(t.round(4).to_string(index=False))
print("\nhard-group MSE share: %.2fM of the %.2fM total (sentinel-driven part only)"%(1e8*np.sum((p[hard]-y[hard])**2)/len(y)/1e6, 1e8*np.sum((p-y)**2)/len(y)/1e6))
