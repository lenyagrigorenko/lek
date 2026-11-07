"""Хвост не-потолочного SI, разложение дисперсии для MSE и проверка типа сплита (сходство test→train против train→train).

Запуск из корня репозитория: python experiments/analysis/05_tail_and_split_type.py
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
RDLogger.DisableLog("rdApp.*")
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
D="data/raw"
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
tr["sent"]=(tr["SI VACV"]==10000).astype(int)
y=tr["SI VACV"]; ns=tr[tr.sent==0]
print("=== non-sentinel SI (n=%d) ==="%len(ns))
print(ns["SI VACV"].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(3).to_string())
print("mean raw:", ns["SI VACV"].mean().round(1), " var raw:", ns["SI VACV"].var().round(0), " mean log10:", np.log10(ns["SI VACV"].clip(1e-3)).mean().round(3))
bins=[-1,0.01,0.1,0.5,0.99,1.01,2,5,10,30,100,300,1000,3000,10000,1e6]
print(pd.cut(ns["SI VACV"],bins).value_counts().sort_index().to_string())
print("\n--- top 14 non-sentinel by SI:")
print(ns.sort_values("SI VACV",ascending=False)[["Title","Culture","CC50_uM","IC50_uM_VACV","SI VACV","strain"]].head(14).to_string(index=False))
print("\n--- exact-zero/near-zero SI:"); print(tr[tr["SI VACV"]<0.01][["Title","CC50_uM","IC50_uM_VACV","SI VACV"]].to_string(index=False))

# variance decomposition for MSE
p=tr.sent.mean(); m_ns=ns["SI VACV"].mean(); v_ns=ns["SI VACV"].var()
tot=y.var(ddof=0)
print(f"\nTOTAL var(SI)={tot:,.0f}; sentinel-Bernoulli part ~ {p*(1-p)*(10000-m_ns)**2:,.0f}; within non-sentinel part ~ {(1-p)*v_ns:,.0f}")

# ---------- split-type check: nearest-neighbour similarity ----------
gen=rfg.GetMorganGenerator(radius=2,fpSize=2048)
def fps(smis): 
    out=[]
    for s in smis:
        m=Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out
ftr=fps(tr.SMILES); fte=fps(te.SMILES)
print("\nbad smiles train:", sum(f is None for f in ftr), " test:", sum(f is None for f in fte))
ftr_ok=[f for f in ftr if f is not None]
def maxsim_to(fl, ref, skip_self=False):
    r=[]
    for i,f in enumerate(fl):
        s=np.array(DataStructs.BulkTanimotoSimilarity(f,ref))
        if skip_self: s[i]=-1
        r.append(s.max())
    return np.array(r)
loo=maxsim_to(ftr_ok, ftr_ok, True)
t2t=maxsim_to([f for f in fte if f is not None], ftr_ok)
# test->test
fte_ok=[f for f in fte if f is not None]
t2t_self=maxsim_to(fte_ok, fte_ok, True)
q=[.05,.1,.25,.5,.75,.9]
print("\nmax Tanimoto(Morgan2) quantiles", q)
print("train->train (LOO)  ", np.quantile(loo,q).round(3), " frac>=0.99: %.3f"%(loo>=.99).mean(), " frac>=0.7: %.3f"%(loo>=.7).mean())
print("test ->train        ", np.quantile(t2t,q).round(3), " frac>=0.99: %.3f"%(t2t>=.99).mean(), " frac>=0.7: %.3f"%(t2t>=.7).mean())
print("test ->test (LOO)   ", np.quantile(t2t_self,q).round(3), " frac>=0.7: %.3f"%(t2t_self>=.7).mean())
# if split is random from one pool: test->(train+test) ~ train->(train+test)
# compare: train->train LOO with train size 320 vs test->train; for random split expect test->train ~ train->train(LOO)
