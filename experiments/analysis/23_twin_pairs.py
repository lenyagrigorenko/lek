"""Q: how well does the label of a molecule's twin (same InChIKey connectivity block: a duplicate or a stereoisomer) predict its own label?

Gives the constants of `src/vacv_si/ensemble/twins.py`. Needs `python -m vacv_si prepare -c configs/exp/final.yaml` first (data/processed/all_ext.pkl).
Integrity: the statistics use external records of molecules that are NOT in the competition (train or test), so no evaluation molecule is involved.
"""
import numpy as np
import pandas as pd

df = pd.read_pickle("data/processed/all_ext.pkl")
comp = df[df.split.isin(["train", "test"])]
ext = df[(df.split == "ext") & df.si.notna()].copy()
ext = ext[~ext.ik14.isin(set(comp.ik14))]
print(f"external records of non-competition molecules: {len(ext)} ({ext.ik14.nunique()} connectivity keys)")

pairs = []
for (ik, doi), g in ext.groupby(["ik14", "doi"]):
    if len(g) >= 2 and g.inchikey.nunique() >= 2:                  # same paper, same connectivity, different stereo
        v = g.sort_values("inchikey")
        pairs.append((v.iloc[0].si, v.iloc[1].si))
p = pd.DataFrame(pairs, columns=["a", "b"])
za, zb = p.a == 0, p.b == 0
print(f"stereoisomer pairs of the same paper: {len(p)} | both active {int((~za & ~zb).sum())} | one inactive {int((za ^ zb).sum())} | both inactive {int((za & zb).sum())}")
act = p[(p.a > 0) & (p.b > 0)]
la, lb = np.log10(act.a), np.log10(act.b)
print(f"both active: corr(log10 SI) = {np.corrcoef(la, lb)[0, 1]:.2f}, median |log10 ratio| = {np.median(np.abs(la - lb)):.2f}, share within x3 = {np.mean(np.abs(la - lb) < 0.48):.2f}")
x, y = np.r_[la, lb], np.r_[lb, la]
slope, icpt = np.polyfit(x, y, 1)
print(f"partner log10 SI = {icpt:.2f} + {slope:.2f} * own log10 SI   (residual sd {np.std(y - icpt - slope * x):.2f}, n = {len(x)})   -> twins.py: a, b")
print(f"P(partner inactive | one member active) = {int((za ^ zb).sum()) / max(1, int((za ^ zb).sum() + (~za & ~zb).sum())):.2f} -> q_active ~ 0.06;"
      f" P(partner inactive | one member inactive) = {int((za & zb).sum()) / max(1, int((za & zb).sum() + (za ^ zb).sum())):.3f} -> q_zero ~ 0.875")

tr = df[df.split == "train"].reset_index(drop=True)
rows = []
for ik, g in tr.groupby("ik14"):
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            rows.append((g.iloc[i].title, g.iloc[j].title, g.iloc[i].si, g.iloc[j].si))
d = pd.DataFrame(rows, columns=["t1", "t2", "si1", "si2"])
agree = ((d.si1 == 0) & (d.si2 == 0)) | ((d.si1 > 0) & (d.si2 > 0) & (np.abs(np.log10(d.si1.clip(lower=1e-2)) - np.log10(d.si2.clip(lower=1e-2))) < 0.3))
print(f"\ncompetition train, pairs of the same molecule: {len(d)}; labels agree (both inactive or within x2): {int(agree.sum())}")
te = df[df.split == "test"]
print(f"test rows whose molecule also occurs in train: {int(te.ik14.isin(set(tr.ik14)).sum())} of {len(te)}")
