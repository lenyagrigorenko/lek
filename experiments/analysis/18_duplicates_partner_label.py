"""Насколько совпадают метки у дубликатов в train и почему подстановка метки партнёра опасна.

Запуск из корня репозитория: python experiments/analysis/18_duplicates_partner_label.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
pd.set_option("display.width",230); pd.set_option("display.max_colwidth",44)
df = pd.read_pickle("data/processed/all_ext.pkl"); k = df[df.split!="ext"].reset_index(drop=True)
tr = k[k.split=="train"].reset_index(drop=True)
o = pd.read_csv("reports/oof_final_final.csv"); tr["p"] = o.p.values; tr["M"] = o.M.values; tr["pred"] = o.pred.values
rows = []
for ik, d in tr.groupby("ik14"):
    if len(d) < 2: continue
    d = d.reset_index(drop=True)
    for i in range(len(d)):
        for j in range(len(d)):
            if i == j: continue
            a, b = d.iloc[i], d.iloc[j]          # predict row a from its partner b
            rows.append(dict(title=a.title, partner=b.title, same_smiles=a.smiles_std == b.smiles_std, same_strain=a.strain.strip()==b.strain.strip(),
                             sent_a=int(a.is_sentinel), sent_partner=int(b.is_sentinel), si_a=a.si, si_partner=b.si, p_oof=round(a.p,2), pred_oof=round(a.pred), err_model=(a.si-a.pred)**2, err_partner=(a.si-b.si)**2))
R = pd.DataFrame(rows); print(R.drop(columns=["err_model","err_partner"]).to_string(index=False))
print("\nrows predicted from their partner's label: n=%d | sentinel flag agrees: %d | mean squared error  model(OOF): %.3gM   partner-label: %.3gM"%(len(R), int((R.sent_a==R.sent_partner).sum()), R.err_model.mean()/1e6, R.err_partner.mean()/1e6))
