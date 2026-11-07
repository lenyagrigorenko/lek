"""Концентрация ошибки на нескольких строках; дубликаты ключей и SI > 10 000 во внешнем наборе.

Запуск из корня репозитория: python experiments/analysis/11_review_check_outliers_and_external.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
o=pd.read_csv("reports/oof_final_final.csv"); df=pd.read_pickle("data/processed/all_ext.pkl"); tr=df[df.split=="train"].reset_index(drop=True)
o["title"]=tr.title.values; o["strain"]=tr.strain.str.slice(0,38).values
e2=(o.si-o.pred)**2; tot=e2.sum()
srt=e2.sort_values(ascending=False)
print("share of total OOF squared error: top1 %.1f%%  top5 %.1f%%  top10 %.1f%%"%(100*srt.iloc[0]/tot,100*srt.iloc[:5].sum()/tot,100*srt.iloc[:10].sum()/tot))
print("MSE all %.2fM | without worst row %.2fM | constant-mean baseline all %.2fM, without worst row %.2fM"%(e2.mean()/1e6, e2.drop(srt.index[0]).mean()/1e6, ((o.si-o.si.mean())**2).mean()/1e6, ((o.si.drop(srt.index[0])-o.si.drop(srt.index[0]).mean())**2).mean()/1e6))
print("\ntop-6 rows by squared error:"); print(o.loc[srt.index[:6],["title","strain","si","p","M","pred"]].round(2).to_string())
ext=pd.read_csv("data/external/vacv_external.csv")
d=ext[ext.duplicated(["ik14","document_id","assay_id","cell_line"],keep=False)].sort_values(["ik14","document_id","assay_id"])
print("\nduplicate-key rows:",len(d),"| distinct molecule ids per key (stereoisomers/salts?):",d.groupby(["ik14","document_id","assay_id"]).record_id.nunique().describe()[["min","max"]].to_dict())
print(d[["record_id","ik14","IC50_uM","CC50_uM","SI","document_id"]].head(8).round(3).to_string(index=False))
x=ext[ext.SI>1e4].sort_values("SI",ascending=False)
print("\nSI > 1e4 rows:"); print(x[["record_id","IC50_uM","CC50_uM","SI","doi","cell_line","ic50_type","ic50_relation","cc50_type"]].round(4).to_string(index=False))
print("\nCC50 provenance: cell-line match of CC50 and antiviral assay is NOT stored; missing-IC50 rows:",int(ext.ic50_missing.sum()),"of",len(ext))
