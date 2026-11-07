"""Симуляция: помогает ли второй слот с другой «хвостовой ставкой» при зачёте лучшей из двух посылок.

Запуск из корня репозитория: python experiments/analysis/20_best_of_two_simulation.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
o = pd.read_csv("reports/oof_final_final.csv"); si, p, M = o.si.values, o.p.values, o.M.values
rng = np.random.default_rng(0)
A = p*1e4 + (1-p)*M
def sim(pred_b, n, reps=6000):
    ea, eb, mn, wins = [], [], [], 0
    ea_all = (si-A)**2; eb_all = (si-pred_b)**2
    for _ in range(reps):
        i = rng.integers(0, len(si), n); a = ea_all[i].mean(); b = eb_all[i].mean()
        ea.append(a); eb.append(b); mn.append(min(a,b)); wins += b < a
    return np.mean(ea), np.mean(eb), np.mean(mn), wins/reps
n = 418   # private 70% of the 597 test rows
print(f"bootstrap of {n}-row 'private' sets from the 320 OOF rows (constant-mean baseline ~22M)")
print("variant                       E[MSE_A]   E[MSE_B]   E[min(A,B)]   P(B<A)")
for thr in (300, 500, 1000):
    for f in (1.5, 2.0, 3.0, 5.0):
        Mb = np.where(M >= thr, M*f, M); B = p*1e4 + (1-p)*Mb
        a, b, m, w = sim(B, n); print(f"tail x{f:<3} for M>={thr:<5}     {a/1e6:8.3f}M {b/1e6:8.3f}M   {m/1e6:8.3f}M   {w:.2f}")
# a second, different bet: sharper / softer sentinel probabilities
for name, q in [("p sharpened (logit x1.3)", 1/(1+np.exp(-1.3*np.log(np.clip(p,1e-3,1-1e-3)/(1-np.clip(p,1e-3,1-1e-3)))))), ("p shrunk (x0.8 + 0.2*mean)", 0.8*p+0.2*p.mean())]:
    B = q*1e4 + (1-q)*M; a, b, m, w = sim(B, n); print(f"{name:28s}{a/1e6:8.3f}M {b/1e6:8.3f}M   {m/1e6:8.3f}M   {w:.2f}")
