"""Шум публичного лидерборда: MSE на случайных подвыборках по ~179 строк из OOF-ошибок.

Запуск из корня репозитория: python experiments/analysis/12_leaderboard_noise.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import numpy as np, pandas as pd
o = pd.read_csv("reports/oof_final_final.csv")
e2 = (o["si"] - o["pred"]).to_numpy() ** 2
n = 179                                        # ~30% of the 597 test rows
rng = np.random.default_rng(0)
mse = np.array([e2[rng.choice(len(e2), n, replace=False)].mean() for _ in range(20000)])
print("OOF MSE on all 320 rows: %.2fM" % (e2.mean() / 1e6))
print("MSE on random %d-row subsets:  5%%=%.2fM  25%%=%.2fM  50%%=%.2fM  75%%=%.2fM  95%%=%.2fM" % ((n,) + tuple(np.quantile(mse, [.05, .25, .5, .75, .95]) / 1e6)))
top = np.argsort(-e2)[:3]
print("largest single squared errors (M of MSE if the row is in a %d-row subset): %s" % (n, [round(e2[i] / n / 1e6, 2) for i in top]))
print("rows with error^2/n > 1M:", int((e2 / n > 1e6).sum()), "of", len(e2))
print("MSE without the single worst row: %.2fM" % (np.delete(e2, top[0]).mean() / 1e6))
