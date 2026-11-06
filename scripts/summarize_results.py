"""Markdown table with the honest numbers of the three feature tiers, read from reports/ensemble_<tag>.json.

    python scripts/summarize_results.py
Every number is cross-fitted: the sentinel blender, the retransformation and the model selection are re-fitted inside CV folds.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = [("final", "все признаки (основной ответ)"), ("assay", "химия + условия ассея"), ("chem", "только химия")]


def main() -> None:
    rows, const_wo = [], float("nan")
    for tag, label in TIERS:
        path = ROOT / "reports" / f"ensemble_{tag}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        p, c, g = r["pipeline"], r["clf"], r["reg"]
        rows.append((label, p.get("oof_mse_crossfit", float("nan")), p.get("mse_without_worst_row", float("nan")),
                     c["nested_brier"][c["chosen"]][0], g["nested_rmse"][g["chosen"]][0], tag))
        const_wo = p.get("const_mse_without_worst_row", float("nan"))
    print("| уровень признаков | честный OOF-MSE | MSE без худшей строки | Brier «потолка» | RMSE log10 SI | посылка |")
    print("|---|---|---|---|---|---|")
    for label, m, mw, b, rm, tag in rows:
        sub = "submissions/submission.csv" if tag == "final" else f"submissions/submission_{tag}.csv"
        print(f"| {label} | {m / 1e6:.2f} M | {mw / 1e6:.2f} M | {b:.4f} | {rm:.3f} | `{sub}` |")
    print(f"| константа (среднее train) | 22.08 M | {const_wo / 1e6:.2f} M | 0.1890 | 1.049 | — |" if rows else "")


if __name__ == "__main__":
    main()
