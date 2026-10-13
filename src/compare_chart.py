"""Chart: model comparison with 95% bootstrap confidence intervals."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
d = pd.read_csv("reports/test_predictions.csv")
y = d["is_late"].values
models = [("rule", "Rule of thumb"), ("logreg", "Logistic regression"),
          ("xgboost", "XGBoost"), ("pytorch", "PyTorch neural net")]
rng = np.random.default_rng(0)
idx = [rng.integers(0, len(d), len(d)) for _ in range(1000)]
rows = []
for key, label in models:
    p = d[f"p_{key}"].values
    boots = [average_precision_score(y[i], p[i]) for i in idx]
    rows.append((label, average_precision_score(y, p), *np.percentile(boots, [2.5, 97.5])))

fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, (label, v, lo, hi) in enumerate(rows):
    c = ORANGE if label == "Rule of thumb" else BLUE
    ax.plot([lo, hi], [i, i], color=c, linewidth=2)
    ax.plot(v, i, "o", color=c, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(hi + 0.01, i, f"{v:.2f}", va="center", color=INK, fontsize=11)
ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK2)
ax.invert_yaxis()
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("PR-AUC on unseen later invoices (higher = better at finding late payers)", color=INK2)
ax.set_title("All three ML models tie — and all beat the rule of thumb", loc="left", fontsize=12, color=INK)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.tick_params(colors=INK2, length=0)
fig.text(0.01, 0.01, "Lines = 95% bootstrap confidence intervals (1,000 resamples)", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("reports/figures/05_model_comparison.png", dpi=150)
json.dump([{"model": r[0], "pr_auc": round(r[1], 3), "ci_low": round(r[2], 3), "ci_high": round(r[3], 3)} for r in rows],
          open("reports/model_ci.json", "w"), indent=2)
print(rows)
