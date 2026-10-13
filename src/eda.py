"""Exploratory analysis: the business story behind late payments.

Writes charts to reports/figures and key numbers to reports/eda_summary.json.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

df = pd.read_csv("data/processed/invoices_features.csv",
                 parse_dates=["InvoiceDate", "DueDate", "SettledDate"])
summary = {}


def bar_rates(ax, labels, rates, counts, color, title):
    bars = ax.bar(labels, rates, color=color, width=0.55)
    for b, r, n in zip(bars, rates, counts):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.015, f"{r:.0%}",
                ha="center", color=INK, fontsize=11, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.02, f"n={n}",
                ha="center", color="white", fontsize=9)
    ax.set_ylim(0, max(rates) * 1.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.grid(axis="x", visible=False)


# 1. Drivers: dispute + billing channel
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
g = df.groupby("Disputed")["is_late"].agg(["mean", "count"]).loc[["No", "Yes"]]
bar_rates(axes[0], ["Not disputed", "Disputed"], g["mean"], g["count"], BLUE,
          "Share paid late, by dispute")
g2 = df.groupby("PaperlessBill")["is_late"].agg(["mean", "count"]).loc[["Electronic", "Paper"]]
bar_rates(axes[1], ["Electronic bill", "Paper bill"], g2["mean"], g2["count"], BLUE,
          "Share paid late, by billing channel")
fig.tight_layout(); fig.savefig("reports/figures/01_drivers.png", dpi=150); plt.close(fig)
summary["late_rate_overall"] = round(df["is_late"].mean(), 3)
summary["late_rate_disputed"] = round(g.loc["Yes", "mean"], 3)
summary["late_rate_not_disputed"] = round(g.loc["No", "mean"], 3)
summary["late_rate_paper"] = round(g2.loc["Paper", "mean"], 3)
summary["late_rate_electronic"] = round(g2.loc["Electronic", "mean"], 3)

# 2. Past behaviour predicts future behaviour (the core idea of the model)
bins = [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
labels = ["0–10%", "10–30%", "30–50%", "50–70%", "70–100%"]
df["hist_bucket"] = pd.cut(df["hist_late_rate"], bins, labels=labels)
h = df[df["hist_n_invoices"] >= 3].groupby("hist_bucket", observed=True)["is_late"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(8, 4.2))
bar_rates(ax, h.index.astype(str), h["mean"], h["count"], ORANGE,
          "Customers who paid late before keep paying late")
ax.set_xlabel("Customer's past late-payment rate (known at invoice date, ≥3 past invoices)")
ax.set_ylabel("Share of next invoices paid late")
fig.tight_layout(); fig.savefig("reports/figures/02_history_effect.png", dpi=150); plt.close(fig)
summary["late_rate_by_history_bucket"] = {k: round(v, 3) for k, v in h["mean"].items()}

# 3. How late are late invoices?
late = df.loc[df["is_late"] == 1, "DaysLate"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(late, bins=range(1, 47, 2), color=BLUE, edgecolor=SURFACE, linewidth=2)
ax.axvline(late.median(), color=INK2, linestyle="--", linewidth=1.5)
ax.text(late.median() + 0.8, ax.get_ylim()[1] * 0.9,
        f"median {late.median():.0f} days late", color=INK2)
ax.set_title("How late are the late invoices?", loc="left", fontsize=12)
ax.set_xlabel("Days paid after due date"); ax.set_ylabel("Invoices")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("reports/figures/03_days_late.png", dpi=150); plt.close(fig)
summary["late_days_median"] = float(late.median())
summary["late_days_p90"] = float(late.quantile(0.9))

# 4. DSO-style view over time: average days to settle per month
m = df.set_index("InvoiceDate").resample("MS").agg(
    {"DaysToSettle": "mean", "is_late": "mean", "InvoiceAmount": "sum"})
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(m.index, m["DaysToSettle"], color=BLUE, linewidth=2, marker="o", markersize=5)
ax.axhline(30, color=INK2, linestyle=":", linewidth=1.2)
ax.text(m.index[-1], 30.25, "payment terms: 30 days", color=INK2, fontsize=9, ha="right")
ax.set_title("Average days to collect cash, by invoice month", loc="left", fontsize=12)
ax.set_ylabel("Days to settle")
fig.tight_layout(); fig.savefig("reports/figures/04_days_to_settle_trend.png", dpi=150); plt.close(fig)
summary["avg_days_to_settle"] = round(df["DaysToSettle"].mean(), 1)

# 5. Concentration: a few customers drive most lateness
c = df.groupby("customerID").agg(late_days=("DaysLate", "sum")).sort_values("late_days", ascending=False)
share_top10 = c["late_days"].head(10).sum() / c["late_days"].sum()
summary["share_of_late_days_top10_customers"] = round(share_top10, 3)
summary["n_invoices"] = int(len(df)); summary["n_customers"] = int(df["customerID"].nunique())
summary["date_range"] = [str(df["InvoiceDate"].min().date()), str(df["InvoiceDate"].max().date())]

json.dump(summary, open("reports/eda_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
