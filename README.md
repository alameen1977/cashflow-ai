# CashFlow AI — Collections Intelligence System

Predicts which customer invoices will be paid late, automates prioritised
follow-ups (with human approval for high-value accounts), and forecasts
incoming cash.

**Stack:** Python · pandas · PyTorch · XGBoost · FastAPI · Docker · n8n · LLM API

> Status: 🚧 Step 1 of 4 complete (data & analysis)

## The business problem

Sales are made, but cash sits in receivables. Collections teams chase every
customer the same way, learn too late who won't pay, and can't tell the CFO
how much cash is coming next month. For a company with AED 50M annual
revenue, each extra day of DSO ties up roughly AED 137k of cash.

## Data

[IBM Accounts Receivable sample dataset](https://github.com/pthiagu2/DataMining/blob/master/WA_Fn-UseC_-Accounts-Receivable.csv)
— 2,466 invoices, 100 customers, 5 countries, Jan 2012 – Dec 2013,
30-day payment terms.

## Step 1 findings (exploratory analysis)

| Finding | Number |
|---|---|
| Invoices paid late | 35.6% |
| Late rate when the invoice is **disputed** vs not | 68% vs 26% |
| Late rate for **paper** vs electronic bills | 43% vs 28% |
| Customers with a bad track record (70–100% past late rate) who pay the next invoice late | 79% (vs 6% for reliable customers) |
| Median delay of a late invoice | 8 days (90th percentile: 20 days) |
| Share of all late-days caused by the 10 worst customers | 37% |

![Past behaviour predicts future behaviour](reports/figures/02_history_effect.png)
![Drivers](reports/figures/01_drivers.png)

**Business takeaways**
1. A customer's own payment history is the strongest early-warning signal —
   collections should be risk-based, not one-size-fits-all.
2. Resolving disputes fast is a collections lever: disputed invoices are
   2.6× more likely to be late.
3. Moving customers to e-invoicing is associated with fewer late payments
   (correlation, not proven cause).

## Design decisions (and why)

- **No data leakage.** A customer's history features only use invoices that
  were already *settled* before the new invoice was issued — exactly what a
  real collections team would know on that day.
- **New customers** with no history get the portfolio-average late rate
  known at that time.
- **Dispute flag:** in practice a dispute is raised after the invoice is
  sent, so the system re-scores open invoices daily as disputes come in.
- **Known data limitation:** the dataset only contains invoices that were
  eventually settled. Invoices issued near the end of 2013 are therefore
  biased towards fast payers (visible as falling days-to-settle in late 2013).
  Model evaluation will account for this.

## Project structure

```
data/raw/            original dataset
data/processed/      engineered features
src/features.py      leakage-safe feature engineering
src/eda.py           analysis + charts
reports/figures/     charts
api/                 FastAPI prediction service   (step 2)
n8n/                 collections workflow export  (step 3)
docs/                architecture, demo           (step 4)
```

## Run it

```bash
pip install -r requirements.txt
python src/features.py
python src/eda.py
```

## Roadmap

- [x] Step 1 — Data, leakage-safe features, exploratory analysis
- [ ] Step 2 — Baseline (logistic regression), XGBoost and PyTorch models; honest comparison; FastAPI service
- [ ] Step 3 — n8n collections autopilot with human-approval step
- [ ] Step 4 — Cash forecast, dashboard, demo video
