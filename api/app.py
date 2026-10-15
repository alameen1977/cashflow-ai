"""CashFlow AI - prediction API (FastAPI).

Champion model : logistic regression (tied best accuracy, fully explainable)
Challenger     : PyTorch neural network (scored side-by-side for monitoring)

Endpoints
  GET  /health                       service check
  POST /score                        score one new/open invoice
  GET  /portfolio/open-invoices      all invoices open on a given date,
                                     ranked by cash at risk, with the
                                     recommended collections action
  GET  /portfolio/cash-forecast      expected cash inflow by week
"""
import os
import pickle
import sys
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import FEATURES, load_raw  # noqa: E402

# ---------- load models ----------
M = pickle.load(open(os.path.join(ROOT, "models/models.pkl"), "rb"))
LOGREG, DAYS_LATE, SETTLE = M["logreg"], M["days_late"], M["days_to_settle"]


# PyTorch challenger: trained in PyTorch (src/train.py), weights exported to
# numpy so the API runs without the 700MB+ torch library (fits free hosting).
_W = np.load(os.path.join(ROOT, "models/pytorch_mlp_weights.npz"))


def mlp_predict(X: np.ndarray) -> np.ndarray:
    """Same forward pass as LatePaymentNet (dropout is off at inference)."""
    h = (X - _W["scaler_mean"]) / _W["scaler_scale"]
    h = np.maximum(h @ _W["net_0_weight"].T + _W["net_0_bias"], 0)
    h = np.maximum(h @ _W["net_3_weight"].T + _W["net_3_bias"], 0)
    z = (h @ _W["net_6_weight"].T + _W["net_6_bias"]).ravel()
    return 1 / (1 + np.exp(-z))


# ---------- the "company ledger" (invoice history) ----------
LEDGER = load_raw(os.path.join(ROOT, "data/raw/ar.csv"))
FEAT = pd.read_csv(os.path.join(ROOT, "data/processed/invoices_features.csv"),
                   dtype={"invoiceNumber": str, "customerID": str},
                   parse_dates=["InvoiceDate", "DueDate", "SettledDate"])
HIGH_VALUE = float(LEDGER["InvoiceAmount"].quantile(0.90))  # approval threshold

REASONS = {
    "hist_late_rate": "often paid late in the past",
    "hist_avg_days_late": "past late payments were long",
    "hist_max_days_late": "has had a very late payment before",
    "hist_last_days_late": "last invoice was paid late",
    "disputed": "invoice is under dispute",
    "hist_dispute_rate": "frequently disputes invoices",
    "paperless": "receives paper (not electronic) bills",
    "customer_open_invoices": "has several unpaid invoices open",
}


def _probs(X: np.ndarray):
    p_champ = LOGREG.predict_proba(X)[:, 1]
    return p_champ, mlp_predict(X)


def _reasons(x: np.ndarray, top=3):
    """Explain a prediction: which inputs pushed risk UP the most."""
    scaler, lr = LOGREG.named_steps["standardscaler"], LOGREG.named_steps["logisticregression"]
    contrib = lr.coef_[0] * (x - scaler.mean_) / scaler.scale_
    out = []
    for i in np.argsort(-contrib):
        f = FEATURES[i]
        if contrib[i] <= 0.15 or len(out) >= top:
            break
        if f == "paperless":
            if x[i] == 0:
                out.append(REASONS[f])
        elif f in REASONS:
            out.append(REASONS[f])
    return out or ["no strong risk signals"]


def _action(p: float, amount: float, days_to_due: int):
    if p >= 0.7:
        tier, act = "HIGH", "Firm reminder now + phone call from account manager"
    elif p >= 0.4:
        tier, act = "MEDIUM", "Friendly reminder before due date"
    else:
        tier, act = "LOW", "Standard reminder on due date"
    if days_to_due < 0:
        act = "OVERDUE - " + act
    needs_approval = tier == "HIGH" and amount >= HIGH_VALUE
    return tier, act, needs_approval


app = FastAPI(title="CashFlow AI", version="1.0",
              description="Predicts late invoice payments and prioritises collections.")


DASHBOARD = os.path.join(ROOT, "api", "static", "index.html")


@app.get("/", include_in_schema=False)
def dashboard():
    """The CashFlow AI dashboard (a single page that calls this API)."""
    return FileResponse(DASHBOARD)


@app.get("/customers")
def customers():
    """Customer list with headline payment history (for the dashboard form)."""
    g = LEDGER.groupby("customerID").agg(invoices=("is_late", "size"),
                                         late_rate=("is_late", "mean"),
                                         country=("countryCode", "first"))
    g = g.sort_values("late_rate", ascending=False)
    return [{"customer_id": c, "invoices": int(r.invoices),
             "late_rate": round(float(r.late_rate), 3), "country_code": int(r.country)}
            for c, r in g.iterrows()]


@app.get("/health")
def health():
    return {"status": "ok", "champion": "logistic_regression", "challenger": "pytorch_mlp",
            "ledger_invoices": len(LEDGER)}


class InvoiceIn(BaseModel):
    customer_id: str = Field(examples=["9149-MATVB"])
    invoice_amount: float = Field(gt=0, examples=[85.5])
    invoice_date: date = Field(examples=["2013-09-01"])
    disputed: bool = False
    paperless: bool = True
    country_code: int = Field(391, examples=[391])


@app.post("/score")
def score(inv: InvoiceIn):
    t = pd.Timestamp(inv.invoice_date)
    hist = LEDGER[LEDGER["customerID"] == inv.customer_id]
    past = hist[hist["SettledDate"] < t]
    prev = hist[hist["InvoiceDate"] < t]
    row = {
        "amount": inv.invoice_amount, "log_amount": np.log1p(inv.invoice_amount),
        "disputed": int(inv.disputed), "paperless": int(inv.paperless),
        **{f"country_{c}": int(inv.country_code == c) for c in [391, 406, 770, 818, 897]},
        "invoice_month": t.month, "invoice_dow": t.dayofweek,
        "hist_n_invoices": len(past),
        "hist_late_rate": past["is_late"].mean() if len(past) else float(LEDGER["is_late"].mean()),
        "hist_avg_days_late": past["DaysLate"].mean() if len(past) else 0,
        "hist_max_days_late": past["DaysLate"].max() if len(past) else 0,
        "hist_dispute_rate": (past["Disputed"] == "Yes").mean() if len(past) else 0,
        "hist_last_days_late": past.sort_values("SettledDate")["DaysLate"].iloc[-1] if len(past) else 0,
        "days_since_last_invoice": (t - prev["InvoiceDate"].max()).days if len(prev) else -1,
        "customer_open_invoices": int((prev["SettledDate"] >= t).sum()),
    }
    x = np.array([[row[f] for f in FEATURES]], dtype=float)
    p, p_chal = _probs(x)
    exp_days = float(max(DAYS_LATE.predict(x)[0], 1)) if p[0] >= 0.5 else 0.0
    tier, act, appr = _action(p[0], inv.invoice_amount, 30)
    return {
        "customer_id": inv.customer_id, "known_customer": len(hist) > 0,
        "past_invoices_used": len(past),
        "probability_late": round(float(p[0]), 3),
        "challenger_probability_late": round(float(p_chal[0]), 3),
        "expected_days_late_if_late": round(float(max(DAYS_LATE.predict(x)[0], 1)), 1),
        "risk_tier": tier, "recommended_action": act, "needs_human_approval": appr,
        "reasons": _reasons(x[0]),
    }


@app.get("/portfolio/open-invoices")
def open_invoices(as_of: date = Query(date(2013, 9, 15), description="'Today' in the simulation"),
                  min_tier: Optional[str] = Query(None, description="LOW / MEDIUM / HIGH"),
                  limit: int = Query(50, le=500)):
    t = pd.Timestamp(as_of)
    op = FEAT[(FEAT["InvoiceDate"] <= t) & (FEAT["SettledDate"] > t)].copy()
    if op.empty:
        raise HTTPException(404, "No open invoices on that date (data covers 2012-01 to 2013-12)")
    X = op[FEATURES].values.astype(float)
    op["p_late"], op["p_challenger"] = _probs(X)
    op["exp_days_late"] = np.maximum(DAYS_LATE.predict(X), 1)
    op["cash_at_risk"] = op["p_late"] * op["InvoiceAmount"]
    op = op.sort_values("cash_at_risk", ascending=False)
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    out = []
    for i, r in enumerate(op.itertuples(index=False)):
        days_to_due = (r.DueDate - t).days
        tier, act, appr = _action(r.p_late, r.InvoiceAmount, days_to_due)
        if min_tier and rank[tier] < rank[min_tier.upper()]:
            continue
        out.append({
            "priority": len(out) + 1, "invoice_number": r.invoiceNumber,
            "customer_id": r.customerID,
            "customer_email": f"ap+{r.customerID.lower()}@example.com",  # demo contact
            "amount": round(r.InvoiceAmount, 2),
            "invoice_date": str(r.InvoiceDate.date()), "due_date": str(r.DueDate.date()),
            "days_to_due": days_to_due, "disputed": bool(r.disputed),
            "probability_late": round(float(r.p_late), 3),
            "challenger_probability_late": round(float(r.p_challenger), 3),
            "expected_days_late_if_late": round(float(r.exp_days_late), 1),
            "cash_at_risk": round(float(r.cash_at_risk), 2),
            "risk_tier": tier, "recommended_action": act, "needs_human_approval": appr,
            "reasons": _reasons(np.array([getattr(r, f) for f in FEATURES], dtype=float)),
        })
        if len(out) >= limit:
            break
    return {"as_of": str(as_of), "open_invoices_total": int(len(op)),
            "total_open_amount": round(float(op["InvoiceAmount"].sum()), 2),
            "total_cash_at_risk": round(float(op["cash_at_risk"].sum()), 2),
            "high_value_threshold": round(HIGH_VALUE, 2),
            "invoices": out}


def forecast_table(as_of: pd.Timestamp, weeks: int):
    """Expected payment date = invoice date + predicted days-to-pay.
    Compared with the common shortcut 'everyone pays on the due date'."""
    t = as_of
    op = FEAT[(FEAT["InvoiceDate"] <= t) & (FEAT["SettledDate"] > t)].copy()
    X = op[FEATURES].values.astype(float)
    pred_days = np.clip(SETTLE.predict(X), 1, None)
    ai_date = op["InvoiceDate"] + pd.to_timedelta(np.round(pred_days), unit="D")
    floor = t + pd.Timedelta(days=1)
    ai_date = ai_date.where(ai_date > t, floor)
    naive_date = op["DueDate"].where(op["DueDate"] > t, floor)
    rows = []
    for w in range(weeks):
        a, b = t + timedelta(weeks=w), t + timedelta(weeks=w + 1)
        rows.append({
            "week_start": str(a.date()),
            "forecast_ai": round(float(op.loc[(ai_date > a) & (ai_date <= b), "InvoiceAmount"].sum()), 2),
            "forecast_due_date_rule": round(float(op.loc[(naive_date > a) & (naive_date <= b), "InvoiceAmount"].sum()), 2),
            "actual_collected": round(float(op.loc[(op["SettledDate"] > a) & (op["SettledDate"] <= b), "InvoiceAmount"].sum()), 2),
        })
    return op, rows


@app.get("/portfolio/cash-forecast")
def cash_forecast(as_of: date = Query(date(2013, 9, 15)), weeks: int = Query(6, le=12)):
    op, rows = forecast_table(pd.Timestamp(as_of), weeks)
    return {"as_of": str(as_of), "open_amount": round(float(op["InvoiceAmount"].sum()), 2),
            "note": "actual_collected is shown because this is historical data (backtest)",
            "weeks": rows}
