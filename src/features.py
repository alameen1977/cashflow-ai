"""Feature engineering for late-payment prediction.

Key rule (no data leakage): when we score an invoice, we may only use
information that was known on its invoice date. A customer's payment
history therefore only counts invoices that were already SETTLED before
the new invoice was issued.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/ar.csv"

FEATURES = [
    "amount", "log_amount", "disputed", "paperless",
    "country_391", "country_406", "country_770", "country_818", "country_897",
    "invoice_month", "invoice_dow",
    "hist_n_invoices", "hist_late_rate", "hist_avg_days_late",
    "hist_max_days_late", "hist_dispute_rate", "hist_last_days_late",
    "days_since_last_invoice", "customer_open_invoices",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"invoiceNumber": str, "customerID": str})
    for c in ["InvoiceDate", "DueDate", "SettledDate", "PaperlessDate"]:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y")
    df = df.sort_values(["InvoiceDate", "invoiceNumber"]).reset_index(drop=True)
    df["is_late"] = (df["DaysLate"] > 0).astype(int)
    return df


def _customer_history(group: pd.DataFrame) -> pd.DataFrame:
    """For each invoice, summarise this customer's invoices settled before it."""
    rows = []
    for _, inv in group.iterrows():
        t = inv["InvoiceDate"]
        past = group[group["SettledDate"] < t]          # outcome already known
        prev_issued = group[group["InvoiceDate"] < t]
        open_now = prev_issued[prev_issued["SettledDate"] >= t]
        if len(past):
            last = past.sort_values("SettledDate").iloc[-1]
            rows.append({
                "hist_n_invoices": len(past),
                "hist_late_rate": past["is_late"].mean(),
                "hist_avg_days_late": past["DaysLate"].mean(),
                "hist_max_days_late": past["DaysLate"].max(),
                "hist_dispute_rate": (past["Disputed"] == "Yes").mean(),
                "hist_last_days_late": last["DaysLate"],
            })
        else:  # new customer: no history yet
            rows.append({
                "hist_n_invoices": 0, "hist_late_rate": np.nan,
                "hist_avg_days_late": np.nan, "hist_max_days_late": np.nan,
                "hist_dispute_rate": np.nan, "hist_last_days_late": np.nan,
            })
        rows[-1]["days_since_last_invoice"] = (
            (t - prev_issued["InvoiceDate"].max()).days if len(prev_issued) else np.nan
        )
        rows[-1]["customer_open_invoices"] = len(open_now)
    return pd.DataFrame(rows, index=group.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount"] = out["InvoiceAmount"]
    out["log_amount"] = np.log1p(out["InvoiceAmount"])
    out["disputed"] = (out["Disputed"] == "Yes").astype(int)
    out["paperless"] = (out["PaperlessBill"] == "Electronic").astype(int)
    for code in [391, 406, 770, 818, 897]:
        out[f"country_{code}"] = (out["countryCode"] == code).astype(int)
    out["invoice_month"] = out["InvoiceDate"].dt.month
    out["invoice_dow"] = out["InvoiceDate"].dt.dayofweek

    hist = pd.concat(
        [_customer_history(g) for _, g in out.groupby("customerID")]
    )
    out = out.join(hist)

    # New customers: fill with the portfolio average known at that time
    # (simple, conservative choice; documented in the README).
    global_late = out["is_late"].expanding().mean().shift(1).fillna(out["is_late"].iloc[0])
    out["hist_late_rate"] = out["hist_late_rate"].fillna(global_late)
    for c in ["hist_avg_days_late", "hist_max_days_late",
              "hist_dispute_rate", "hist_last_days_late"]:
        out[c] = out[c].fillna(0)
    out["days_since_last_invoice"] = out["days_since_last_invoice"].fillna(-1)
    return out


if __name__ == "__main__":
    df = build_features(load_raw())
    df.to_csv("data/processed/invoices_features.csv", index=False)
    print(df.shape)
    print(df[FEATURES + ["is_late", "DaysLate"]].describe().T.round(2))
