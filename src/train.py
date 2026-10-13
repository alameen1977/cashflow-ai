"""Step 2: train and honestly compare late-payment models.

Models compared on the SAME time-based split:
  0. Rule of thumb  - "customer paid >50% of past invoices late"
  1. Logistic regression (simple, explainable baseline)
  2. XGBoost (gradient-boosted trees, strong on tabular data)
  3. PyTorch neural network (multi-layer perceptron)

Time-based split: train on older invoices, test on newer ones -
exactly how the model would be used in real life (learn from the past,
predict the future). A random split would leak future information.
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, mean_absolute_error)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import FEATURES

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv("data/processed/invoices_features.csv", parse_dates=["InvoiceDate"])
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# --- time-based split: first 75% of invoice dates train, last 25% test ---
cut = df["InvoiceDate"].quantile(0.75)
train, test = df[df["InvoiceDate"] <= cut], df[df["InvoiceDate"] > cut]
# validation slice (last 15% of train) for early stopping only
vcut = train["InvoiceDate"].quantile(0.85)
tr, va = train[train["InvoiceDate"] <= vcut], train[train["InvoiceDate"] > vcut]

X_tr, y_tr = tr[FEATURES].values, tr["is_late"].values
X_va, y_va = va[FEATURES].values, va["is_late"].values
X_train, y_train = train[FEATURES].values, train["is_late"].values
X_te, y_te = test[FEATURES].values, test["is_late"].values


def capture_at_top(y, p, frac=0.2):
    """Business metric: if collectors only call the riskiest 20% of
    invoices, what share of all late invoices do they reach?"""
    k = int(np.ceil(len(p) * frac))
    top = np.argsort(-p)[:k]
    return y[top].sum() / max(y.sum(), 1)


def evaluate(name, p):
    return {
        "model": name,
        "ROC_AUC": round(roc_auc_score(y_te, p), 3),
        "PR_AUC": round(average_precision_score(y_te, p), 3),
        "Brier": round(brier_score_loss(y_te, np.clip(p, 0, 1)), 3),
        "late_caught_in_top20pct": round(capture_at_top(y_te, p), 3),
    }


results, preds = [], {}

# 0. Rule of thumb
p_rule = (test["hist_late_rate"].values > 0.5).astype(float)
results.append(evaluate("Rule: past late rate > 50%", p_rule)); preds["rule"] = p_rule

# 1. Logistic regression
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
logreg.fit(X_train, y_train)
p_lr = logreg.predict_proba(X_te)[:, 1]
results.append(evaluate("Logistic regression", p_lr)); preds["logreg"] = p_lr

# 2. XGBoost (early stopping on validation slice, then refit on full train)
xgb = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.03,
                    subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
                    eval_metric="logloss", early_stopping_rounds=50,
                    random_state=SEED)
xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_n = xgb.best_iteration + 1
xgb_final = XGBClassifier(n_estimators=best_n, max_depth=3, learning_rate=0.03,
                          subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
                          random_state=SEED)
xgb_final.fit(X_train, y_train)
p_xgb = xgb_final.predict_proba(X_te)[:, 1]
results.append(evaluate("XGBoost", p_xgb)); preds["xgboost"] = p_xgb


# 3. PyTorch MLP
class LatePaymentNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_fit, y_fit, X_val=None, y_val=None, epochs=None):
    scaler = StandardScaler().fit(X_fit)
    Xt = torch.tensor(scaler.transform(X_fit), dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    torch.manual_seed(SEED)
    model = LatePaymentNet(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_ep, best_state, patience = 1e9, 0, None, 0
    max_ep = epochs or 300
    for ep in range(max_ep):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
                vl = loss_fn(model(Xv), torch.tensor(y_val, dtype=torch.float32)).item()
            if vl < best - 1e-4:
                best, best_ep, patience = vl, ep + 1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 25:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_ep


def predict_mlp(model, scaler, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(scaler.transform(X), dtype=torch.float32))).numpy()


_, _, mlp_epochs = train_mlp(X_tr, y_tr, X_va, y_va)
mlp, mlp_scaler, _ = train_mlp(X_train, y_train, epochs=max(mlp_epochs, 10))
p_mlp = predict_mlp(mlp, mlp_scaler, X_te)
results.append(evaluate("PyTorch neural network", p_mlp)); preds["pytorch"] = p_mlp

# --- pick the winner by PR-AUC (best for finding the late minority) ---
res = pd.DataFrame(results)
print(f"Train: {len(train)} invoices ({train.InvoiceDate.min().date()} to {train.InvoiceDate.max().date()})")
print(f"Test:  {len(test)} invoices ({test.InvoiceDate.min().date()} to {test.InvoiceDate.max().date()})")
print(f"Test late rate: {y_te.mean():.1%}\n")
print(res.to_string(index=False))
ml = res[res["model"] != "Rule: past late rate > 50%"]
winner = ml.sort_values(["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]["model"]
print(f"\nWinner: {winner}")

# --- days-late model (how late, given it is late) for the cash forecast ---
late_tr = train[train["is_late"] == 1]
reg = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                   subsample=0.9, random_state=SEED)
reg.fit(late_tr[FEATURES].values, late_tr["DaysLate"].values)
late_te = test[test["is_late"] == 1]
mae = mean_absolute_error(late_te["DaysLate"], reg.predict(late_te[FEATURES].values))
naive_mae = mean_absolute_error(late_te["DaysLate"],
                                np.full(len(late_te), late_tr["DaysLate"].median()))
print(f"Days-late model MAE: {mae:.1f} days (naive median guess: {naive_mae:.1f} days)")

# --- days-to-settle model (when will the cash actually arrive?) ---
settle = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                      subsample=0.9, random_state=SEED)
settle.fit(X_train, train["DaysToSettle"].values)
settle_mae = mean_absolute_error(test["DaysToSettle"], settle.predict(X_te))
settle_naive = mean_absolute_error(test["DaysToSettle"], np.full(len(test), 30))
print(f"Days-to-pay model MAE: {settle_mae:.1f} days (assume 'pays on due date': {settle_naive:.1f} days)")

# feature importance (XGBoost gain) for explainability
imp = pd.Series(xgb_final.get_booster().get_score(importance_type="gain"))
imp.index = [FEATURES[int(k[1:])] for k in imp.index]
imp = (imp / imp.sum()).sort_values(ascending=False).round(3)
print("\nTop drivers (XGBoost):\n", imp.head(8).to_string())

# --- save artefacts ---
import os
os.makedirs("models", exist_ok=True)
pickle.dump({"logreg": logreg, "xgboost": xgb_final, "days_late": reg, "days_to_settle": settle,
             "features": FEATURES}, open("models/models.pkl", "wb"))
torch.save({"state_dict": mlp.state_dict(), "scaler": mlp_scaler,
            "n_in": len(FEATURES)}, "models/pytorch_mlp.pt")
out = test[["invoiceNumber", "customerID", "InvoiceDate", "InvoiceAmount", "is_late", "DaysLate"]].copy()
for k, v in preds.items():
    out[f"p_{k}"] = v
out.to_csv("reports/test_predictions.csv", index=False)
json.dump({"results": results, "winner": winner,
           "days_late_mae": round(mae, 2), "days_late_naive_mae": round(naive_mae, 2),
           "days_to_pay_mae": round(settle_mae, 2), "days_to_pay_naive_mae": round(settle_naive, 2),
           "top_drivers": imp.head(8).to_dict(),
           "train_size": len(train), "test_size": len(test),
           "test_late_rate": round(float(y_te.mean()), 3)},
          open("reports/model_results.json", "w"), indent=2)
