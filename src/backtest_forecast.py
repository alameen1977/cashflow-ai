"""Backtest the weekly cash forecast on every Sunday of the test period.

Metric: WAPE (weighted absolute percentage error) of weekly collections,
for the next 4 weeks from each as-of date. Lower is better.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))
from app import forecast_table  # noqa: E402

dates = pd.date_range("2013-06-09", "2013-10-27", freq="W-SUN")
err_ai, err_rule, total = [], [], []
for d in dates:
    _, rows = forecast_table(d, 4)
    for r in rows:
        err_ai.append(abs(r["forecast_ai"] - r["actual_collected"]))
        err_rule.append(abs(r["forecast_due_date_rule"] - r["actual_collected"]))
        total.append(r["actual_collected"])
res = {"as_of_dates": len(dates), "weeks_scored": len(total),
       "wape_ai": round(sum(err_ai) / sum(total), 3),
       "wape_due_date_rule": round(sum(err_rule) / sum(total), 3)}
res["error_reduction"] = round(1 - res["wape_ai"] / res["wape_due_date_rule"], 3)
json.dump(res, open("reports/forecast_backtest.json", "w"), indent=2)
print(res)
