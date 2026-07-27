"""
backtest_check.py
==================
Answers the question: "is the model actually any good?"

How it works:
  Pretends the last 3 known months never happened, forecasts them using
  only the data before that, then compares the forecast to what actually
  happened. This is the standard way to sanity-check a forecasting model
  — testing on data it was NOT trained on, not just checking it runs.

Run:
    python backtest_check.py

Reading the output:
  MAPE (Mean Absolute Percentage Error) = average % the forecast was off
  by. Rough guide for this kind of order-booking data:
    < 15%   strong
    15-30%  reasonable, usable for planning
    30-50%  directional only, use with caution
    > 50%   not reliable yet — needs more/better data (this is expected
            for "Others", which has very little history)
"""

import numpy as np
import pandas as pd
from src.data_source import FileDataSource
from src.forecast_model import build_monthly_series, _forecast_one_series

DATA_PATH = "data/sales_orders_v2.xlsx"
HOLDOUT_MONTHS = 3


def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def main():
    df = FileDataSource(DATA_PATH).get_sales_data()
    monthly = build_monthly_series(df, value_col="so_qty")

    print(f"Backtest: holding out the last {HOLDOUT_MONTHS} months per division\n")
    print(f"{'Division':<15}{'Method':<38}{'MAPE':<10}{'Verdict'}")
    print("-" * 85)

    for division, grp in monthly.groupby("division"):
        grp = grp.sort_values("month")
        series = grp.set_index("month")["so_qty"]

        if len(series) <= HOLDOUT_MONTHS + 4:
            print(f"{division:<15}{'(not enough history to backtest)':<38}")
            continue

        train = series.iloc[:-HOLDOUT_MONTHS]
        actual_holdout = series.iloc[-HOLDOUT_MONTHS:].values

        result = _forecast_one_series(train, periods=HOLDOUT_MONTHS)
        forecast_holdout = result["forecast"]

        score = mape(actual_holdout, forecast_holdout)
        if score is None:
            verdict = "n/a (holdout was all zero)"
            score_str = "n/a"
        else:
            score_str = f"{score:.1f}%"
            if score < 15: verdict = "Strong"
            elif score < 30: verdict = "Reasonable"
            elif score < 50: verdict = "Directional only"
            else: verdict = "Not reliable yet"

        print(f"{division:<15}{result['method']:<38}{score_str:<10}{verdict}")

        for m, a, f in zip(series.index[-HOLDOUT_MONTHS:].strftime("%b %Y"), actual_holdout, forecast_holdout):
            print(f"      {m}:  actual {a:>10,.0f}   forecast {f:>10,.0f}")
        print()


if __name__ == "__main__":
    main()
