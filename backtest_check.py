"""
backtest_check.py
==================
Answers the question: "is the model actually any good?"

How it works:
  Pretends the last HOLDOUT_MONTHS known COMPLETE months never happened,
  forecasts them using only the data before that (which itself now
  internally runs its own walk-forward backtest to pick the best model —
  see src/forecast_model.py), then compares that forecast to what
  actually happened. This is the standard way to sanity-check a
  forecasting model — testing on data it was NOT trained on, not just
  checking that it runs.

  IMPORTANT FIX: earlier versions of this script held out the last 3
  calendar months blindly, which on this data included a PARTIAL month
  (the export was pulled mid-month, so the latest month only has ~3
  weeks of orders in it). Comparing a full-month forecast against a
  partial-month actual made the model look far worse than it really is
  — that inflated MAPE was measuring a data artifact, not model error.
  This script now detects and excludes that partial trailing month
  before choosing the holdout window.

Run:
    python backtest_check.py

Reading the output:
  MAPE (Mean Absolute Percentage Error) = average % the forecast was off
  by, on this specific holdout. Rough guide for this kind of
  order-booking data:
    < 15%   strong
    15-30%  reasonable, usable for planning
    30-50%  directional only, use with caution
    > 50%   not reliable yet — needs more/better data (this is expected
            for "Others", which has very little history)

  "Model's own backtest MAPE" is the model's measured walk-forward
  accuracy across several rolling folds (see forecast_model.py) — it's
  reported here too so you can sanity-check that the single holdout
  result above isn't a fluke in either direction.
"""

import numpy as np
import pandas as pd
from src.data_source import FileDataSource
from src.forecast_model import build_monthly_series, _forecast_one_series, _effective_cutoff

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

    # Exclude any partial trailing month before building the holdout window,
    # so the holdout "actual" is always a genuine complete month.
    clean_cutoff, was_adjusted = _effective_cutoff(df, None)
    if was_adjusted:
        print(f"Note: source data's trailing month is partial — "
              f"excluding it, using complete data through {clean_cutoff.date()}\n")

    monthly = build_monthly_series(df, value_col="so_qty", range_end=clean_cutoff)

    print(f"Backtest: holding out the last {HOLDOUT_MONTHS} complete months per division\n")
    print(f"{'Division':<15}{'Method':<45}{'Holdout MAPE':<15}{'Own backtest MAPE':<20}{'Verdict'}")
    print("-" * 110)

    for division, grp in monthly.groupby("division"):
        grp = grp.sort_values("month")
        series = grp.set_index("month")["so_qty"]

        if len(series) <= HOLDOUT_MONTHS + 4:
            print(f"{division:<15}{'(not enough history to backtest)':<45}")
            continue

        train = series.iloc[:-HOLDOUT_MONTHS]
        actual_holdout = series.iloc[-HOLDOUT_MONTHS:].values

        result = _forecast_one_series(train, periods=HOLDOUT_MONTHS)
        forecast_holdout = result["forecast"]

        score = mape(actual_holdout, forecast_holdout)
        own_bt = result["backtest_mape"]
        own_bt_str = f"{own_bt:.1f}%" if own_bt is not None else "n/a"

        if score is None:
            verdict = "n/a (holdout was all zero)"
            score_str = "n/a"
        else:
            score_str = f"{score:.1f}%"
            if score < 15: verdict = "Strong"
            elif score < 30: verdict = "Reasonable"
            elif score < 50: verdict = "Directional only"
            else: verdict = "Not reliable yet"

        print(f"{division:<15}{result['method']:<45}{score_str:<15}{own_bt_str:<20}{verdict}")

        for m, a, f in zip(series.index[-HOLDOUT_MONTHS:].strftime("%b %Y"), actual_holdout, forecast_holdout):
            print(f"      {m}:  actual {a:>10,.0f}   forecast {f:>10,.0f}")
        print()


if __name__ == "__main__":
    main()
