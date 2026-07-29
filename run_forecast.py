"""
run_forecast.py
================
Entry point for division-level sales forecasting.

USAGE TODAY:
    python run_forecast.py
    (reads data/sales_orders_final.xlsx — the official export, already
    trimmed by the team to data through 31 Mar 2026)

CUTOFF_DATE below is kept as an explicit safeguard even though the source
file itself already stops at 31 Mar 2026 — if a future export ever
includes more recent rows, training still won't silently pull them in
without CUTOFF_DATE being deliberately moved forward first.

USAGE ONCE THE .NET APIs ARE READY:
    Change the `source = FileDataSource(...)` line below to:
        source = APIDataSource(base_url="https://<erp-host>", api_key="<key>")
    Nothing else in this file, or in forecast_model.py, needs to change.
"""

import json
import os
from src.data_source import FileDataSource
from src.forecast_model import forecast_by_division

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_orders_final.xlsx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "forecast_results.json")

CUTOFF_DATE = "2026-03-31"   # train only on data up to here
FORECAST_PERIODS = 4         # months ahead to forecast
LOOKBACK_YEARS = None        # None = use all available history


def main():
    # --- 1. Load data (swap this line for APIDataSource later) ---
    source = FileDataSource(DATA_PATH)
    df = source.get_sales_data()

    print(f"Loaded {len(df)} order rows across {df['division'].nunique()} divisions")
    print(f"Full data range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print(f"Training cutoff: {CUTOFF_DATE}  |  Forecast horizon: {FORECAST_PERIODS} months"
          f"  |  Lookback: {LOOKBACK_YEARS or 'all available'}")
    print()

    # --- 2. Forecast (SO Qty used as the demand proxy — see forecast_model.py) ---
    output = forecast_by_division(df, value_col="so_qty", periods=FORECAST_PERIODS,
                                   cutoff_date=CUTOFF_DATE, lookback_years=LOOKBACK_YEARS)
    results = output["divisions"]
    meta = output["meta"]

    print(f"Trained on {meta['training_months']} months "
          f"({meta['training_start']} to {meta['cutoff_date']})"
          + (" [lookback clamped to available data]" if meta["lookback_clamped"] else ""))
    print()

    # --- 3. Print a quick summary to console, including actual-vs-forecast where known ---
    for division, r in results.items():
        print(f"[{division}]  method: {r['method']}  (orders in training window: {r['total_history_orders']})")
        for m, v, lo, hi, act, act_status in zip(
            r["forecast_months"], r["forecast"], r["lower"], r["upper"], r["actual"], r["actual_status"]
        ):
            line = f"   {m}:  forecast {v:,.0f}   (range {lo:,.0f} - {hi:,.0f})"
            if act_status == "complete":
                diff_pct = ((act - v) / act * 100) if act else 0
                line += f"   | ACTUAL {act:,.0f}  (forecast was {diff_pct:+.1f}% vs actual)"
            elif act_status == "partial":
                line += f"   | ACTUAL {act:,.0f} (partial month — not a fair comparison)"
            print(line)
        print()

    # --- 4. Export for the dashboard ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
