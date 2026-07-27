"""
run_forecast.py
================
Entry point for Phase 1: division-level sales forecasting.

USAGE TODAY:
    python run_forecast.py
    (reads data/sales_orders_v2.xlsx — the latest export from the team)

USAGE ONCE THE .NET APIs ARE READY:
    Change the `source = FileDataSource(...)` line below to:
        source = APIDataSource(base_url="https://<erp-host>", api_key="<key>")
    Nothing else in this file, or in forecast_model.py, needs to change.
"""

import json
import os
from src.data_source import FileDataSource
from src.forecast_model import forecast_by_division

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_orders_v2.xlsx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "forecast_results.json")


def main():
    # --- 1. Load data (swap this line for APIDataSource later) ---
    source = FileDataSource(DATA_PATH)
    df = source.get_sales_data()

    print(f"Loaded {len(df)} order rows across {df['division'].nunique()} divisions")
    print(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print()

    # --- 2. Forecast (SO Qty used as the demand proxy — see forecast_model.py) ---
    results = forecast_by_division(df, value_col="so_qty", periods=3)

    # --- 3. Print a quick summary to console ---
    for division, r in results.items():
        print(f"[{division}]  method: {r['method']}  (history: {r['total_history_orders']} orders)")
        for m, v, lo, hi in zip(r["forecast_months"], r["forecast"], r["lower"], r["upper"]):
            print(f"   {m}:  {v:,.0f}   (range {lo:,.0f} – {hi:,.0f})")
        print()

    # --- 4. Export for the dashboard ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
