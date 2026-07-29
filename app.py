"""
app.py
======
Demand Forecasting microservice for Garments Mantra.

This is what the .NET team calls. They don't run Python, don't touch the
model, and don't need to know it's Python at all — they just hit these
HTTP endpoints and get JSON back, exactly like any other API in the system.

RUN LOCALLY:
    uvicorn app:app --reload --port 8000

Then either:
  - Open http://localhost:8000/  in a browser -> serves the dashboard,
    which itself calls the endpoints below (proves the full loop works).
  - Or call the endpoints directly, e.g.:
      curl http://localhost:8000/forecast/v1/all
      curl "http://localhost:8000/forecast/v1/all?lookback_years=2"
      curl http://localhost:8000/forecast/v1/division/Woven

TRAINING CUTOFF:
    The model trains on data up to CUTOFF_DATE below (2026-03-31). The
    source file (data/sales_orders_final.xlsx) is itself already trimmed
    to this date by the team — CUTOFF_DATE is kept as an explicit
    safeguard so a future export with more recent rows doesn't silently
    get pulled into training without a deliberate decision to move the
    cutoff forward.

MODEL ACCURACY:
    Each division's forecasting method is chosen automatically by
    backtesting several candidates (Holt-Winters, damped/undamped trend,
    seasonal-naive, moving average, Croston's for sparse series) against
    held-out historical months and picking whichever predicted best. The
    resulting backtest accuracy (SMAPE) is returned per division and shown
    on the dashboard's "Model Confidence" panel — see forecast_model.py
    for the full explanation.

WHEN THE .NET APIs (GetSalesHistory etc.) ARE READY:
    Only src/data_source.py changes (swap FileDataSource for APIDataSource).
    This file, the model, and the dashboard don't need to change at all —
    that's the whole point of keeping the data source behind one interface.
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from src.data_source import FileDataSource
from src.forecast_model import forecast_by_division

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "sales_orders_final.xlsx")
DASHBOARD_PATH = os.path.join(BASE_DIR, "output", "dashboard.html")
VENDOR_DIR = os.path.join(BASE_DIR, "output", "vendor")
ASSETS_DIR = os.path.join(BASE_DIR, "output", "assets")

CUTOFF_DATE = "2026-03-31"   # train only on data up to here — see module docstring
FORECAST_PERIODS = 4         # months ahead to forecast

app = FastAPI(
    title="Garments Mantra — Demand Forecasting Service",
    description="Division-level sales forecasting with a selectable historical lookback window.",
    version="1.1.0",
)

# Serves output/vendor/chart.umd.js locally at /vendor/chart.umd.js — this is
# what lets the dashboard's chart work on networks that block public CDNs
# like cdnjs.cloudflare.com (common on corporate networks).
app.mount("/vendor", StaticFiles(directory=VENDOR_DIR), name="vendor")

# Serves the Garments Mantra logo and any other dashboard branding assets.
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Allows the dashboard (or any other origin, incl. the ERP front end) to
# call this API directly from a browser. Tighten allow_origins to the
# actual ERP domain before this goes anywhere near production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The raw dataframe is cached separately from computed forecasts — loading
# it is the expensive part (Excel parse / parquet cache). Forecasts for
# each lookback window (2 / 3 / 5 years / all) are cheap to compute and are
# cached per-window so switching the dashboard's dropdown doesn't require
# reloading the whole dataset.
_df_cache: dict = {"df": None}
_forecast_cache: dict = {}  # keyed by lookback_years (None means "all")


def _get_df():
    if _df_cache["df"] is None:
        _df_cache["df"] = FileDataSource(DATA_PATH).get_sales_data()
    return _df_cache["df"]


def _compute(lookback_years: Optional[float] = None):
    df = _get_df()
    output = forecast_by_division(df, value_col="so_qty", periods=FORECAST_PERIODS,
                                   cutoff_date=CUTOFF_DATE, lookback_years=lookback_years)
    _forecast_cache[lookback_years] = output
    return output


@app.on_event("startup")
def _on_startup():
    _compute(None)  # pre-warm the default (all-history) view


@app.get("/health")
def health():
    """Simple liveness check — useful for the .NET team to confirm the service is up."""
    loaded = _forecast_cache.get(None)
    return {
        "status": "ok",
        "divisions_loaded": list(loaded["divisions"].keys()) if loaded else [],
        "cutoff_date": CUTOFF_DATE,
    }


@app.get("/forecast/v1/all")
def get_all_forecasts(
    lookback_years: Optional[float] = Query(
        None, description="Restrict training data to this many years before the cutoff. "
                           "Omit for all available history. Examples: 2, 3, 5."
    )
):
    """
    Returns the forecast for every division. This is what the dashboard calls.
    Pass ?lookback_years=2|3|5 to control how much history feeds the model —
    if the data doesn't go back that far, all available data is used instead
    (see the "lookback_clamped" flag in the response's "meta" section).
    """
    if lookback_years not in _forecast_cache:
        _compute(lookback_years)
    return _forecast_cache[lookback_years]


@app.get("/forecast/v1/division/{division_name}")
def get_division_forecast(
    division_name: str,
    lookback_years: Optional[float] = Query(None),
):
    """Returns the forecast for a single division (case/spacing-insensitive)."""
    if lookback_years not in _forecast_cache:
        _compute(lookback_years)

    divisions = _forecast_cache[lookback_years]["divisions"]
    normalized = division_name.lower().replace(" ", "").replace("-", "")
    for name, result in divisions.items():
        if name.lower().replace(" ", "").replace("-", "") == normalized:
            return result

    raise HTTPException(
        status_code=404,
        detail=f"Division '{division_name}' not found. Available: {list(divisions.keys())}",
    )


@app.post("/forecast/v1/refresh")
def refresh_forecast():
    """
    Recomputes forecasts for all cached lookback windows from the current
    data file. Call this after the underlying Excel/API data has been
    updated, instead of restarting the whole service. Also deletes the
    parquet cache so the next load actually re-reads the source file
    rather than serving stale cached data.
    """
    cache_path = os.path.splitext(DATA_PATH)[0] + ".parquet"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    _df_cache["df"] = None
    windows_to_refresh = list(_forecast_cache.keys()) or [None]
    _forecast_cache.clear()
    for w in windows_to_refresh:
        _compute(w)
    return {"status": "refreshed", "windows_recomputed": [w or "all" for w in windows_to_refresh]}


@app.get("/")
def serve_dashboard():
    """Serves the dashboard UI, which itself calls /forecast/v1/all — this
    is the full frontend+backend loop running together."""
    return FileResponse(DASHBOARD_PATH)


if __name__ == "__main__":
    # Lets this run as `python app.py` too, not just `uvicorn app:app ...`.
    # Render (and most PaaS platforms) set the PORT env var — binding to
    # 0.0.0.0 is required so the platform's router can reach the container.
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
