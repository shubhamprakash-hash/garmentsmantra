"""
app.py
======
Demand Forecasting microservice for Garments Mantra — Phase 1 (sales only).

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
      curl http://localhost:8000/forecast/v1/division/Woven

WHEN THE .NET APIs (GetSalesHistory etc.) ARE READY:
    Only src/data_source.py changes (swap FileDataSource for APIDataSource).
    This file, the model, and the dashboard don't need to change at all —
    that's the whole point of keeping the data source behind one interface.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.data_source import FileDataSource
from src.forecast_model import forecast_by_division

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "sales_orders_v2.xlsx")
DASHBOARD_PATH = os.path.join(BASE_DIR, "output", "dashboard.html")
VENDOR_DIR = os.path.join(BASE_DIR, "output", "vendor")
ASSETS_DIR = os.path.join(BASE_DIR, "output", "assets")

app = FastAPI(
    title="Garments Mantra — Demand Forecasting Service",
    description="Phase 1: division-level sales forecasting.",
    version="1.0.0",
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

# Forecast is computed once at startup and cached in memory — recomputing
# on every request would mean re-reading a 70MB+ Excel file per call.
# Call POST /forecast/v1/refresh after the underlying data changes.
_cache: dict = {"results": None}


def _compute():
    df = FileDataSource(DATA_PATH).get_sales_data()
    results = forecast_by_division(df, value_col="so_qty", periods=3)
    _cache["results"] = results
    return results


@app.on_event("startup")
def _on_startup():
    _compute()


@app.get("/health")
def health():
    """Simple liveness check — useful for the .NET team to confirm the service is up."""
    return {"status": "ok", "divisions_loaded": list((_cache["results"] or {}).keys())}


@app.get("/forecast/v1/all")
def get_all_forecasts():
    """Returns the forecast for every division. This is what the dashboard calls."""
    if _cache["results"] is None:
        _compute()
    return _cache["results"]


@app.get("/forecast/v1/division/{division_name}")
def get_division_forecast(division_name: str):
    """Returns the forecast for a single division (case/spacing-insensitive)."""
    if _cache["results"] is None:
        _compute()

    normalized = division_name.lower().replace(" ", "").replace("-", "")
    for name, result in _cache["results"].items():
        if name.lower().replace(" ", "").replace("-", "") == normalized:
            return result

    raise HTTPException(
        status_code=404,
        detail=f"Division '{division_name}' not found. Available: {list(_cache['results'].keys())}",
    )


@app.post("/forecast/v1/refresh")
def refresh_forecast():
    """
    Recomputes the forecast from the current data file. Call this after the
    underlying Excel/API data has been updated, instead of restarting the
    whole service. Also deletes the parquet cache so the next load actually
    re-reads the source file rather than serving stale cached data.
    """
    cache_path = os.path.splitext(DATA_PATH)[0] + ".parquet"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return _compute()


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
