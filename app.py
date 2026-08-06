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
    CUTOFF_DATE is None, meaning the model trains on ALL data returned by
    the live API, up to whatever the latest order date is at fetch time —
    it does not stop at a fixed date anymore. (Earlier, while data came
    from a one-time Excel export, CUTOFF_DATE was fixed at 2026-03-31 as a
    safeguard against training on rows beyond what the export was known to
    contain. Now that data is pulled live, there's no such fixed boundary
    to protect against — every /forecast/v1/refresh pulls whatever is
    currently in the source system and trains on all of it.)

MODEL ACCURACY:
    Each division's forecasting method is chosen automatically by
    backtesting several candidates (Holt-Winters, damped/undamped trend,
    seasonal-naive, moving average, Croston's for sparse series) against
    held-out historical months and picking whichever predicted best. The
    resulting backtest accuracy (SMAPE) is returned per division and shown
    on the dashboard's "Model Confidence" panel — see forecast_model.py
    for the full explanation.

LIVE DATA vs EXCEL:
    Controlled entirely by env vars — no code change needed to switch:
        GM_API_BASE_URL   set  -> pulls live data via APIDataSource
        GM_API_BASE_URL unset  -> falls back to the Excel file (local dev/testing)
    Other APIDataSource env vars: GM_API_KEY, GM_API_ENDPOINT (default
    /api/v1/GetSalesHistory), GM_API_AUTH_MODE (api_key | bearer | none).
    See src/data_source.py -> APIDataSource for full details.

WHAT TO HAND THE .NET TEAM:
    Give them the base URL of this service (e.g. https://garmentsmantra.onrender.com).
    They can either (a) embed/link the "/" route directly — the dashboard
    renders itself, nothing for them to build — or (b) call /forecast/v1/all
    or /forecast/v1/division/{name} and render the JSON in their own screen.
    Either way, this file and the dashboard don't change based on their choice.
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from src.data_source import FileDataSource, APIDataSource
from src.forecast_model import forecast_by_division

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "sales_orders_final.xlsx")
DASHBOARD_PATH = os.path.join(BASE_DIR, "output", "dashboard.html")
VENDOR_DIR = os.path.join(BASE_DIR, "output", "vendor")
ASSETS_DIR = os.path.join(BASE_DIR, "output", "assets")

# Which data source to use is driven entirely by env vars, so switching from
# the Excel export to the live API (once .NET hands over real values) is a
# deploy-config change, not a code change:
#   - GM_API_BASE_URL set        -> uses live APIDataSource
#   - GM_API_BASE_URL not set    -> falls back to the Excel FileDataSource
#     (keeps local dev/testing working without needing API access)
USE_LIVE_API = bool(os.environ.get("GM_API_BASE_URL"))

CUTOFF_DATE = None          # None = train on all data up to the latest date present
                             # (was a fixed "2026-03-31" for the Excel-file era; now
                             # that data is live, the cutoff should always be "as of
                             # now" — see module docstring)
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


def _build_data_source():
    if USE_LIVE_API:
        return APIDataSource()  # reads GM_API_BASE_URL / GM_API_KEY / etc from env
    return FileDataSource(DATA_PATH)


def _get_df():
    if _df_cache["df"] is None:
        source = _build_data_source()
        _df_cache["df"] = source.get_sales_data()
    return _df_cache["df"]


def _compute(lookback_years: Optional[float] = None):
    df = _get_df()
    output = forecast_by_division(df, value_col="so_qty", periods=FORECAST_PERIODS,
                                   cutoff_date=CUTOFF_DATE, lookback_years=lookback_years)
    _forecast_cache[lookback_years] = output
    return output


# Every window the dashboard's Training Window buttons can request. Kept in
# one place so the startup warm-up (below) and the dashboard's own options
# can't drift apart.
LOOKBACK_WINDOWS = [None, 2, 3, 5]


@app.on_event("startup")
def _on_startup():
    # Pre-warm EVERY lookback window, not just "All Available" (None). Each
    # window used to only be computed the first time someone clicked that
    # button on the dashboard — which meant that request sat blocked for
    # however long model-fitting took (worse on Render's free-tier CPU than
    # in local testing), and felt like the dashboard was stuck/not loading.
    # Precomputing all of them here means every click is served straight
    # from _forecast_cache with no live computation at all.
    for w in LOOKBACK_WINDOWS:
        _compute(w)


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """Simple liveness check — useful for the .NET team to confirm the service is up.
    Accepts HEAD as well as GET: uptime monitors (e.g. UptimeRobot) default to
    sending HEAD requests, which FastAPI/Starlette doesn't auto-allow on a
    GET-only route — a HEAD request there returns 405, which uptime monitors
    then wrongly report as the service being down."""
    loaded = _forecast_cache.get(None)
    return {
        "status": "ok",
        "divisions_loaded": list(loaded["divisions"].keys()) if loaded else [],
        "cutoff_date": CUTOFF_DATE,
        "data_source": "live_api" if USE_LIVE_API else "excel_file",
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
    data source (live API or Excel file). Call this after the underlying
    data has changed, instead of restarting the whole service — e.g. on a
    schedule from the .NET side, or manually. For the Excel source this also
    deletes the parquet cache so the next load re-reads the file instead of
    serving a stale cached copy; the live API source has no such cache, so
    this always re-fetches from it.
    """
    if not USE_LIVE_API:
        cache_path = os.path.splitext(DATA_PATH)[0] + ".parquet"
        if os.path.exists(cache_path):
            os.remove(cache_path)

    _df_cache["df"] = None
    windows_to_refresh = list(_forecast_cache.keys()) or LOOKBACK_WINDOWS
    _forecast_cache.clear()
    for w in windows_to_refresh:
        _compute(w)
    return {
        "status": "refreshed",
        "windows_recomputed": [w or "all" for w in windows_to_refresh],
        "data_source": "live_api" if USE_LIVE_API else "excel_file",
    }


@app.api_route("/", methods=["GET", "HEAD"])
def serve_dashboard():
    """Serves the dashboard UI, which itself calls /forecast/v1/all — this
    is the full frontend+backend loop running together. Also accepts HEAD —
    see the note on /health above; this is the route UptimeRobot is actually
    pinging (garmentsmantra.onrender.com/), so it needs the same fix."""
    return FileResponse(DASHBOARD_PATH)


if __name__ == "__main__":
    # Lets this run as `python app.py` too, not just `uvicorn app:app ...`.
    # Render (and most PaaS platforms) set the PORT env var — binding to
    # 0.0.0.0 is required so the platform's router can reach the container.
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
