"""
forecast_model.py
==================
Division-level monthly demand forecast, with a configurable training cutoff,
a selectable historical lookback window, and automatic model selection based
on backtested accuracy rather than a fixed rule.

Why division-level, not design-level:
  The vast majority of Design Nos appear only once or twice across the
  full history — there isn't enough repeat history to forecast per-design.
  That's the classic garments "cold start" problem. Division (Knits-Kids /
  Knits-Men / Woven / Others) is the lowest level with enough repeated
  monthly observations to fit a model on.

Why SO Qty as the demand signal (for now):
  This data only has order-booking (SO Qty) and dispatch (Dispatch Qty),
  not true sell-through. SO Qty is used as the working proxy for demand
  until GetSalesHistory (actual sales/invoice data) is available from the
  .NET team — see data_source.py. Swapping the input column later requires
  no change to the modeling logic below.

  IMPORTANT HONEST NOTE: order-booking data for this business is genuinely
  lumpy (a single bulk order can be 5-10x a normal month) — no amount of
  model tuning turns that into a smooth, easily-forecastable series. What
  changed here is that the SELECTED model is now chosen by actually
  checking which one predicts held-out months best, instead of a fixed
  rule — so the forecast reflects the best achievable fit to this data,
  not just the first "reasonable-sounding" method. Real accuracy gains
  beyond this point come from better data (true sell-through, not
  bookings), not from a fancier model on the same data.

Training cutoff (CUTOFF_DATE, set in app.py / run_forecast.py):
  The model trains only on data up to a fixed cutoff (2026-03-31), even
  though the source file has data beyond that. This lets the forecast for
  the months after the cutoff be checked against what actually happened —
  see get_actuals_for_months() — which is also how model selection below
  is validated (backtesting against real held-out months, same principle).

Lookback window:
  The dashboard lets the user choose how much history feeds the model —
  2 / 3 / 5 years back from the cutoff, or all available.

MODEL SELECTION (the core change in this version):
  For each division, several candidate methods are fit and backtested via
  rolling-origin validation (train on an earlier cutoff, forecast the next
  `periods` months, compare to what's already known, repeat for 1-2 more
  origins). The candidate with the lowest average SMAPE (symmetric
  percentage error — handles zeros better than plain MAPE) is used for
  the final forecast. Candidates:
    1. Holt-Winters, damped trend + additive seasonality, fit in log-space
       (log1p) — damping prevents the trend from compounding into an
       unrealistic runaway forecast; log-space keeps a few huge bulk-order
       months from distorting the whole fit. Needs 24+ active months.
    2. Holt-Winters, non-damped trend + seasonality, log-space. Same data
       requirement — included because damping isn't always better; let
       the backtest decide.
    3. Holt's trend, damped, log-space. No seasonality — for series with
       a trend but not enough history/regularity for reliable seasonality.
    4. Holt's trend, non-damped, log-space.
    5. Seasonal naive — forecast month = the same calendar month one year
       earlier. A famously hard baseline to beat on real seasonal business
       data, and immune to overfitting since it has no parameters at all.
    6. 3-month moving average — simple, robust fallback.
    7. Croston's method — for intermittent/sparse series (e.g. "Others"),
       always included as a candidate since trend/seasonal methods
       structurally don't make sense for mostly-zero series.
  Whichever of these actually backtests best for a given division wins —
  the choice isn't fixed per division ahead of time.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Data aggregation & actuals lookup (unchanged from previous version)
# =============================================================================

def build_monthly_series(df: pd.DataFrame, value_col: str = "so_qty",
                          range_start=None, range_end=None) -> pd.DataFrame:
    """
    Aggregate raw order rows into a division x month demand table, with
    every calendar month present (missing months filled with 0) across
    [range_start, range_end] inclusive.
    """
    all_divisions = df["division"].unique()

    d = df
    if range_start is not None:
        d = d[d["order_date"] >= range_start]
    if range_end is not None:
        d = d[d["order_date"] <= range_end]

    monthly = (
        d.assign(month=d["order_date"].dt.to_period("M").dt.to_timestamp())
         .groupby(["division", "month"])[value_col]
         .sum()
         .reset_index()
    )

    if range_start is not None and range_end is not None:
        full_range = pd.date_range(range_start.to_period("M").to_timestamp(),
                                    range_end.to_period("M").to_timestamp(), freq="MS")
    elif len(monthly):
        full_range = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    else:
        full_range = pd.DatetimeIndex([])

    filled = []
    for division in all_divisions:
        grp = monthly[monthly["division"] == division]
        if len(grp):
            s = grp.set_index("month")[value_col].reindex(full_range, fill_value=0)
        else:
            s = pd.Series(0.0, index=full_range)
        filled.append(pd.DataFrame({"division": division, "month": full_range, value_col: s.values}))

    return pd.concat(filled, ignore_index=True) if filled else pd.DataFrame(columns=["division", "month", value_col])


def get_actuals_for_months(full_df: pd.DataFrame, value_col: str, division: str,
                            months: pd.DatetimeIndex) -> dict:
    """
    Looks up what ACTUALLY happened in `months` for `division`, using the
    full (untrimmed) dataset — even though the model itself was only
    trained up to the cutoff.
    """
    data_max = full_df["order_date"].max()
    sub = full_df[full_df["division"] == division]

    values, status = [], []
    for m in months:
        month_end = m + pd.offsets.MonthEnd(0)
        if data_max < m:
            values.append(None)
            status.append("unavailable")
        else:
            month_total = sub[(sub["order_date"] >= m) & (sub["order_date"] <= month_end)][value_col].sum()
            values.append(float(month_total))
            status.append("complete" if data_max >= month_end else "partial")

    return {"values": values, "status": status}


# =============================================================================
# Accuracy metric
# =============================================================================

def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE — bounded 0-200%, doesn't blow up when actual is 0
    (plain MAPE is undefined there), which matters a lot for sparse series."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual) + np.abs(forecast)
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(2 * np.abs(forecast - actual) / denom) * 100)


# =============================================================================
# Candidate forecasting methods
# All take (values: np.ndarray, periods: int) -> np.ndarray of length periods
# =============================================================================

def _fit_ets_log(values: np.ndarray, periods: int, seasonal: bool, damped: bool):
    """Shared ETS fitter, working in log1p-space to tame huge bulk-order
    spikes, with optional trend damping to prevent runaway extrapolation."""
    log_vals = np.log1p(np.clip(values, a_min=0, a_max=None))
    kwargs = dict(trend="add", damped_trend=damped, initialization_method="estimated")
    if seasonal:
        kwargs.update(seasonal="add", seasonal_periods=12)
    model = ExponentialSmoothing(log_vals, **kwargs)
    fit = model.fit(optimized=True)
    log_forecast = fit.forecast(periods)
    return np.expm1(np.clip(log_forecast, a_min=None, a_max=20))  # clip guards against pathological blowups


def _candidate_hw_damped(values, periods):
    return _fit_ets_log(values, periods, seasonal=True, damped=True)


def _candidate_hw_undamped(values, periods):
    return _fit_ets_log(values, periods, seasonal=True, damped=False)


def _candidate_trend_damped(values, periods):
    return _fit_ets_log(values, periods, seasonal=False, damped=True)


def _candidate_trend_undamped(values, periods):
    return _fit_ets_log(values, periods, seasonal=False, damped=False)


def _candidate_seasonal_naive(values, periods):
    """Forecast = the value from the same calendar month one year earlier.
    No parameters to overfit — a strong, boringly-reliable baseline for
    genuinely seasonal business data."""
    if len(values) < 12:
        raise ValueError("not enough history for seasonal naive")
    last_year = values[-12:]
    reps = int(np.ceil(periods / 12))
    return np.tile(last_year, reps)[:periods]


def _candidate_moving_avg(values, periods):
    window = values[-3:] if len(values) >= 3 else values
    avg = window.mean() if len(window) else 0.0
    return np.repeat(avg, periods)


def _candidate_croston(values, periods, alpha: float = 0.1):
    """Croston's method for intermittent/lumpy demand (mostly zero months
    with occasional spikes) — see module docstring."""
    demand_sizes, intervals, gap = [], [], 1
    for v in values:
        if v > 0:
            demand_sizes.append(v)
            intervals.append(gap)
            gap = 1
        else:
            gap += 1

    if not demand_sizes:
        return np.zeros(periods)

    a, p = demand_sizes[0], intervals[0]
    for i in range(1, len(demand_sizes)):
        a = alpha * demand_sizes[i] + (1 - alpha) * a
        p = alpha * intervals[i] + (1 - alpha) * p

    rate = a / p if p > 0 else 0.0
    return np.repeat(rate, periods)


# Each candidate: (name, fit_fn, minimum_history_months_required)
_CANDIDATES = [
    ("Holt-Winters (damped trend + seasonality, log-space)", _candidate_hw_damped, 24),
    ("Holt-Winters (trend + seasonality, log-space)", _candidate_hw_undamped, 24),
    ("Holt's trend (damped, log-space)", _candidate_trend_damped, 6),
    ("Holt's trend (log-space)", _candidate_trend_undamped, 6),
    ("Seasonal naive (same month last year)", _candidate_seasonal_naive, 13),
    ("3-month moving average", _candidate_moving_avg, 1),
    ("Croston's method (intermittent demand)", _candidate_croston, 1),
]


# =============================================================================
# Rolling-origin backtest — picks the candidate that actually predicts best
# =============================================================================

def _backtest_candidate(values: np.ndarray, fit_fn, periods: int, min_history: int,
                         max_folds: int = 2) -> float | None:
    """
    Rolling-origin backtest: repeatedly pretend an earlier point was "now",
    forecast forward `periods` months, and compare to what's already known
    to have happened. Returns the average SMAPE across folds, or None if
    there isn't enough history to run even one fold.
    """
    n = len(values)
    scores = []
    for fold in range(max_folds):
        holdout_end = n - fold * periods
        holdout_start = holdout_end - periods
        train_end = holdout_start
        if train_end < min_history or holdout_start < 0:
            break
        train = values[:train_end]
        actual = values[holdout_start:holdout_end]
        try:
            forecast = fit_fn(train, periods)
            if len(forecast) != periods or np.any(~np.isfinite(forecast)):
                continue
            scores.append(_smape(actual, np.clip(forecast, 0, None)))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else None


def _compute_all_candidates(values: np.ndarray, periods: int) -> list[dict]:
    """
    Fits and backtests EVERY candidate in _CANDIDATES exactly once. This is
    the single expensive pass — both the "auto" model selection and the
    dashboard's full method-comparison dropdown are built from this same
    list afterwards, instead of each independently re-fitting/backtesting
    (which used to double the model-fitting cost per division and was the
    main reason switching the dashboard's lookback window felt slow).
    Each entry: {name, fit_fn, min_hist, forecast (np.ndarray), backtest_smape}.
    Candidates with too little history, or that fail to fit on the full
    series, are omitted.
    """
    candidates = []
    for name, fit_fn, min_hist in _CANDIDATES:
        if len(values) < min_hist:
            continue
        try:
            point_forecast = np.clip(fit_fn(values, periods), a_min=0, a_max=None)
            if len(point_forecast) != periods or np.any(~np.isfinite(point_forecast)):
                continue
        except Exception:
            continue
        backtest_smape = _backtest_candidate(values, fit_fn, periods, min_hist)
        candidates.append({
            "name": name, "fit_fn": fit_fn, "min_hist": min_hist,
            "forecast": point_forecast, "backtest_smape": backtest_smape,
        })
    return candidates


def _band_for(point_forecast: np.ndarray, backtest_smape: float | None, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Confidence band from backtest SMAPE when available, else in-sample residual spread."""
    if backtest_smape is not None:
        err_frac = min(backtest_smape / 100, 1.5)  # cap so a bad backtest doesn't make bands absurd
        lower = np.clip(point_forecast * (1 - err_frac), a_min=0, a_max=None)
        upper = point_forecast * (1 + err_frac)
    else:
        resid_std = values.std() if len(values) > 1 else point_forecast.mean() * 0.3
        lower = np.clip(point_forecast - 1.28 * resid_std, a_min=0, a_max=None)
        upper = point_forecast + 1.28 * resid_std
    return lower, upper


def _select_best_model(values: np.ndarray, candidates: list[dict]) -> dict | None:
    """
    Picks whichever already-computed candidate scored lowest backtest SMAPE,
    restricted to the subset that's structurally appropriate for how sparse
    the series is — this matters because a flat-rate method (Croston's) can
    spuriously "win" a small backtest sample purely by luck even on a dense,
    regular series, since it never overshoots the way a trend/seasonal model
    can. Croston's is only eligible to WIN for genuinely intermittent series
    (active_ratio < 0.4) — it's not statistically meant for anything else —
    though it's still shown, unfiltered, in the full method-comparison list.
    Returns None if nothing could be backtested (falls back to a simple
    structurally-appropriate default in the caller).
    """
    n_active = int((values > 0).sum())
    active_ratio = n_active / len(values) if len(values) else 0
    is_sparse = active_ratio < 0.4

    if is_sparse:
        eligible = [c for c in candidates if c["fit_fn"] in (_candidate_croston, _candidate_moving_avg)]
    else:
        eligible = [c for c in candidates if c["fit_fn"] is not _candidate_croston]

    scored = [c for c in eligible if c["backtest_smape"] is not None]
    if scored:
        return min(scored, key=lambda c: c["backtest_smape"])
    return None


def _forecast_one_series(series: pd.Series, periods: int) -> dict:
    """
    Selects the best-backtested model for this division's series, fits it
    on the FULL available series, and forecasts `periods` months ahead.
    Uncertainty band is derived from the backtest's out-of-sample error
    when available (more honest than in-sample fit residuals), falling
    back to in-sample residual spread otherwise.
    """
    values = series.values.astype(float)
    n_active = int((values > 0).sum())
    active_ratio = n_active / len(values) if len(values) else 0
    is_sparse = active_ratio < 0.4

    candidates = _compute_all_candidates(values, periods)
    best = _select_best_model(values, candidates)

    if best is not None:
        method, fit_fn = best["name"], best["fit_fn"]
        point_forecast, backtest_smape = best["forecast"], best["backtest_smape"]
    else:
        # Nothing could be backtested (very short series) — fall back to
        # the structurally-appropriate simple method.
        if is_sparse:
            method, fit_fn = "Croston's method (intermittent demand, insufficient history to backtest)", _candidate_croston
        else:
            method, fit_fn = "3-month moving average (insufficient history to backtest)", _candidate_moving_avg
        try:
            point_forecast = np.clip(fit_fn(values, periods), a_min=0, a_max=None)
        except Exception:
            point_forecast = _candidate_moving_avg(values, periods)
            method = "3-month moving average (fallback — selected model failed on full data)"
        backtest_smape = None

    # Guard against a flat all-zero forecast for a division that HAS had
    # real orders before — this happens when the winning method (e.g.
    # 3-month moving average) simply landed on a recent run of zero
    # months, which backtests fine (zero predicting zero is a "perfect"
    # score) but isn't useful for planning. Croston's method instead uses
    # the whole order history and the gaps between orders, so it still
    # produces a non-zero rate for a division that orders sporadically.
    if n_active > 0 and not np.any(point_forecast > 0) and fit_fn is not _candidate_croston:
        croston_entry = next((c for c in candidates if c["fit_fn"] is _candidate_croston), None)
        croston_forecast = croston_entry["forecast"] if croston_entry is not None else _candidate_croston(values, periods)
        if np.any(croston_forecast > 0):
            point_forecast = croston_forecast
            method = (f"Croston's method (intermittent demand — {method} "
                      "produced an all-zero forecast, which isn't useful for "
                      "planning, so an intermittent-demand estimate is used instead)")
            backtest_smape = None

    lower, upper = _band_for(point_forecast, backtest_smape, values)

    all_methods = []
    for c in candidates:
        m_lower, m_upper = _band_for(c["forecast"], c["backtest_smape"], values)
        all_methods.append({
            "name": c["name"],
            "backtest_smape": round(c["backtest_smape"], 1) if c["backtest_smape"] is not None else None,
            "forecast": c["forecast"].tolist(),
            "lower": m_lower.tolist(),
            "upper": m_upper.tolist(),
        })
    # Show the more accurate (lower SMAPE) methods first; methods that
    # couldn't be backtested (None) are listed last rather than first.
    all_methods.sort(key=lambda m: (m["backtest_smape"] is None, m["backtest_smape"] or 0))

    return {
        "method": method,
        "backtest_smape": round(backtest_smape, 1) if backtest_smape is not None else None,
        "history": values.tolist(),
        "forecast": point_forecast.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "active_months": n_active,
        "all_methods": all_methods,
    }


def forecast_by_division(df: pd.DataFrame, value_col: str = "so_qty", periods: int = 4,
                          cutoff_date=None, lookback_years=None) -> dict:
    """
    Top-level entry point.

    cutoff_date:    train only on data up to and including this date (e.g.
                    "2026-03-31"). Defaults to the latest date in the data
                    if not given.
    lookback_years: if given, further restrict training data to the N years
                    immediately before cutoff_date. Clamped to available
                    data if the request exceeds it (reported in meta).

    Returns {"divisions": {name: {...}}, "meta": {...}}.
    """
    if cutoff_date is None:
        cutoff_date = df["order_date"].max()
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    data_min = df["order_date"].min()
    data_max = df["order_date"].max()

    range_start = None
    lookback_clamped = False
    if lookback_years is not None:
        requested_start = cutoff_date - pd.DateOffset(years=lookback_years) + pd.Timedelta(days=1)
        if requested_start < data_min:
            range_start = data_min
            lookback_clamped = True
        else:
            range_start = requested_start

    monthly = build_monthly_series(df, value_col=value_col, range_start=range_start, range_end=cutoff_date)

    results = {}
    for division, grp in monthly.groupby("division"):
        grp = grp.sort_values("month")
        series = grp.set_index("month")[value_col]

        result = _forecast_one_series(series, periods=periods)
        history_months = [d.strftime("%b %Y") for d in series.index]

        last_month = series.index.max()
        future_months = pd.date_range(last_month, periods=periods + 1, freq="MS")[1:]
        forecast_months = [d.strftime("%b %Y") for d in future_months]

        actuals = get_actuals_for_months(df, value_col, division, future_months)

        result["history_months"] = history_months
        result["forecast_months"] = forecast_months
        result["actual"] = actuals["values"]
        result["actual_status"] = actuals["status"]
        result["total_history_orders"] = int(len(df[(df["division"] == division) &
                                                      (df["order_date"] >= (range_start or data_min)) &
                                                      (df["order_date"] <= cutoff_date)]))

        # Sparse/intermittent flag + last real order — lets the dashboard
        # explain a zero "Last 3-Mo Avg" (e.g. "Sporadic — last order Dec
        # 2025") instead of showing a bare 0 with no context.
        active_ratio = result["active_months"] / len(series) if len(series) else 0
        result["is_sparse"] = active_ratio < 0.4
        nonzero_idx = np.nonzero(series.values > 0)[0]
        if len(nonzero_idx):
            last_idx = int(nonzero_idx[-1])
            result["last_active_month"] = history_months[last_idx]
            result["last_active_value"] = float(series.values[last_idx])
        else:
            result["last_active_month"] = None
            result["last_active_value"] = None

        results[division] = result

    meta = {
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "data_min_date": data_min.strftime("%Y-%m-%d"),
        "data_max_date": data_max.strftime("%Y-%m-%d"),
        "lookback_years_requested": lookback_years,
        "lookback_clamped": lookback_clamped,
        "training_start": (range_start or data_min).strftime("%Y-%m-%d"),
        "training_months": len(next(iter(results.values()))["history_months"]) if results else 0,
        "forecast_periods": periods,
    }

    return {"divisions": results, "meta": meta}
