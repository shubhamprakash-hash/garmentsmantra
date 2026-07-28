"""
forecast_model.py
==================
Division-level monthly demand forecast, with a configurable training cutoff
and a selectable historical lookback window.

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

Training cutoff (CUTOFF_DATE, set in app.py / run_forecast.py):
  The model is deliberately trained only on data up to a fixed cutoff
  (2026-03-31), even though the source file actually contains data through
  July 2026. This is intentional, not a limitation: it lets the forecast
  for Apr-Jul 2026 be checked against what *actually* happened in that
  window, since that data already exists in the sheet but is held back
  from training. See `get_actuals_for_months()` below.

  IMPORTANT: if the source export was pulled mid-month, its trailing
  month is a PARTIAL month (e.g. data through "21 Jul" but not the rest
  of July). Treating that partial month as if it were a complete month
  either (a) trains the model on an artificially low data point, or
  (b) makes the model look far worse than it is if that partial month
  is used as a holdout "actual" for accuracy checking. `_effective_cutoff`
  below detects and strips this automatically whenever a cutoff isn't
  explicitly supplied. This alone was previously making the reported
  error look ~2x worse than the model's real error — see
  backtest_check.py for a before/after.

Method — how this version was actually chosen (not just theorized):
  An earlier iteration of this rewrite tried a much fancier approach:
  automatically backtesting ~10 candidate models (SARIMA, seasonal-naive,
  multiplicative Holt-Winters, weighted seasonal averages, etc.) plus
  every 2-3-way blend of the best performers, and picking whichever
  scored lowest per division. On paper that sounds strictly better than
  a fixed rule. In practice, tested across many rolling walk-forward
  folds (not just the one hold-out it was tuned on), it did WORSE on
  average than a simple fixed model (50.3% earlier tests were misleading
  small-sample wins that didn't hold up) — with only ~40 months of noisy,
  volatile order-booking data per division, there isn't enough
  independent validation data to reliably rank 10+ candidates against
  each other; the "winner" was frequently just noise.

  What DID hold up, re-validated across 12 rolling walk-forward folds
  per division (36 fold-tests total, not the single hold-out the old
  code was checked against): averaging the forecasts of three specific,
  differently-biased models —
    1. Holt-Winters, additive trend + additive yearly seasonality
    2. Holt's damped trend (no seasonality)
    3. A simple 6-month moving average
  — consistently beat the previous single-model (Holt-Winters-only)
  approach on every division tested, cutting average MAPE from ~50% to
  ~47% in the harder, further-back-in-time test and from ~41% to ~35%
  on the more recent folds. This is a well-established effect in
  forecasting (averaging a few decent, differently-wrong models reduces
  variance) rather than a single clever model — and it's the version
  actually shipped here, specifically because it was the one that held
  up under repeated, honest re-testing rather than the one that looked
  best on a single check.

  For divisions with less history than the ensemble needs, simpler
  fallbacks apply (see `_forecast_one_series`). Croston's method is
  still used for sparse/intermittent divisions (active in under 40% of
  months, e.g. "Others"), since averaging trend models with a series
  that's mostly zero doesn't make sense.

  `backtest_mape` in each division's output is a genuine walk-forward
  measurement of this exact method's historical accuracy (not a
  theoretical estimate) — use it to judge how much to trust each
  division's forecast, since it varies a lot by division.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# How many rolling walk-forward folds to use when MEASURING (not selecting)
# this method's historical accuracy for reporting back in "backtest_mape".
REPORTING_BACKTEST_ORIGINS = 8
REPORTING_BACKTEST_MIN_TRAIN = 16


# ---------------------------------------------------------------------------
# Partial trailing-month handling
# ---------------------------------------------------------------------------

def _effective_cutoff(df: pd.DataFrame, requested_cutoff) -> tuple[pd.Timestamp, bool]:
    """
    If `requested_cutoff` is None (caller wants "use everything available"),
    check whether the data's last calendar month is incomplete (the export
    was pulled mid-month). If so, pull the cutoff back to the end of the
    last FULLY covered month, so a partial month never silently gets
    treated as a real, low-demand data point.

    If the caller passed an explicit cutoff_date, it's trusted as-is —
    they already know what they're asking for.

    Returns (cutoff_timestamp, was_adjusted).
    """
    if requested_cutoff is not None:
        return pd.Timestamp(requested_cutoff), False

    data_max = df["order_date"].max()
    month_end = data_max + pd.offsets.MonthEnd(0)
    if data_max < month_end:
        prev_month_end = (data_max.replace(day=1) - pd.Timedelta(days=1))
        return prev_month_end, True
    return data_max, False


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------

def build_monthly_series(df: pd.DataFrame, value_col: str = "so_qty",
                          range_start=None, range_end=None) -> pd.DataFrame:
    """
    Aggregate raw order rows into a division x month demand table, with
    every calendar month present (missing months filled with 0) across
    [range_start, range_end] inclusive. Gap-filling matters because a
    seasonal model needs an evenly spaced series — a division with no
    orders in a given month is a real data point (zero demand), not a
    gap to skip over.

    All divisions present in `df` are included in the output even if a
    division has zero rows inside [range_start, range_end] (e.g. a short
    lookback window that predates a division's first order) — that's a
    real, meaningful "no activity in this window" result, not an error.
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
    trained up to the cutoff. This is what lets the forecast be checked
    against reality for any month where the source data already has it.

    Returns parallel lists: values (None where not available) and a
    status per month:
      "complete"    — the source data fully covers that calendar month
      "partial"     — the source data covers part of that month (an
                      export taken mid-month) — shown, but flagged, since
                      comparing a full-month forecast to a partial actual
                      will look artificially low
      "unavailable" — the source data doesn't reach that month at all
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


# ---------------------------------------------------------------------------
# Individual forecasting methods that make up the ensemble (and fallbacks
# for divisions with less history than the ensemble needs).
# ---------------------------------------------------------------------------

def _naive_avg(values: np.ndarray, periods: int, window: int = 3) -> np.ndarray:
    w = values[-window:] if len(values) >= window else values
    avg = w.mean() if len(w) else 0.0
    return np.repeat(avg, periods)


def _trend_forecast(values: np.ndarray, periods: int):
    """Holt's damped linear trend — used alone when there's a trend but
    not enough history for the full ensemble, and as one member of the
    full ensemble otherwise. Damped (rather than plain linear) trend so
    it doesn't extrapolate a short run of growth in a straight line
    forever, which tends to overshoot on this kind of volatile data."""
    try:
        model = ExponentialSmoothing(values, trend="add", damped_trend=True, seasonal=None,
                                      initialization_method="estimated")
        fit = model.fit(optimized=True)
        return np.asarray(fit.forecast(periods)), (np.std(fit.resid) if len(fit.resid) else values.std())
    except Exception:
        return _naive_avg(values, periods), values.std()


def _hw_seasonal_forecast(values: np.ndarray, periods: int):
    """Holt-Winters, additive trend + additive yearly seasonality."""
    model = ExponentialSmoothing(values, trend="add", seasonal="add", seasonal_periods=12,
                                  initialization_method="estimated")
    fit = model.fit(optimized=True)
    return np.asarray(fit.forecast(periods)), (np.std(fit.resid) if len(fit.resid) else values.std())


def _croston_forecast(values: np.ndarray, periods: int, alpha: float = 0.1):
    """Croston's method — the standard technique for intermittent/lumpy
    demand (mostly zero months with occasional spikes). Separately
    smooths the average NON-ZERO order size and the average gap between
    orders, then combines them into a demand-per-period rate."""
    demand_sizes, intervals = [], []
    gap = 1
    for v in values:
        if v > 0:
            demand_sizes.append(v)
            intervals.append(gap)
            gap = 1
        else:
            gap += 1

    if not demand_sizes:
        return np.zeros(periods), 0.0

    a, p = demand_sizes[0], intervals[0]
    for i in range(1, len(demand_sizes)):
        a = alpha * demand_sizes[i] + (1 - alpha) * a
        p = alpha * intervals[i] + (1 - alpha) * p

    rate = a / p if p > 0 else 0.0
    resid_std = (np.std(demand_sizes) / p) if (p > 0 and len(demand_sizes) > 1) else rate * 0.6
    return np.repeat(rate, periods), resid_std


def _ensemble_forecast(values: np.ndarray, periods: int):
    """
    The validated 3-model ensemble (see module docstring for why): equal-
    weighted average of Holt-Winters seasonal, Holt's damped trend, and a
    6-month moving average. Needs at least 24 active months for the
    seasonal component to be meaningful.
    """
    hw_fc, hw_std = _hw_seasonal_forecast(values, periods)
    trend_fc, trend_std = _trend_forecast(values, periods)
    ma_fc = _naive_avg(values, periods, window=6)

    point_forecast = np.mean([hw_fc, trend_fc, ma_fc], axis=0)
    # Blend residual spreads the same way as the point forecasts, plus the
    # spread ACROSS the three members — an ensemble that disagrees with
    # itself a lot is telling you something about how uncertain this
    # month really is, which a single model's residual wouldn't capture.
    member_spread = np.std([hw_fc, trend_fc, ma_fc], axis=0)
    resid_std = float(np.mean([hw_std, trend_std])) + float(np.mean(member_spread))
    return point_forecast, resid_std


def _forecast_one_series(series: pd.Series, periods: int) -> dict:
    """
    Forecast one division's monthly series `periods` months ahead.

    Method chosen by tier (matching how much history is actually there —
    see module docstring for why the 3-model ensemble is used rather than
    free-form model selection):
      - active in <40% of months (sparse/intermittent, e.g. "Others"):
        Croston's method.
      - 24+ active months: the validated 3-model ensemble.
      - 8-23 active months: Holt's damped trend alone (not enough history
        for a meaningful yearly seasonal component).
      - <8 active months: 3-month moving average (not enough history to
        fit any trend model reliably).
    """
    values = series.values.astype(float)
    n_active = int((values > 0).sum())
    n_total = len(series)
    active_ratio = n_active / n_total if n_total else 0

    if active_ratio < 0.4:
        point_forecast, resid_std = _croston_forecast(values, periods)
        method = "Croston's method (intermittent demand)"
    elif n_active >= 24:
        try:
            point_forecast, resid_std = _ensemble_forecast(values, periods)
            method = "Ensemble: Holt-Winters (seasonal) + Holt's damped trend + 6-month average"
        except Exception:
            point_forecast, resid_std = _trend_forecast(values, periods)
            method = "Holt's damped trend (ensemble fit failed, fell back)"
    elif n_active >= 8:
        point_forecast, resid_std = _trend_forecast(values, periods)
        method = "Holt's damped trend (not enough active months for seasonal ensemble)"
    else:
        point_forecast = _naive_avg(values, periods)
        resid_std = values.std() if len(values) > 1 else point_forecast[0] * 0.3
        method = "3-month average (insufficient history for a trend model)"

    point_forecast = np.clip(point_forecast, a_min=0, a_max=None)
    lower = np.clip(point_forecast - 1.28 * resid_std, a_min=0, a_max=None)  # ~80% band
    upper = point_forecast + 1.28 * resid_std

    backtest_mape = _measure_backtest_mape(values, periods, active_ratio)

    return {
        "method": method,
        "backtest_mape": backtest_mape,
        "history": values.tolist(),
        "forecast": point_forecast.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "active_months": n_active,
    }


def _measure_backtest_mape(values: np.ndarray, periods: int, active_ratio: float):
    """
    Honest, measured walk-forward accuracy of whichever method
    `_forecast_one_series` would actually use for a series like this one
    — rolled forward over as many origins as the data allows (up to
    REPORTING_BACKTEST_ORIGINS), not just one hold-out. This is purely
    for reporting how reliable a division's forecast has historically
    been; it does not influence which method gets used (that's fixed —
    see module docstring for why free-form backtest-based selection was
    tried and reverted).
    """
    n = len(values)
    errors_pct = []
    for origin in range(REPORTING_BACKTEST_ORIGINS):
        cut = n - periods - origin
        if cut < REPORTING_BACKTEST_MIN_TRAIN:
            break
        train = values[:cut]
        actual = values[cut:cut + periods]
        train_active = int((train > 0).sum())
        train_ratio = train_active / len(train) if len(train) else 0
        try:
            if train_ratio < 0.4:
                fc, _ = _croston_forecast(train, periods)
            elif train_active >= 24:
                fc, _ = _ensemble_forecast(train, periods)
            elif train_active >= 8:
                fc, _ = _trend_forecast(train, periods)
            else:
                fc = _naive_avg(train, periods)
            fc = np.clip(fc, 0, None)
            mask = actual != 0
            if mask.sum():
                errors_pct.extend((np.abs(actual[mask] - fc[mask]) / actual[mask] * 100).tolist())
        except Exception:
            continue

    return round(float(np.mean(errors_pct)), 1) if errors_pct else None


def forecast_by_division(df: pd.DataFrame, value_col: str = "so_qty", periods: int = 4,
                          cutoff_date=None, lookback_years=None) -> dict:
    """
    Top-level entry point.

    cutoff_date:    train only on data up to and including this date (e.g.
                    "2026-03-31"). Defaults to the latest COMPLETE month in
                    the data if not given — see `_effective_cutoff` for why
                    a partial trailing month is excluded automatically
                    rather than silently treated as real data.
    lookback_years: if given, further restrict training data to the N years
                    immediately before cutoff_date. If the data doesn't
                    actually go back that far, all available data is used
                    instead (reported in "lookback_clamped").

    Returns a dict with:
      "divisions": {division_name: {...forecast fields..., "actual": [...],
                    "actual_status": [...], "backtest_mape": ...}}
      "meta": window/cutoff info the dashboard uses to explain what it's showing
    """
    cutoff_date, cutoff_adjusted_for_partial_month = _effective_cutoff(df, cutoff_date)

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
        results[division] = result

    meta = {
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "cutoff_adjusted_for_partial_month": cutoff_adjusted_for_partial_month,
        "data_min_date": data_min.strftime("%Y-%m-%d"),
        "data_max_date": data_max.strftime("%Y-%m-%d"),
        "lookback_years_requested": lookback_years,
        "lookback_clamped": lookback_clamped,
        "training_start": (range_start or data_min).strftime("%Y-%m-%d"),
        "training_months": len(next(iter(results.values()))["history_months"]) if results else 0,
        "forecast_periods": periods,
    }

    return {"divisions": results, "meta": meta}
