"""
forecast_model.py
==================
Phase-1 sales forecasting: division-level monthly demand forecast.

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

Method:
  With the updated dataset now covering ~3.5 years (43 months) for most
  divisions, there's enough history to estimate yearly seasonality, not
  just trend. Model choice per division, in order:
    1. Holt-Winters with additive trend + additive seasonality (period=12)
       when a division has 24+ active months AND orders in most months
       (a real, regular series).
    2. Croston's method when a division's orders are sparse/intermittent
       — active in less than 40% of months, e.g. "Others". A trend model
       fit on a mostly-zero series (like Others: 6 active months out of
       43) extrapolates nonsense; Croston's is built specifically for
       lumpy, irregular demand like this.
    3. Holt's linear trend (no seasonality) for series with a visible
       trend but under 24 active months and not intermittent.
    4. A 3-month moving average as a last-resort fallback for very short
       or degenerate series.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")


def build_monthly_series(df: pd.DataFrame, value_col: str = "so_qty") -> pd.DataFrame:
    """
    Aggregate raw order rows into a division x month demand table, with
    every calendar month present (missing months filled with 0). Gap-filling
    matters here because a seasonal model needs an evenly spaced series —
    a division with no orders in a given month is a real data point (zero
    demand that month), not a gap to skip over.

    The most recent calendar month is dropped if the data doesn't actually
    reach that month's last day (e.g. an export taken mid-month) — an
    incomplete month looks like a demand crash to the model and to any
    backtest, when it's really just a partial data pull.
    """
    max_date = df["order_date"].max()
    month_end = max_date + pd.offsets.MonthEnd(0)
    trimmed = False
    if max_date < month_end:
        cutoff = max_date.to_period("M").to_timestamp()
        df = df[df["order_date"].dt.to_period("M").dt.to_timestamp() < cutoff]
        trimmed = True

    monthly = (
        df.assign(month=df["order_date"].dt.to_period("M").dt.to_timestamp())
          .groupby(["division", "month"])[value_col]
          .sum()
          .reset_index()
    )

    full_range = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    filled = []
    for division, grp in monthly.groupby("division"):
        s = grp.set_index("month")[value_col].reindex(full_range, fill_value=0)
        filled.append(pd.DataFrame({"division": division, "month": full_range, value_col: s.values}))

    result = pd.concat(filled, ignore_index=True)
    if trimmed:
        print(f"Note: dropped incomplete trailing month (data only ran to {max_date.date()}) "
              f"before fitting/backtesting.")
    return result


def _forecast_one_series(series: pd.Series, periods: int = 3) -> dict:
    """
    Fit the best available model on a single division's monthly series and
    forecast `periods` months ahead. Returns point forecast + a simple
    uncertainty band derived from in-sample residual volatility (not a full
    statistical prediction interval — good enough for a phase-1 planning
    number).

    Model choice is based on how many months actually have orders in them
    (n_active) and how sparse the series is (active_ratio), not just the
    calendar span. A division can span 43 months on the calendar but only
    have real activity in 6 of them (e.g. "Others") — fitting a trend or
    seasonal model on a mostly-zero-padded series extrapolates noise, not
    signal, so sparse series get routed to Croston's method instead.
    """
    n_active = int((series.values > 0).sum())
    n_total = len(series)
    active_ratio = n_active / n_total if n_total else 0
    values = series.values.astype(float)

    if active_ratio < 0.4:
        point_forecast, resid_std, method = _croston_forecast(values, periods)
    elif n_active >= 24:
        try:
            model = ExponentialSmoothing(values, trend="add", seasonal="add",
                                          seasonal_periods=12,
                                          initialization_method="estimated")
            fit = model.fit(optimized=True)
            point_forecast = fit.forecast(periods)
            resid_std = np.std(fit.resid) if len(fit.resid) else values.std()
            method = "Holt-Winters (trend + yearly seasonality)"
        except Exception:
            point_forecast, resid_std, method = _trend_forecast(values, periods)
    elif n_active >= 4:
        point_forecast, resid_std, method = _trend_forecast(values, periods)
    else:
        point_forecast, resid_std, method = _naive_forecast(values, periods)

    point_forecast = np.clip(point_forecast, a_min=0, a_max=None)
    lower = np.clip(point_forecast - 1.28 * resid_std, a_min=0, a_max=None)  # ~80% band
    upper = point_forecast + 1.28 * resid_std

    return {
        "method": method,
        "history": values.tolist(),
        "forecast": point_forecast.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "active_months": n_active,
    }


def _croston_forecast(values: np.ndarray, periods: int, alpha: float = 0.1):
    """
    Croston's method — the standard technique for intermittent/lumpy demand
    (mostly zero months with occasional spikes). Separately smooths the
    average NON-ZERO order size and the average gap between orders, then
    combines them into a demand-per-period rate. This is what "Others"
    needs instead of a trend model, which has no concept of "mostly zero."
    """
    demand_sizes = []
    intervals = []
    gap = 1
    for v in values:
        if v > 0:
            demand_sizes.append(v)
            intervals.append(gap)
            gap = 1
        else:
            gap += 1

    if not demand_sizes:
        return np.zeros(periods), 0.0, "Croston's method (no order history yet)"

    a = demand_sizes[0]
    p = intervals[0]
    for i in range(1, len(demand_sizes)):
        a = alpha * demand_sizes[i] + (1 - alpha) * a
        p = alpha * intervals[i] + (1 - alpha) * p

    rate_per_month = a / p if p > 0 else 0.0
    resid_std = (np.std(demand_sizes) / p) if (p > 0 and len(demand_sizes) > 1) else rate_per_month * 0.6

    return np.repeat(rate_per_month, periods), resid_std, "Croston's method (intermittent demand)"


def _trend_forecast(values: np.ndarray, periods: int):
    """Holt's linear trend (no seasonality) — used when there's a trend but
    fewer than 24 months of history (not enough for 2 full yearly cycles)."""
    try:
        model = ExponentialSmoothing(values, trend="add", seasonal=None,
                                      initialization_method="estimated")
        fit = model.fit(optimized=True)
        point_forecast = fit.forecast(periods)
        resid_std = np.std(fit.resid) if len(fit.resid) else values.std()
        return point_forecast, resid_std, "Holt's linear trend"
    except Exception:
        return _naive_forecast(values, periods)


def _naive_forecast(values: np.ndarray, periods: int):
    """Fallback for very short series: last-3-month average, flat-projected."""
    window = values[-3:] if len(values) >= 3 else values
    avg = window.mean() if len(window) else 0.0
    resid_std = window.std() if len(window) > 1 else avg * 0.2
    return np.repeat(avg, periods), resid_std, "3-month average (insufficient history for trend model)"


def forecast_by_division(df: pd.DataFrame, value_col: str = "so_qty", periods: int = 3) -> dict:
    """
    Top-level entry point. Returns a dict keyed by division, each containing
    the month labels (history + forecast) and the series computed above.
    """
    monthly = build_monthly_series(df, value_col=value_col)
    results = {}

    for division, grp in monthly.groupby("division"):
        grp = grp.sort_values("month")
        series = grp.set_index("month")[value_col]

        result = _forecast_one_series(series, periods=periods)
        history_months = [d.strftime("%b %Y") for d in series.index]

        last_month = series.index.max()
        future_months = pd.date_range(last_month, periods=periods + 1, freq="MS")[1:]
        forecast_months = [d.strftime("%b %Y") for d in future_months]

        result["history_months"] = history_months
        result["forecast_months"] = forecast_months
        result["total_history_orders"] = int(len(df[df["division"] == division]))
        results[division] = result

    return results
