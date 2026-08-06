"""
data_source.py
================
Abstraction layer for getting sales/order data into the forecasting pipeline.

FileDataSource: reads the "Order to Dispatch Status Report" export (CSV/Excel).
                Kept as a local/offline fallback and for re-running old exports.
APIDataSource:  pulls live order history from the .NET-exposed endpoint
                (e.g. GetSalesHistory). This is what app.py uses in production
                once GM_API_BASE_URL is set — see APIDataSource's docstring
                for the env vars that configure it.

Neither forecast_model.py nor run_forecast.py needs to change to switch
sources — they only depend on the DataSource interface (get_sales_data).
"""

from __future__ import annotations
import os
import pandas as pd
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Common interface every data source must implement."""

    @abstractmethod
    def get_sales_data(self) -> pd.DataFrame:
        """
        Must return a DataFrame with (at minimum) these columns:
            order_date   : datetime64
            design_no    : str
            division     : str
            party_name   : str
            so_qty       : float   (order booking quantity)
            dispatch_qty : float   (actually dispatched — closest proxy to 'sold' we have today)
            pending_qty  : float
            order_value  : float
        """
        raise NotImplementedError


class FileDataSource(DataSource):
    """
    Reads the 'Order to Dispatch Status Report' export — CSV or Excel (.xlsx).
    File type is auto-detected from the extension.

    NOTE: this is an interim source. SO Qty / Dispatch Qty are order-booking
    and production-dispatch figures, not a true sell-through/POS signal.
    They're used here as the best available proxy for demand until
    GetSalesHistory is available from the .NET team (see data_source
    docstring above).
    """

    def __init__(self, path: str):
        self.path = path

    NEEDED_COLUMNS = [
        "Order Date", "Design No", "DivisionName", "Party Name",
        "SO Qty", "Dispatch Qty", "Pending Qty", "Order Value", "Sale Value",
    ]

    def _cache_path(self) -> str:
        base, _ = os.path.splitext(self.path)
        return base + ".parquet"

    def _read_raw(self) -> pd.DataFrame:
        cache_path = self._cache_path()

        # If a cached parquet copy exists, use it — this is what makes cold
        # starts (e.g. on Render, after the free tier spins down, or on a
        # fresh deploy) fast instead of a 20-30s Excel re-parse every time.
        # After the underlying data file changes, delete the .parquet file
        # (or call POST /forecast/v1/refresh, which does this for you) to
        # force a rebuild — mtime isn't used for validity since git checkouts
        # don't reliably preserve original file modification times.
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        if self.path.lower().endswith((".xlsx", ".xls")):
            # The export has 100+ columns (production-stage tracking etc.)
            # that this model doesn't use. Reading only the needed columns
            # cuts load time drastically on a large export like this one.
            raw = pd.read_excel(self.path, header=1, usecols=self.NEEDED_COLUMNS)
        else:
            raw = pd.read_csv(self.path, header=1, usecols=self.NEEDED_COLUMNS)

        try:
            raw.to_parquet(cache_path, index=False)
        except Exception:
            pass  # caching is an optimization, not a requirement — fine if it fails
        return raw

    def get_sales_data(self) -> pd.DataFrame:
        # Row 0 of this export is a report title, real headers are on row 1
        raw = self._read_raw()

        df = pd.DataFrame({
            "order_date":   pd.to_datetime(raw["Order Date"], format="%d/%m/%y", errors="coerce"),
            "design_no":    raw["Design No"].astype(str),
            "division":     raw["DivisionName"].astype(str).str.strip(),
            "party_name":   raw["Party Name"].astype(str),
            "so_qty":       pd.to_numeric(raw["SO Qty"], errors="coerce").fillna(0),
            "dispatch_qty": pd.to_numeric(raw["Dispatch Qty"], errors="coerce").fillna(0),
            "pending_qty":  pd.to_numeric(raw["Pending Qty"], errors="coerce").fillna(0)
                             if "Pending Qty" in raw.columns
                             else (pd.to_numeric(raw["SO Qty"], errors="coerce").fillna(0)
                                   - pd.to_numeric(raw["Dispatch Qty"], errors="coerce").fillna(0)).clip(lower=0),
            "order_value":  pd.to_numeric(raw["Order Value"], errors="coerce").fillna(0),
            "sale_value":   pd.to_numeric(raw["Sale Value"], errors="coerce").fillna(0)
                             if "Sale Value" in raw.columns else 0,
        })

        df = df.dropna(subset=["order_date"])
        df = df[df["division"] != "nan"]
        return df.reset_index(drop=True)


# Backward-compatible alias — earlier version of this script only handled CSV.
CSVDataSource = FileDataSource


class APIDataSource(DataSource):
    """
    Live data source — pulls order history from the .NET-exposed endpoint
    (GetForecastingOrderDetails) instead of the Excel export.

    CONFIG (all via constructor args or env vars — env vars let app.py pick
    this source up with zero code changes once the .NET team hands over
    real values):
        base_url      / GM_API_BASE_URL       e.g. https://gm-api-staging.goldenbuzz.in
        endpoint      / GM_API_ENDPOINT       default: /api/GetForecastingOrderDetails
        auth_mode     / GM_API_AUTH_MODE      "security_code" (default) | "api_key" | "bearer" | "none"
        auth_location / GM_API_AUTH_LOCATION  "header" (default) | "query" — where the
                                               credentials get sent; switch to "query" if
                                               the API responds 401 to the header form.

        --- security_code mode (the current staging API) ---
        security_code / GM_API_SECURITY_CODE
        company_code  / GM_API_COMPANY_CODE
        Sent as headers "SecurityCode" / "CompanyCode" by default, or as query
        string params of the same names if auth_location="query".

        --- api_key / bearer mode (kept for a possible future/different API) ---
        api_key / GM_API_KEY

    FIELD MAPPING: expects the response to be a JSON array of objects (or a
    JSON object with a "data"/"records"/"result" list inside it — all three
    are auto-detected). Field names are matched case- and
    underscore/camelCase-insensitively against:
        orderDate, designNo, division/divisionName, partyName,
        soQty, dispatchQty, pendingQty, orderValue, saleValue
    so small naming differences from the .NET side (e.g. "OrderDate" vs
    "orderDate" vs "order_date") don't require a code change — only a
    genuinely different field NAME (e.g. "qty" instead of "soQty") would.
    If the real response uses different names, add them to FIELD_ALIASES
    below rather than changing the rest of the pipeline.
    """

    DEFAULT_ENDPOINT = "/api/GetForecastingOrderDetails"

    # maps our internal column -> list of acceptable source field names
    # (checked case-insensitively, ignoring underscores/spaces)
    FIELD_ALIASES = {
        "order_date":   ["orderdate", "order_date", "date"],
        "design_no":    ["designno", "design_no", "designnumber"],
        "division":     ["division", "divisionname", "division_name"],
        "party_name":   ["partyname", "party_name", "customername", "party"],
        "so_qty":       ["soqty", "so_qty", "orderqty", "orderqty1"],
        "dispatch_qty": ["dispatchqty", "dispatch_qty"],
        "pending_qty":  ["pendingqty", "pending_qty"],
        "order_value":  ["ordervalue", "order_value"],
        "sale_value":   ["salevalue", "sale_value", "dispatchvalue", "dispatch_value"],
    }

    def __init__(self, base_url: str = None, endpoint: str = None,
                 auth_mode: str = None, auth_location: str = None,
                 security_code: str = None, company_code: str = None,
                 api_key: str = None, timeout: int = 30):
        self.base_url = (base_url or os.environ.get("GM_API_BASE_URL", "")).rstrip("/")
        self.endpoint = endpoint or os.environ.get("GM_API_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.auth_mode = (auth_mode or os.environ.get("GM_API_AUTH_MODE", "security_code")).lower()
        self.auth_location = (auth_location or os.environ.get("GM_API_AUTH_LOCATION", "header")).lower()
        self.security_code = security_code or os.environ.get("GM_API_SECURITY_CODE", "")
        self.company_code = company_code or os.environ.get("GM_API_COMPANY_CODE", "")
        self.api_key = api_key or os.environ.get("GM_API_KEY", "")
        self.timeout = timeout

        if not self.base_url:
            raise ValueError(
                "APIDataSource needs a base_url (or GM_API_BASE_URL env var) — "
                "the host of the live sales-data API, e.g. https://gm-api-staging.goldenbuzz.in"
            )

    def _auth_params(self) -> tuple[dict, dict]:
        """Returns (headers, query_params) for the credential scheme in use."""
        if self.auth_mode == "none":
            return {}, {}

        if self.auth_mode == "bearer":
            if not self.api_key:
                raise ValueError("auth_mode='bearer' needs api_key / GM_API_KEY.")
            return {"Authorization": f"Bearer {self.api_key}"}, {}

        if self.auth_mode == "api_key":
            if not self.api_key:
                raise ValueError("auth_mode='api_key' needs api_key / GM_API_KEY.")
            return {"X-API-Key": self.api_key}, {}

        # default: security_code — SecurityCode + CompanyCode, as header or query param
        if not self.security_code or not self.company_code:
            raise ValueError(
                "auth_mode='security_code' needs both security_code/GM_API_SECURITY_CODE "
                "and company_code/GM_API_COMPANY_CODE."
            )
        creds = {"SecurityCode": self.security_code, "CompanyCode": self.company_code}
        if self.auth_location == "query":
            return {}, creds
        return creds, {}

    @staticmethod
    def _normalize_key(k: str) -> str:
        return str(k).lower().replace("_", "").replace(" ", "")

    def _extract_records(self, payload):
        """Handle the common response shapes: a bare list, or an object
        wrapping the list under a conventional key."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "records", "result", "results", "items"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        raise ValueError(
            f"Unexpected response shape from {self.endpoint}: expected a JSON "
            f"array, or an object with a 'data'/'records'/'result' list inside it. "
            f"Got: {type(payload).__name__}"
        )

    def _map_record(self, record: dict) -> dict:
        normalized = {self._normalize_key(k): v for k, v in record.items()}
        mapped = {}
        for target, aliases in self.FIELD_ALIASES.items():
            value = None
            for alias in aliases:
                if alias in normalized:
                    value = normalized[alias]
                    break
            mapped[target] = value
        return mapped

    def _request(self, url: str):
        """
        Tries the configured auth_location first; if that gets a 401 and
        auth_mode is security_code, automatically retries with the other
        location (header <-> query) rather than requiring a human to have
        pre-verified which one the API expects. Remembers whichever worked
        so a later refresh doesn't re-probe.
        """
        import requests

        headers, params = self._auth_params()
        resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)

        if resp.status_code == 401 and self.auth_mode == "security_code":
            fallback_location = "query" if self.auth_location == "header" else "header"
            alt_headers, alt_params = ({}, {"SecurityCode": self.security_code, "CompanyCode": self.company_code}) \
                if fallback_location == "query" \
                else ({"SecurityCode": self.security_code, "CompanyCode": self.company_code}, {})
            alt_resp = requests.get(url, headers=alt_headers, params=alt_params, timeout=self.timeout)
            if alt_resp.status_code == 200:
                self.auth_location = fallback_location  # remember for next call
                return alt_resp

        return resp

    def get_sales_data(self) -> pd.DataFrame:
        url = f"{self.base_url}{self.endpoint}"
        resp = self._request(url)
        resp.raise_for_status()
        records = self._extract_records(resp.json())

        if not records:
            raise ValueError(f"{url} returned zero records — check the source system has data.")

        mapped_rows = [self._map_record(r) for r in records]
        raw = pd.DataFrame(mapped_rows)

        # order_date/so_qty being entirely NaN means the field alias didn't
        # match anything in the response (a mapping bug). division is
        # excluded from this check — it's legitimately null for some real
        # orders (mapped to "Unassigned" below), so an all-null division
        # column isn't necessarily a mapping bug.
        missing_required = [
            col for col in ("order_date", "so_qty")
            if raw[col].isna().all()
        ]
        if missing_required:
            sample_keys = list(records[0].keys())
            raise ValueError(
                f"Could not find a matching field for {missing_required} in the API "
                f"response. Fields present in the response were: {sample_keys}. "
                f"Update APIDataSource.FIELD_ALIASES in src/data_source.py to add "
                f"the real field name(s)."
            )

        # Null/blank division -> "Unassigned" rather than silently dropping the
        # order. Real data from GetForecastingOrderDetails does return null
        # divisions for some orders; excluding them would mean the forecast
        # (and the dashboard's totals) quietly undercounts actual order volume.
        # If it turns out Unassigned is a large share of the total, that's a
        # signal to go back to the .NET team about why division isn't set,
        # rather than something to paper over silently.
        division_raw = raw["division"]
        is_blank = division_raw.isna() | division_raw.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
        division_clean = division_raw.astype(str).str.strip()
        division_clean = division_clean.mask(is_blank, "Unassigned")

        df = pd.DataFrame({
            "order_date":   pd.to_datetime(raw["order_date"], errors="coerce"),
            "design_no":    raw["design_no"].astype(str),
            "division":     division_clean,
            "party_name":   raw["party_name"].astype(str),
            "so_qty":       pd.to_numeric(raw["so_qty"], errors="coerce").fillna(0),
            "dispatch_qty": pd.to_numeric(raw["dispatch_qty"], errors="coerce").fillna(0),
            "pending_qty":  pd.to_numeric(raw["pending_qty"], errors="coerce").fillna(0)
                             if raw["pending_qty"].notna().any()
                             else (pd.to_numeric(raw["so_qty"], errors="coerce").fillna(0)
                                   - pd.to_numeric(raw["dispatch_qty"], errors="coerce").fillna(0)).clip(lower=0),
            "order_value":  pd.to_numeric(raw["order_value"], errors="coerce").fillna(0),
            "sale_value":   pd.to_numeric(raw["sale_value"], errors="coerce").fillna(0)
                             if raw["sale_value"].notna().any() else 0,
        })

        df = df.dropna(subset=["order_date"])
        return df.reset_index(drop=True)
