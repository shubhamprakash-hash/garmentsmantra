"""
data_source.py
================
Abstraction layer for getting sales/order data into the forecasting pipeline.

TODAY:   Reads from the "Order to Dispatch Status Report" export (CSV or Excel).
LATER:   Once the .NET team exposes GetSalesHistory / GetPendingSalesOrders,
         swap in APIDataSource below. Nothing in forecast_model.py or
         run_forecast.py needs to change — they only depend on the
         DataSource interface (get_sales_data), not on where it comes from.

To switch sources later, change ONE line in run_forecast.py:
    source = FileDataSource(path)      -->      source = APIDataSource(base_url, api_key)
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
    STUB — wire this up once the .NET team confirms GetSalesHistory /
    GetPendingSalesOrders. Kept here now so the swap later is a one-line
    change in run_forecast.py rather than a rewrite.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def get_sales_data(self) -> pd.DataFrame:
        import requests  # local import: only needed once this path is used
        headers = {"Authorization": f"Bearer {self.api_key}"}

        resp = requests.get(f"{self.base_url}/api/v1/GetSalesHistory", headers=headers)
        resp.raise_for_status()
        records = resp.json()

        df = pd.DataFrame(records)
        df["order_date"] = pd.to_datetime(df["orderDate"])
        df = df.rename(columns={
            "designNo": "design_no",
            "division": "division",
            "partyName": "party_name",
            "soQty": "so_qty",
            "dispatchQty": "dispatch_qty",
            "pendingQty": "pending_qty",
            "orderValue": "order_value",
        })
        return df[["order_date", "design_no", "division", "party_name",
                   "so_qty", "dispatch_qty", "pending_qty", "order_value"]]
