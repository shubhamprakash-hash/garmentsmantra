"""
test_api_connection.py
=======================
Run this once, standalone (only needs `requests`), to confirm:
  1. Whether the staging API wants SecurityCode/CompanyCode as HEADERS or
     as QUERY PARAMS (I couldn't test this myself — that domain isn't
     reachable from where this was written).
  2. The EXACT field names in a real response, so src/data_source.py's
     FIELD_ALIASES can be corrected if anything doesn't already match.

USAGE:
    pip install requests
    python test_api_connection.py

It prints which auth style worked, the HTTP status, and (on success) the
keys of the first record plus a total record count. Paste that output
back and I'll tune FIELD_ALIASES / GM_API_AUTH_LOCATION to match exactly.
"""

import requests

BASE_URL = "https://gm-api-staging.goldenbuzz.in"
ENDPOINT = "/api/GetForecastingOrderDetails"
SECURITY_CODE = "BppYwgD6PYi62DHre7G4RA"
COMPANY_CODE = "C004"

url = f"{BASE_URL}{ENDPOINT}"


def try_headers():
    print("\n--- Attempt 1: SecurityCode/CompanyCode as HEADERS ---")
    headers = {"SecurityCode": SECURITY_CODE, "CompanyCode": COMPANY_CODE}
    r = requests.get(url, headers=headers, timeout=30)
    print(f"Status: {r.status_code}")
    return r


def try_query():
    print("\n--- Attempt 2: SecurityCode/CompanyCode as QUERY PARAMS ---")
    params = {"SecurityCode": SECURITY_CODE, "CompanyCode": COMPANY_CODE}
    r = requests.get(url, params=params, timeout=30)
    print(f"Status: {r.status_code}")
    return r


def report_success(resp, mode):
    print(f"\n*** SUCCESS using {mode} auth (GM_API_AUTH_LOCATION={mode.lower()}) ***")
    data = resp.json()
    records = data if isinstance(data, list) else (
        data.get("data") or data.get("records") or data.get("result") or data.get("results") or []
    )
    print(f"Total records returned: {len(records)}")
    if records:
        print(f"Fields on first record: {list(records[0].keys())}")
        print(f"First record: {records[0]}")


resp = try_headers()
if resp.status_code == 200:
    report_success(resp, "HEADER")
else:
    print(f"Header attempt failed ({resp.status_code}). Body: {resp.text[:300]}")
    resp2 = try_query()
    if resp2.status_code == 200:
        report_success(resp2, "QUERY")
    else:
        print(f"Query attempt also failed ({resp2.status_code}). Body: {resp2.text[:300]}")
        print("\nBoth attempts failed — likely wrong header/param names (not just "
              "location), an expired staging credential, or a POST-only endpoint. "
              "Share this output and I'll adjust.")
