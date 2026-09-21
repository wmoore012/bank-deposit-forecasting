"""Download a reproducible slice of FDIC quarterly bank financials.

Run once with ``.venv/bin/python download_fdic.py``. The notebook reads the
resulting CSV and never needs network access during grading.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
START_YEAR = 2010
END_YEAR = 2024
DESTINATION = ROOT / "data" / f"fdic_financials_{START_YEAR}_{END_YEAR}.csv"
FIELDS = ("REPDTE", "CERT", "NAME", "STALP", "DEPDOM", "ASSET", "LNLSNET", "CHBAL", "EQ")
QUARTERS = tuple(f"{year}{month_day}" for year in range(START_YEAR, END_YEAR + 1)
                 for month_day in ("0331", "0630", "0930", "1231"))
ENDPOINT = "https://api.fdic.gov/banks/financials"


def fetch_quarter(quarter: str) -> list[dict]:
    params = urlencode({
        "filters": f"REPDTE:{quarter}",
        "fields": ",".join(FIELDS),
        "limit": 10000,
        "offset": 0,
    })
    url = f"{ENDPOINT}?{params}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                payload = json.load(response)
            rows = [item["data"] for item in payload["data"]]
            expected = payload["meta"]["total"]
            if len(rows) != expected:
                raise RuntimeError(f"{quarter}: fetched {len(rows)} of {expected} rows")
            if any(row["REPDTE"] != quarter for row in rows):
                raise RuntimeError(f"{quarter}: response includes another quarter")
            print(f"{quarter}: {len(rows):,} banks", flush=True)
            return rows
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    rows = [row for quarter in QUARTERS for row in fetch_quarter(quarter)]
    keys = {(row["CERT"], row["REPDTE"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("Duplicate bank-quarter rows from FDIC")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)
    temporary.replace(DESTINATION)
    print(f"Saved {len(rows):,} rows to {DESTINATION} ({DESTINATION.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
