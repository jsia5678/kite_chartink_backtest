from __future__ import annotations

import datetime as dt
from typing import Iterable, List

import pandas as pd
import pytz

from .types import BacktestInputRow


def parse_chartink_csv(csv_path: str, tz: pytz.BaseTzInfo) -> List[BacktestInputRow]:
    df = pd.read_csv(csv_path)
    # Flexible mapping to support both separate date/time and combined datetime
    col_map = {
        "stock": None,       # Stock/Symbol/Tradingsymbol
        "entry_date": None,  # Entry Date/Date (can include time)
        "entry_time": None,  # Entry Time/Time (optional)
    }
    for c in df.columns:
        lc = c.strip().lower().replace(" ", "_")
        if lc in ("stock", "symbol", "tradingsymbol"):
            col_map["stock"] = c
        elif lc in ("entry_date", "date", "datetime"):
            col_map["entry_date"] = c
        elif lc in ("entry_time", "time"):
            col_map["entry_time"] = c

    # Must have stock and some form of date
    if col_map["stock"] is None or col_map["entry_date"] is None:
        missing = [k for k in ("stock", "entry_date") if col_map[k] is None]
        raise ValueError(f"Missing required columns in CSV: {missing}")

    rows: List[BacktestInputRow] = []
    for _, rec in df.iterrows():
        stock = str(rec[col_map["stock"]]).strip()
        date_raw = rec[col_map["entry_date"]]

        # If we have an explicit time column, parse separately
        if col_map["entry_time"] is not None:
            # Prefer ISO parsing first to avoid dayfirst warnings, then fall back
            s = str(date_raw).strip()
            date_val = None
            try:
                if len(s) == 10 and s[4] == '-' and s[7] == '-':
                    date_val = dt.datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                date_val = None
            if date_val is None:
                date_val = pd.to_datetime(s, dayfirst=True).date()
            time_str = str(rec[col_map["entry_time"]]).strip()
            # Handle HH:MM or HH:MM:SS
            try:
                t = dt.datetime.strptime(time_str, "%H:%M").time()
            except ValueError:
                try:
                    t = dt.datetime.strptime(time_str, "%H:%M:%S").time()
                except ValueError:
                    # Also try 12-hr clock with am/pm
                    t = dt.datetime.strptime(time_str, "%I:%M %p").time()
        else:
            # Combined datetime like "04-08-2025 10:15 am"
            dt_parsed = pd.to_datetime(str(date_raw), dayfirst=True, errors="coerce")
            if pd.isna(dt_parsed):
                # Try common explicit formats
                for fmt in ("%d-%m-%Y %I:%M %p", "%d/%m/%Y %I:%M %p", "%Y-%m-%d %H:%M:%S"):
                    try:
                        dt_parsed = dt.datetime.strptime(str(date_raw), fmt)
                        break
                    except Exception:
                        continue
            if pd.isna(dt_parsed):
                raise ValueError(f"Unparseable datetime: {date_raw}")
            if isinstance(dt_parsed, pd.Timestamp):
                date_val = dt_parsed.date()
                t = dt_parsed.time()
            else:
                date_val = dt_parsed.date()  # type: ignore
                t = dt_parsed.time()  # type: ignore

        rows.append(BacktestInputRow(stock=stock, entry_date=date_val, entry_time=t))

    # Keep only the earliest occurrence per symbol (by entry_date + entry_time)
    earliest_by_stock: dict[str, BacktestInputRow] = {}
    for r in rows:
        key = r.stock.strip().upper()
        current_dt = dt.datetime.combine(r.entry_date, r.entry_time)
        prev = earliest_by_stock.get(key)
        if prev is None:
            earliest_by_stock[key] = r
        else:
            prev_dt = dt.datetime.combine(prev.entry_date, prev.entry_time)
            if current_dt < prev_dt:
                earliest_by_stock[key] = r

    # Return in chronological order for deterministic processing
    rows_dedup = sorted(
        earliest_by_stock.values(),
        key=lambda r: dt.datetime.combine(r.entry_date, r.entry_time),
    )
    return rows_dedup


def nearest_prior_timestamp(index: pd.DatetimeIndex, target: dt.datetime):
    idx = index.sort_values()
    earlier = idx[idx <= target]
    if len(earlier) == 0:
        return None
    return earlier[-1]


def trading_days_ahead(start_date: dt.date, n: int) -> dt.date:
    # Simplified: assumes trading days are Mon-Fri, excluding weekends.
    # For Indian markets, this ignores exchange holidays. Could be enhanced.
    days_added = 0
    current = start_date
    while days_added < n:
        current += dt.timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            days_added += 1
    return current


