from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd
import pytz

from .kite_service import KiteService
from .utils import parse_chartink_csv, nearest_prior_timestamp, trading_days_ahead
from .types import BacktestInputRow



def compute_entry_exit_for_row(
    kite: KiteService,
    row: BacktestInputRow,
    num_days: int,
    exchange: str,
    tz: pytz.BaseTzInfo,
) -> dict:
    # Build localized entry timestamp
    entry_dt_local = tz.localize(dt.datetime.combine(row.entry_date, row.entry_time))

    # Get instrument token
    token = kite.resolve_instrument_token(symbol=row.stock, exchange=exchange)

    # 1h candles around entry date
    start = entry_dt_local - dt.timedelta(days=5)
    end = entry_dt_local + dt.timedelta(days=1)
    candles_1h = kite.fetch_ohlc(token=token, interval="60minute", start=start, end=end, tz=tz)
    if candles_1h.empty:
        raise ValueError(f"No 1h data for {row.stock} around {entry_dt_local}")

    # Find nearest earlier candle at or before entry time
    entry_ts = nearest_prior_timestamp(candles_1h.index, entry_dt_local)
    if entry_ts is None:
        raise ValueError(f"No prior candle for {row.stock} at {entry_dt_local}")
    entry_price = float(candles_1h.loc[entry_ts, "close"]) if "close" in candles_1h.columns else float(candles_1h.loc[entry_ts, "ohlc"]["close"])  # type: ignore

    # Determine exit date: N trading days after entry date close
    exit_trade_date = trading_days_ahead(entry_dt_local.date(), num_days)
    # Get daily candles to fetch close for exit date
    daily_start = tz.localize(dt.datetime.combine(entry_dt_local.date() - dt.timedelta(days=10), dt.time(9, 0)))
    daily_end = tz.localize(dt.datetime.combine(exit_trade_date + dt.timedelta(days=5), dt.time(15, 30)))
    daily = kite.fetch_ohlc(token=token, interval="day", start=daily_start, end=daily_end, tz=tz)
    if daily.empty:
        raise ValueError(f"No daily data for {row.stock}")

    # Resolve exit date by matching on calendar date and picking that day or next available
    daily_dates = pd.Index([ts.astimezone(tz).date() for ts in daily.index])
    # Find first index where date >= exit_trade_date
    candidate_positions = [i for i, d in enumerate(daily_dates) if d >= exit_trade_date]
    if not candidate_positions:
        exit_pos = len(daily) - 1
    else:
        exit_pos = candidate_positions[0]
    exit_ts = daily.index[exit_pos]
    exit_price = float(daily.iloc[exit_pos]["close"])  # type: ignore

    ret_pct = (exit_price - entry_price) / entry_price * 100.0

    return {
        "Stock": row.stock,
        "Entry Date": row.entry_date.isoformat(),
        "Entry Time": row.entry_time.strftime("%H:%M"),
        "Entry Price": round(entry_price, 4),
        "Exit Date": daily_dates[exit_pos].isoformat(),
        "Exit Price": round(exit_price, 4),
        "Return %": round(ret_pct, 4),
    }


def run_backtest_from_csv(
    csv_path: str,
    num_days: int,
    exchange: str,
    timezone_name: str,
) -> pd.DataFrame:
    tz = pytz.timezone(timezone_name)
    kite = KiteService.from_env()

    rows = parse_chartink_csv(csv_path=csv_path, tz=tz)
    results = []
    for r in rows:
        try:
            results.append(
                compute_entry_exit_for_row(
                    kite=kite,
                    row=r,
                    num_days=num_days,
                    exchange=exchange,
                    tz=tz,
                )
            )
        except Exception as e:
            results.append(
                {
                    "Stock": r.stock,
                    "Entry Date": r.entry_date.isoformat(),
                    "Entry Time": r.entry_time.strftime("%H:%M"),
                    "Entry Price": None,
                    "Exit Date": None,
                    "Exit Price": None,
                    "Return %": None,
                    "Error": str(e),
                }
            )

    df = pd.DataFrame(results)
    # Order columns
    preferred = ["Stock", "Entry Date", "Entry Time", "Entry Price", "Exit Date", "Exit Price", "Return %"]
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


