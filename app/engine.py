from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple, Dict

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
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
) -> dict:
    # Build localized entry timestamp
    entry_dt_local = tz.localize(dt.datetime.combine(row.entry_date, row.entry_time))

    # Get instrument token
    token = kite.resolve_instrument_token(symbol=row.stock, exchange=exchange)

    # 1h candles around entry date to price at/just before entry
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

    # Scheduled exit date after N trading days
    exit_trade_date = trading_days_ahead(entry_dt_local.date(), num_days)

    # Defaults
    exit_reason = "Time"
    exit_ts = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
    exit_price: Optional[float] = None
    exit_time_str = "15:30"

    # If SL/TP provided, scan intraday candles from entry onward
    if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0):
        target_price = entry_price * (1.0 + (tp_pct or 0.0) / 100.0)
        stop_price = entry_price * (1.0 - (sl_pct or 0.0) / 100.0)
        scan_start = entry_ts
        scan_end = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
        intraday = kite.fetch_ohlc(token=token, interval="15minute", start=scan_start, end=scan_end, tz=tz)
        if not intraday.empty:
            intraday = intraday[intraday.index > entry_ts]
            for ts, row_c in intraday.iterrows():
                high_v = float(row_c.get("high", row_c.get("close", 0.0)))
                low_v = float(row_c.get("low", row_c.get("close", 0.0)))
                # If both touched in same bar, assume SL first (conservative)
                if (tp_pct and target_price <= high_v) and (sl_pct and stop_price >= low_v):
                    exit_reason = "SL"
                    exit_price = stop_price
                    exit_ts = ts
                    break
                if tp_pct and target_price <= high_v:
                    exit_reason = "TP"
                    exit_price = target_price
                    exit_ts = ts
                    break
                if sl_pct and stop_price >= low_v:
                    exit_reason = "SL"
                    exit_price = stop_price
                    exit_ts = ts
                    break

    # If neither SL nor TP triggered, exit at close of the Nth trading day
    if exit_price is None:
        daily_start = tz.localize(dt.datetime.combine(entry_dt_local.date() - dt.timedelta(days=10), dt.time(9, 0)))
        daily_end = tz.localize(dt.datetime.combine(exit_trade_date + dt.timedelta(days=5), dt.time(15, 30)))
        daily = kite.fetch_ohlc(token=token, interval="day", start=daily_start, end=daily_end, tz=tz)
        if daily.empty:
            raise ValueError(f"No daily data for {row.stock}")
        daily_dates = pd.Index([ts.astimezone(tz).date() for ts in daily.index])
        candidate_positions = [i for i, d in enumerate(daily_dates) if d >= exit_trade_date]
        if not candidate_positions:
            exit_pos = len(daily) - 1
        else:
            exit_pos = candidate_positions[0]
        exit_ts = daily.index[exit_pos]
        exit_price = float(daily.iloc[exit_pos]["close"])  # type: ignore
        exit_ts_local = exit_ts.astimezone(tz)
        exit_time_str = "15:30" if (exit_ts_local.hour == 0 and exit_ts_local.minute == 0) else exit_ts_local.strftime("%H:%M")
    else:
        exit_time_str = exit_ts.astimezone(tz).strftime("%H:%M")

    ret_pct = (exit_price - entry_price) / entry_price * 100.0

    return {
        "Stock": row.stock,
        "Entry Date": row.entry_date.isoformat(),
        "Entry Time": row.entry_time.strftime("%H:%M"),
        "Entry Price": round(entry_price, 4),
        "Exit Date": exit_ts.astimezone(tz).date().isoformat(),
        "Exit Time": exit_time_str,
        "Exit Price": round(exit_price, 4),
        "Return %": round(ret_pct, 4),
        "Exit Reason": exit_reason,
    }


def run_backtest_from_csv(
    csv_path: str,
    num_days: int,
    exchange: str,
    timezone_name: str,
    sl_pct: Optional[float] | None = None,
    tp_pct: Optional[float] | None = None,
    allowed_entry_times: Optional[List[str]] | None = None,
    allowed_cap_buckets: Optional[List[str]] | None = None,
) -> pd.DataFrame:
    tz = pytz.timezone(timezone_name)
    kite = KiteService.from_env()

    rows = parse_chartink_csv(csv_path=csv_path, tz=tz)
    # Optional filter by allowed entry times (strings HH:MM)
    if allowed_entry_times:
        allowed_set = {t.strip() for t in allowed_entry_times if t}
        rows = [r for r in rows if r.entry_time.strftime("%H:%M") in allowed_set]
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
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
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
                    "Exit Time": None,
                    "Exit Price": None,
                    "Return %": None,
                    "Exit Reason": None,
                    "Error": str(e),
                }
            )

    df = pd.DataFrame(results)

    # Derive cap buckets from entry price tertiles
    if "Entry Price" in df.columns:
        prices_series = pd.to_numeric(df["Entry Price"], errors="coerce")
        if prices_series.notna().sum() >= 3:
            q1 = float(prices_series.quantile(1/3))
            q2 = float(prices_series.quantile(2/3))
            def bucket(p: float) -> str:
                if p <= q1:
                    return "Small"
                if p <= q2:
                    return "Mid"
                return "Large"
            df["Cap Bucket"] = prices_series.apply(lambda p: bucket(p) if pd.notna(p) else None)
            if allowed_cap_buckets:
                allowed_set = {b for b in allowed_cap_buckets if b}
                if allowed_set:
                    df = df[df["Cap Bucket"].isin(allowed_set)]

    preferred = ["Stock", "Entry Date", "Entry Time", "Entry Price", "Exit Date", "Exit Time", "Exit Price", "Return %", "Exit Reason"]
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


def compute_equity_and_stats(df: pd.DataFrame) -> Tuple[List[float], Dict[str, float]]:
    returns = [float(x) for x in df.get("Return %", []) if pd.notna(x)]
    equity: List[float] = []
    value = 100.0
    peak = value
    max_dd = 0.0
    for r in returns:
        value *= (1.0 + r / 100.0)
        equity.append(round(value, 4))
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    stats: Dict[str, float] = {
        "trades": float(len(returns)),
        "win_rate_pct": (len(wins) / len(returns) * 100.0) if returns else 0.0,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "risk_reward": rr,
        "total_return_pct": ((equity[-1] / 100.0 - 1.0) * 100.0) if equity else 0.0,
        "max_drawdown_pct": max_dd,
    }
    return equity, stats


def compute_insights(df: pd.DataFrame) -> Dict[str, list]:
    """Derive simple insights:
    - Top 5 entry times by TP hit rate (break ties by trades, then avg return)
    - Performance by cap bucket (Small/Mid/Large) approximated by entry price tertiles
    """
    insights: Dict[str, list] = {"top_entry_times": [], "cap_performance": []}

    # Filter valid rows
    valid = df.copy()
    if "Return %" in valid.columns:
        valid = valid[pd.to_numeric(valid["Return %"], errors="coerce").notna()]

    # Top entry times by TP hits
    if not valid.empty and "Entry Time" in valid.columns:
        grouped = []
        for entry_time, g in valid.groupby("Entry Time"):
            total = len(g)
            tp_hits = int((g.get("Exit Reason") == "TP").sum()) if "Exit Reason" in g.columns else 0
            hit_rate = (tp_hits / total * 100.0) if total > 0 else 0.0
            avg_ret = float(pd.to_numeric(g["Return %"], errors="coerce").mean()) if total > 0 else 0.0
            grouped.append({"time": str(entry_time), "trades": total, "tp_hits": tp_hits, "tp_hit_rate": round(hit_rate, 2), "avg_return": round(avg_ret, 2)})
        grouped.sort(key=lambda x: (x["tp_hit_rate"], x["trades"], x["avg_return"]), reverse=True)
        insights["top_entry_times"] = grouped[:5]

    # Cap performance: if precomputed buckets exist, use them; else approximate by price tertiles
    if not valid.empty:
        tmp = valid.copy()
        if "Cap Bucket" not in tmp.columns and "Entry Price" in tmp.columns:
            prices = pd.to_numeric(tmp["Entry Price"], errors="coerce").dropna()
            if len(prices) >= 3:
                q1 = float(prices.quantile(1/3))
                q2 = float(prices.quantile(2/3))
                def bucket(p: float) -> str:
                    if p <= q1:
                        return "Small"
                    if p <= q2:
                        return "Mid"
                    return "Large"
                tmp["Cap Bucket"] = pd.to_numeric(tmp["Entry Price"], errors="coerce").apply(lambda p: bucket(p) if pd.notna(p) else None)

        if "Cap Bucket" in tmp.columns:
            cap_rows = []
            for cap, g in tmp.groupby("Cap Bucket"):
                gret = pd.to_numeric(g.get("Return %"), errors="coerce").dropna()
                trades = int(len(gret))
                win_rate = (float((gret > 0).sum()) / trades * 100.0) if trades > 0 else 0.0
                avg_ret = float(gret.mean()) if trades > 0 else 0.0
                cap_rows.append({"cap": cap, "trades": trades, "win_rate": round(win_rate, 2), "avg_return": round(avg_ret, 2)})
            cap_rows.sort(key=lambda x: (x["avg_return"], x["win_rate"]), reverse=True)
            insights["cap_performance"] = cap_rows

    return insights


