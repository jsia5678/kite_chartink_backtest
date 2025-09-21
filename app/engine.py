from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple, Dict

import pandas as pd
import pytz

from .kite_service import KiteService
from .utils import parse_chartink_csv, nearest_prior_timestamp, trading_days_ahead
from .types import BacktestInputRow
from .trade_audit import TradeAuditor



def compute_entry_exit_for_row(
    kite: KiteService,
    row: BacktestInputRow,
    num_days: int,
    exchange: str,
    tz: pytz.BaseTzInfo,
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    breakeven_profit_pct: Optional[float] = None,
    breakeven_at_sl: bool = False,
    intraday_interval: str = "15minute",
) -> dict:
    # Build localized entry timestamp
    entry_dt_local = tz.localize(dt.datetime.combine(row.entry_date, row.entry_time))
    
    # Detect if this is a BTST trade based on entry time
    is_btst = row.entry_time >= dt.time(15, 15) and row.entry_time <= dt.time(15, 30)

    # Get instrument token
    token = kite.resolve_instrument_token(symbol=row.stock, exchange=exchange)

    # Intraday candles around entry date to price at/just before entry
    start = entry_dt_local - dt.timedelta(days=5)
    end = entry_dt_local + dt.timedelta(days=1)
    candles_intraday = kite.fetch_ohlc(token=token, interval=intraday_interval, start=start, end=end, tz=tz)
    if candles_intraday.empty:
        raise ValueError(f"No {intraday_interval} data for {row.stock} around {entry_dt_local}")

    # Find nearest earlier candle at or before entry time
    entry_ts = nearest_prior_timestamp(candles_intraday.index, entry_dt_local)
    if entry_ts is None:
        raise ValueError(f"No prior candle for {row.stock} at {entry_dt_local}")
    entry_price = float(candles_intraday.loc[entry_ts, "close"]) if "close" in candles_intraday.columns else float(candles_intraday.loc[entry_ts, "ohlc"]["close"])  # type: ignore

    # Scheduled exit date after N trading days
    exit_trade_date = trading_days_ahead(entry_dt_local.date(), num_days)

    # Defaults - adjust for BTST trades
    if is_btst:
        # BTST: Exit next day at market open
        exit_trade_date = trading_days_ahead(entry_dt_local.date(), 1)
        exit_ts = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(9, 15)))
        exit_time_str = "09:15"
        exit_reason = "BTST_Open"
    else:
        # Regular trades: Exit at market close
        exit_ts = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
        exit_time_str = "15:30"
        exit_reason = "Time"
    
    exit_price: Optional[float] = None

    # If SL/TP provided, scan intraday candles from entry onward using intraday_interval CLOSES
    if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0) or (breakeven_profit_pct is not None) or breakeven_at_sl:
        target_price = entry_price * (1.0 + (tp_pct or 0.0) / 100.0) if tp_pct else None
        base_stop_price = entry_price * (1.0 - (sl_pct or 0.0) / 100.0) if sl_pct else None
        current_stop = base_stop_price
        breakeven_set = False

        scan_start = entry_ts
        scan_end = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
        intraday = kite.fetch_ohlc(token=token, interval=intraday_interval, start=scan_start, end=scan_end, tz=tz)
        if not intraday.empty:
            intraday = intraday[intraday.index > entry_ts]
            for ts, row_c in intraday.iterrows():
                close_v = float(row_c.get("close", row_c.get("ohlc", {}).get("close", 0.0)))  # type: ignore
                # 1) TP check (at close)
                if target_price is not None and close_v >= target_price:
                    exit_reason = "TP"
                    exit_price = close_v
                    exit_ts = ts
                    break

                # 2) Breakeven logic: either fixed profit % or when profit >= SL%
                activate_be = False
                if breakeven_profit_pct is not None:
                    if close_v >= entry_price * (1.0 + breakeven_profit_pct / 100.0):
                        activate_be = True
                if not activate_be and breakeven_at_sl and sl_pct is not None and sl_pct > 0:
                    if close_v >= entry_price * (1.0 + sl_pct / 100.0):
                        activate_be = True
                if activate_be and not breakeven_set:
                    current_stop = max(current_stop or -1e18, entry_price)
                    breakeven_set = True

                # 3) Baseline SL check (at close) using current_stop
                if current_stop is not None and close_v <= current_stop:
                    exit_reason = "SL"
                    exit_price = close_v
                    exit_ts = ts
                    break

    # If neither SL nor TP triggered, handle exit based on strategy type
    if exit_price is None:
        if is_btst:
            # BTST: Exit at next day open with gap-up logic
            day_start = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(9, 0)))
            day_end = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(9, 30)))
            intraday_exit = kite.fetch_ohlc(token=token, interval=intraday_interval, start=day_start, end=day_end, tz=tz)
            
            if not intraday_exit.empty:
                # Get opening price (first candle of the day)
                open_ts = intraday_exit.index[0]
                open_price = float(intraday_exit.iloc[0]["open"])  # type: ignore
                
                # Check for gap-up or gap-down (0.5% threshold)
                gap_up_threshold = entry_price * 1.005
                gap_down_threshold = entry_price * 0.995
                
                if open_price > gap_up_threshold:
                    # Gap-up: Exit at open with profit
                    exit_ts = open_ts
                    exit_price = open_price
                    exit_reason = "BTST_GapUp"
                    exit_time_str = "09:15"
                elif open_price < gap_down_threshold:
                    # Gap-down: Exit at open with loss (cut losses quickly)
                    exit_ts = open_ts
                    exit_price = open_price
                    exit_reason = "BTST_GapDown"
                    exit_time_str = "09:15"
                else:
                    # No significant gap, sell at entry price
                    exit_ts = open_ts
                    exit_price = entry_price
                    exit_reason = "BTST_NoGap"
                    exit_time_str = "09:25"
            else:
                # Fallback for BTST
                exit_price = entry_price
                exit_reason = "BTST_NoGap"
                exit_time_str = "09:25"
        else:
            # Regular trades: Exit at market close
            day_start = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(9, 0)))
            day_end = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
            intraday_exit = kite.fetch_ohlc(token=token, interval=intraday_interval, start=day_start, end=day_end, tz=tz)
            if not intraday_exit.empty:
                exit_ts = intraday_exit.index[-1]
                exit_price = float(intraday_exit.iloc[-1]["close"])  # type: ignore
                exit_time_str = exit_ts.astimezone(tz).strftime("%H:%M")
            else:
                # Fallback to daily close if no intraday data available
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

    # Calculate return percentage with error handling
    if entry_price and exit_price and entry_price > 0:
        ret_pct = (exit_price - entry_price) / entry_price * 100.0
    else:
        ret_pct = 0.0

    return {
        "Stock": row.stock,
        "Entry Date": row.entry_date.isoformat(),
        "Entry Time": row.entry_time.strftime("%H:%M"),
        "Entry Price": round(entry_price, 4) if entry_price else 0.0,
        "Exit Date": exit_ts.astimezone(tz).date().isoformat(),
        "Exit Time": exit_time_str,
        "Exit Price": round(exit_price, 4) if exit_price else 0.0,
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
    symbol_cap_csv_path: Optional[str] | None = None,
    breakeven_profit_pct: Optional[float] | None = None,
    breakeven_at_sl: bool = False,
    enable_audit: bool = True,
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
                    breakeven_profit_pct=breakeven_profit_pct,
                    breakeven_at_sl=breakeven_at_sl,
                )
            )
        except Exception as e:
            results.append(
                {
                    "Stock": r.stock,
                    "Entry Date": r.entry_date.isoformat(),
                    "Entry Time": r.entry_time.strftime("%H:%M"),
                    "Entry Price": 0.0,
                    "Exit Date": r.entry_date.isoformat(),
                    "Exit Time": "15:30",
                    "Exit Price": 0.0,
                    "Return %": 0.0,
                    "Exit Reason": "Error",
                    "Error": str(e),
                }
            )

    df = pd.DataFrame(results)

    # Derive cap buckets: prefer provided metadata CSV; otherwise fallback to entry-price tertiles proxy
    def normalize_symbol(s: str) -> str:
        u = s.strip().upper()
        if u.endswith(".NS"):
            u = u[:-3]
        return u

    cap_series: Optional[pd.Series] = None
    if symbol_cap_csv_path:
        try:
            meta = pd.read_csv(symbol_cap_csv_path)
            # Expect columns: symbol, market_cap
            cols = {c.lower(): c for c in meta.columns}
            sym_col = cols.get("symbol") or cols.get("tradingsymbol") or list(meta.columns)[0]
            mc_col = cols.get("market_cap") or cols.get("marketcap") or list(meta.columns)[1]
            meta = meta[[sym_col, mc_col]].dropna()
            meta[sym_col] = meta[sym_col].astype(str).map(normalize_symbol)
            meta[mc_col] = pd.to_numeric(meta[mc_col], errors="coerce")
            meta = meta.dropna()
            m = meta.set_index(sym_col)[mc_col]
            df_symbols = df.get("Stock").astype(str).map(normalize_symbol)
            cap_series = df_symbols.map(m)
        except Exception:
            cap_series = None

    if cap_series is not None and cap_series.notna().sum() >= 3:
        q1 = float(cap_series.quantile(1/3))
        q2 = float(cap_series.quantile(2/3))
        def cap_bucket(mc: float) -> str:
            if mc <= q1:
                return "Small"
            if mc <= q2:
                return "Mid"
                
            return "Large"
        df["Cap Bucket"] = cap_series.apply(lambda mc: cap_bucket(mc) if pd.notna(mc) else None)
    elif "Entry Price" in df.columns:
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

    if allowed_cap_buckets and "Cap Bucket" in df.columns:
        allowed_set = {b for b in allowed_cap_buckets if b}
        if allowed_set:
            df = df[df["Cap Bucket"].isin(allowed_set)]

    # Apply trade audit if enabled
    if enable_audit:
        try:
            auditor = TradeAuditor(timezone=timezone_name)
            df_audited, audit_summary = auditor.audit_trade_log(df, num_days)
            
            # Add audit summary as metadata (stored in df attributes)
            df_audited.attrs['audit_summary'] = audit_summary
            
            # Use audited dataframe
            df = df_audited
        except Exception as e:
            # If audit fails, continue with original dataframe but log the error
            print(f"Trade audit failed: {e}")
    
    preferred = ["Stock", "Entry Date", "Entry Time", "Entry Price", "Exit Date", "Exit Time", "Exit Price", "Return %", "Exit Reason"]
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]



def compute_equity_and_stats(df: pd.DataFrame) -> Tuple[List[float], Dict[str, float]]:
    returns = [float(x) for x in df.get("Return %", []) if pd.notna(x)]
    # Vectorized equity curve
    if returns:
        import numpy as np
        equity_arr = 100.0 * np.cumprod(1.0 + np.array(returns, dtype=float) / 100.0)
        equity = [round(float(x), 4) for x in equity_arr]
        # Drawdown
        running_peak = np.maximum.accumulate(equity_arr)
        dd_series = (running_peak - equity_arr) / running_peak * 100.0
        max_dd = float(dd_series.max()) if dd_series.size else 0.0
    else:
        equity = []
        max_dd = 0.0
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    # Expectancy per trade (average return per trade)
    expectancy = (sum(returns) / len(returns)) if returns else 0.0

    # CAGR and Calmar Ratio
    cagr_pct = 0.0
    try:
        if not df.empty and equity:
            start_dates = pd.to_datetime(df.get("Entry Date"), errors="coerce")
            end_dates = pd.to_datetime(df.get("Exit Date"), errors="coerce")
            start_dt = start_dates.min()
            end_dt = end_dates.max()
            if pd.notna(start_dt) and pd.notna(end_dt) and end_dt > start_dt:
                days = max((end_dt - start_dt).days, 1)
                years = days / 365.25
                ending = equity[-1] / 100.0
                if ending > 0:
                    if years >= 0.25:
                        # Use geometric annualization only for reasonably long periods
                        cagr_pct = ((ending) ** (365.25 / days) - 1.0) * 100.0
                    else:
                        # For short windows, report period return as "CAGR" proxy to avoid blow-ups
                        cagr_pct = (ending - 1.0) * 100.0
    except Exception:
        cagr_pct = 0.0

    calmar = (cagr_pct / abs(max_dd)) if max_dd > 0 else 0.0
    # Returns (both styles)
    compounded_return_pct = ((equity[-1] / 100.0 - 1.0) * 100.0) if equity else 0.0
    simple_return_pct = sum(returns) if returns else 0.0
    total_return_pct = compounded_return_pct  # backward compatibility
    recovery_factor = (total_return_pct / abs(max_dd)) if max_dd > 0 else 0.0

    stats: Dict[str, float] = {
        "trades": float(len(returns)),
        "win_rate_pct": (len(wins) / len(returns) * 100.0) if returns else 0.0,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "risk_reward": rr,
        "total_return_pct": total_return_pct,
        "compounded_return_pct": compounded_return_pct,
        "simple_return_pct": simple_return_pct,
        "max_drawdown_pct": max_dd,
        "expectancy_pct": expectancy,
        "calmar_ratio": calmar,
        "recovery_factor": recovery_factor,
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


