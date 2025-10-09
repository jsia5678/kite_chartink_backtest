from __future__ import annotations

import datetime as dt
from typing import List, Optional, Tuple, Dict

import pandas as pd
import pytz

from .kite_service import KiteService
from .utils import parse_chartink_csv, nearest_prior_timestamp, trading_days_ahead
from .types import BacktestInputRow
from .trade_audit import TradeAuditor


def compute_entry_exit_batch(
    kite: KiteService,
    rows: List[BacktestInputRow],
    num_days: int,
    exchange: str,
    tz: pytz.BaseTzInfo,
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None,
    breakeven_profit_pct: Optional[float] = None,
    breakeven_at_sl: bool = False,
    sl_mode: str = "fixed",                 # fixed | atr
    atr_multiplier: Optional[float] = None,  # if None, derive from cap (Small=1.5, Mid=1.8, Large=2.0)
    gap_filter_pct: Optional[float] = 1.5,   # skip entries if overnight gap > this % (abs)
    exclude_open_minutes: bool = False,      # exclude 9:15-9:30 entries
    exclude_midday: bool = False,            # exclude 12:00-13:30 entries
    exclude_after_3pm: bool = False,         # exclude entries after 15:00
) -> List[dict]:
    """Optimized batch processing for multiple trades"""
    results = []
    daily_data_cache = {}
    
    # Group trades by stock to minimize API calls
    stock_groups = {}
    for i, row in enumerate(rows):
        if row.stock not in stock_groups:
            stock_groups[row.stock] = []
        stock_groups[row.stock].append((i, row))
    
    # Process each stock group
    for stock, trade_list in stock_groups.items():
        # Fetch data once per stock
        min_date = min(trade[1].entry_date for trade in trade_list) - dt.timedelta(days=5)
        max_date = max(trade[1].entry_date for trade in trade_list) + dt.timedelta(days=num_days + 5)
        
        token = kite.resolve_instrument_token(symbol=stock, exchange=exchange)
        daily_data = kite.fetch_ohlc(token=token, interval="day", 
                                   start=tz.localize(dt.datetime.combine(min_date, dt.time(9, 0))),
                                   end=tz.localize(dt.datetime.combine(max_date, dt.time(15, 30))),
                                   tz=tz)
        
        if daily_data.empty:
            # Handle error for all trades of this stock
            for i, row in trade_list:
                results.append({
                    "Stock": row.stock,
                    "Entry Date": row.entry_date.isoformat(),
                    "Entry Time": row.entry_time.strftime("%H:%M"),
                    "Entry Price": 0.0,
                    "Exit Date": row.entry_date.isoformat(),
                    "Exit Time": "15:30",
                    "Exit Price": 0.0,
                    "Exit Reason": "Error",
                    "Return %": 0.0,
                    "Holding Days": 0
                })
            continue
        
        # Cache the data
        cache_key = f"{stock}_{exchange}"
        daily_data_cache[cache_key] = daily_data
        
        # Process all trades for this stock
        for i, row in trade_list:
            try:
                # Optional time windows filtering at ingestion level
                et = row.entry_time
                if exclude_open_minutes and (et >= dt.time(9,15) and et < dt.time(9,30)):
                    continue
                if exclude_midday and (et >= dt.time(12,0) and et < dt.time(13,30)):
                    continue
                if exclude_after_3pm and (et >= dt.time(15,0)):
                    continue

                result = compute_entry_exit_for_row(
                    kite=kite,
                    row=row,
                    num_days=num_days,
                    exchange=exchange,
                    tz=tz,
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    breakeven_profit_pct=breakeven_profit_pct,
                    breakeven_at_sl=breakeven_at_sl,
                    daily_data_cache=daily_data_cache,
                    sl_mode=sl_mode,
                    atr_multiplier=atr_multiplier,
                    gap_filter_pct=gap_filter_pct,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Handle individual trade errors
                results.append({
                    "Stock": row.stock,
                    "Entry Date": row.entry_date.isoformat(),
                    "Entry Time": row.entry_time.strftime("%H:%M"),
                    "Entry Price": 0.0,
                    "Exit Date": row.entry_date.isoformat(),
                    "Exit Time": "15:30",
                    "Exit Price": 0.0,
                    "Exit Reason": "Error",
                    "Return %": 0.0,
                    "Holding Days": 0
                })
    
    return results



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
    daily_data_cache: Optional[Dict[str, pd.DataFrame]] = None,
    sl_mode: str = "fixed",
    atr_multiplier: Optional[float] = None,
    gap_filter_pct: Optional[float] = 1.5,
) -> dict:
    # Build localized entry timestamp
    entry_dt_local = tz.localize(dt.datetime.combine(row.entry_date, row.entry_time))
    

    # Get instrument token
    token = kite.resolve_instrument_token(symbol=row.stock, exchange=exchange)

    # Use cached data if available, otherwise fetch minimal data
    cache_key = f"{row.stock}_{exchange}"
    if daily_data_cache and cache_key in daily_data_cache:
        daily_data = daily_data_cache[cache_key]
    else:
        # Fetch only necessary data range
        start = entry_dt_local - dt.timedelta(days=5)  # Reduced from 10 days
        end = entry_dt_local + dt.timedelta(days=num_days + 5)  # Only fetch what we need
        daily_data = kite.fetch_ohlc(token=token, interval="day", start=start, end=end, tz=tz)
        if daily_data.empty:
            raise ValueError(f"No daily data for {row.stock} around {entry_dt_local}")
        
        # Cache the data for future use
        if daily_data_cache is not None:
            daily_data_cache[cache_key] = daily_data

    # Optimize date processing - use vectorized operations
    entry_date = entry_dt_local.date()
    daily_dates = pd.Index([ts.astimezone(tz).date() for ts in daily_data.index])
    
    # Find entry position more efficiently
    entry_pos = None
    for i, d in enumerate(daily_dates):
        if d <= entry_date:
            entry_pos = i
        else:
            break
    
    if entry_pos is None:
        raise ValueError(f"No daily data for {row.stock} on or before {entry_date}")
    
    entry_price = float(daily_data.iloc[entry_pos]["close"])  # type: ignore

    # Gap filter: compare today's open vs previous close
    if gap_filter_pct is not None and entry_pos + 1 < len(daily_data):
        prev_close = float(daily_data.iloc[entry_pos]["close"])  # previous day close (since entry_pos points to <= entry_date)
        # find next row which is the entry_date row (if entry_pos corresponds to day before when entries are intraday)
        # safer: locate exact date
        try:
            idx_on_date = None
            for i, d in enumerate(daily_dates):
                if d == entry_date:
                    idx_on_date = i
                    break
            if idx_on_date is not None and "open" in daily_data.columns:
                today_open = float(daily_data.iloc[idx_on_date]["open"])  # type: ignore
                gap_pct = abs((today_open - prev_close) / prev_close) * 100.0 if prev_close else 0.0
                if gap_pct > float(gap_filter_pct):
                    return None  # skip this trade entirely
        except Exception:
            pass

    # Compute cap-based max holding days
    cap_to_max_days = {
        "SMALL": 5,
        "MID": 7,
        "LARGE": 10,
    }
    max_days_by_cap = None
    if getattr(row, "cap_bucket", None):
        cap_key = (row.cap_bucket or "").strip().upper()
        # normalize common variants
        if cap_key.startswith("SMALL"):
            cap_key = "SMALL"
        elif cap_key.startswith("MID"):
            cap_key = "MID"
        elif cap_key.startswith("LARGE"):
            cap_key = "LARGE"
        max_days_by_cap = cap_to_max_days.get(cap_key)

    effective_days = num_days
    if max_days_by_cap is not None:
        effective_days = min(num_days, max_days_by_cap)

    # Defaults - regular trades: Exit at market close based on effective_days
    exit_trade_date = trading_days_ahead(entry_dt_local.date(), effective_days)
    exit_ts = tz.localize(dt.datetime.combine(exit_trade_date, dt.time(15, 30)))
    exit_time_str = "15:30"
    exit_reason = "Time"
    
    exit_price: Optional[float] = None

    # Determine SL/TP based on mode
    target_price = None
    stop_price = None
    if sl_mode == "atr":
        # compute ATR(14) using daily ohlc up to entry date (exclusive)
        try:
            import numpy as np
            highs = pd.to_numeric(daily_data.get("high"), errors="coerce").astype(float)
            lows = pd.to_numeric(daily_data.get("low"), errors="coerce").astype(float)
            closes = pd.to_numeric(daily_data.get("close"), errors="coerce").astype(float)
            tr_list = []
            prev_close = None
            for h, l, c in zip(highs, lows, closes):
                if prev_close is None:
                    tr = h - l
                else:
                    tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
                tr_list.append(tr)
                prev_close = c
            atr_series = pd.Series(tr_list).rolling(14).mean()
            # pick ATR on the bar prior to entry date
            atr_value = None
            for i in range(len(daily_data)):
                if daily_dates[i] == entry_date:
                    atr_value = float(atr_series.iloc[max(i-1, 0)])
                    break
            if atr_value is None:
                atr_value = float(atr_series.iloc[-1])
            # derive multiplier default by cap
            mult = atr_multiplier
            if mult is None:
                cap = (row.cap_bucket or "").upper() if getattr(row, "cap_bucket", None) else ""
                mult = 1.5 if cap.startswith("SMALL") else 1.8 if cap.startswith("MID") else 2.0
            # convert to prices
            stop_price = max(0.0, entry_price - mult * atr_value)
            if tp_pct is not None and tp_pct > 0:
                target_price = entry_price * (1.0 + tp_pct / 100.0)
            else:
                target_price = entry_price + mult * atr_value
        except Exception:
            # fallback to fixed
            if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0):
                target_price = entry_price * (1.0 + (tp_pct or 0.0) / 100.0) if tp_pct else None
                stop_price = entry_price * (1.0 - (sl_pct or 0.0) / 100.0) if sl_pct else None
    else:
        if (sl_pct is not None and sl_pct > 0) or (tp_pct is not None and tp_pct > 0):
            target_price = entry_price * (1.0 + (tp_pct or 0.0) / 100.0) if tp_pct else None
            stop_price = entry_price * (1.0 - (sl_pct or 0.0) / 100.0) if sl_pct else None
        
    # For swing trades, SL/TP is checked at daily close only
    if target_price is not None or stop_price is not None:
        # Optimize SL/TP checking with vectorized operations
        if entry_pos + 1 < len(daily_data):
            # Get all closes from entry date onwards
            future_data = daily_data.iloc[entry_pos + 1:]
            closes = future_data["close"].astype(float)
            
            # Vectorized TP check
            if target_price is not None:
                tp_hits = closes >= target_price
                if tp_hits.any():
                    first_tp_idx = tp_hits.idxmax()
                    exit_reason = "TP"
                    exit_price = float(closes[first_tp_idx])
                    exit_ts = first_tp_idx
            
            # Vectorized SL check (only if TP didn't trigger)
            if exit_price is None and stop_price is not None:
                sl_hits = closes <= stop_price
                if sl_hits.any():
                    first_sl_idx = sl_hits.idxmax()
                    exit_reason = "SL"
                    exit_price = float(closes[first_sl_idx])
                    exit_ts = first_sl_idx

    # If neither SL nor TP triggered, exit at scheduled date using daily data
    if exit_price is None:
        # Find the exit date in our daily data
        exit_dates = pd.Index([ts.astimezone(tz).date() for ts in daily_data.index])
        exit_pos = None
        for i, d in enumerate(exit_dates):
            if d >= exit_trade_date:
                exit_pos = i
                break
        
        if exit_pos is None:
            # If exit date not found, use the last available date
            exit_pos = len(daily_data) - 1
        
        exit_ts = daily_data.index[exit_pos]
        exit_price = float(daily_data.iloc[exit_pos]["close"])  # type: ignore
        exit_ts_local = exit_ts.astimezone(tz)
        # Exit time enforcement: disallow 00:00 and force <= 15:25
        if exit_ts_local.hour == 0 and exit_ts_local.minute == 0:
            # shift to market close
            exit_ts_local = exit_ts_local.replace(hour=15, minute=25, second=0, microsecond=0)
            exit_reason = "Rule-triggered exit"
        if (exit_ts_local.hour, exit_ts_local.minute) > (15, 25):
            exit_ts_local = exit_ts_local.replace(hour=15, minute=25, second=0, microsecond=0)
            exit_reason = "Rule-triggered exit"
        exit_time_str = exit_ts_local.strftime("%H:%M")
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
    # Use optimized batch processing
    results = compute_entry_exit_batch(
                    kite=kite,
        rows=rows,
                    num_days=num_days,
                    exchange=exchange,
                    tz=tz,
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    breakeven_profit_pct=breakeven_profit_pct,
                    breakeven_at_sl=breakeven_at_sl,
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


