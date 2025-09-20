from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import pytz
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .engine import run_backtest_from_csv, compute_equity_and_stats, compute_insights, compute_entry_exit_for_row
import httpx
import json as _json
from .kite_service import KiteService
from .utils import parse_chartink_csv

app = FastAPI(title="Kite Chartink Backtester")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/backtest")
def _to_opt_float(val: Optional[str]) -> Optional[float]:
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


async def backtest_endpoint(
    file: UploadFile = File(...),
    days: int = Form(...),
    exchange: str = Form(default=os.environ.get("EXCHANGE", "NSE")),
    tz: str = Form(default=os.environ.get("MARKET_TZ", "Asia/Kolkata")),
    output: str = Form(default="json"),
    sl_pct: Optional[str] = Form(default=None),
    tp_pct: Optional[str] = Form(default=None),
    entry_time: Optional[str] = Form(default=None),
    entry_time2: Optional[str] = Form(default=None),
    entry_time3: Optional[str] = Form(default=None),
    cap_small: Optional[str] = Form(default=None),
    cap_mid: Optional[str] = Form(default=None),
    cap_large: Optional[str] = Form(default=None),
    cap_meta_csv: Optional[UploadFile] = File(default=None),
    breakeven_profit_pct: Optional[str] = Form(default=None),
    breakeven_at_sl: Optional[str] = Form(default=None),
):
    # Save uploaded file to a temp buffer and run backtest
    content = await file.read()
    tmp = io.BytesIO(content)
    # Pandas can read from buffer, but our engine expects a path. Write temp.
    tmp_path = f"/tmp/{file.filename or 'input.csv'}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        allowed = [t for t in [entry_time, entry_time2, entry_time3] if t]
        # Default to including all caps if none explicitly selected
        cap_allowed = []
        if cap_small:
            cap_allowed.append("Small")
        if cap_mid:
            cap_allowed.append("Mid")
        if cap_large:
            cap_allowed.append("Large")
        if not cap_allowed:
            cap_allowed = ["Small", "Mid", "Large"]

        # Persist cap meta CSV if provided
        cap_meta_path = None
        if cap_meta_csv is not None:
            try:
                content = await cap_meta_csv.read()
                cap_meta_path = f"/tmp/{cap_meta_csv.filename or 'cap_meta.csv'}"
                with open(cap_meta_path, 'wb') as f:
                    f.write(content)
            except Exception:
                cap_meta_path = None

        df = run_backtest_from_csv(
            csv_path=tmp_path,
            num_days=days,
            exchange=exchange,
            timezone_name=tz,
            sl_pct=_to_opt_float(sl_pct),
            tp_pct=_to_opt_float(tp_pct),
            allowed_entry_times=allowed or None,
            allowed_cap_buckets=cap_allowed or None,
            symbol_cap_csv_path=cap_meta_path,
            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
            breakeven_at_sl=bool(breakeven_at_sl),
        )
        if output == "csv":
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv")
        else:
            return JSONResponse(df.to_dict(orient="records"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/login_url")
async def login_url(api_key: str | None = None):
    try:
        url = KiteService.login_url_from_env(api_key_override=api_key)
        return {"login_url": url}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/auth/callback")
async def auth_callback(request: Request, request_token: str, api_secret: str | None = None, api_key: str | None = None, format: str | None = None):
    try:
        # Exchange request_token for access_token
        key = api_key or request.cookies.get("kite_api_key") or os.environ.get("KITE_API_KEY")
        if not key:
            raise ValueError("Missing API key. Please ensure cookies are enabled and try logging in again.")
        
        # Prefer provided api_secret, else read from cookies/env
        secret = api_secret or request.cookies.get("kite_api_secret") or os.environ.get("KITE_API_SECRET")
        if not secret:
            raise ValueError("Missing API secret. Please ensure cookies are enabled and try logging in again.")
        
        # Trim accidental whitespace
        key = key.strip()
        secret = secret.strip()
        
        # Log for debugging (remove in production)
        print(f"Auth callback: key={key[:8]}..., secret={'*' * len(secret)}, request_token={request_token[:8]}...")

        access_token = KiteService.exchange_request_token(api_key=key, api_secret=secret, request_token=request_token)
        
        # Optionally persist to token file
        token_path = os.environ.get("KITE_TOKEN_PATH", "/tmp/kite_token.json")
        try:
            with open(token_path, "w", encoding="utf-8") as f:
                f.write('{"access_token": "' + access_token + '"}')
        except Exception:
            pass

        # If JSON requested explicitly, return JSON; otherwise redirect to home with banner
        accepts = (request.headers.get("accept") or "").lower()
        if format == "json" or "application/json" in accepts:
            return {"access_token": access_token, "saved_to": token_path}

        resp = RedirectResponse(url="/?authed=1", status_code=303)
        # Optional cookie for UX; mark secure only on HTTPS
        forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        cookie_secure = (request.url.scheme == "https") or (forwarded_proto == "https")
        resp.set_cookie("kite_access_token", access_token, max_age=12 * 60 * 60, httponly=True, secure=cookie_secure, samesite="lax")
        resp.set_cookie("kite_api_key", key, max_age=12 * 60 * 60, httponly=True, secure=cookie_secure, samesite="lax")
        return resp
    except Exception as e:
        print(f"Auth callback error: {e}")
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "expired" in error_msg.lower():
            error_msg = "Authentication failed. Please check your API credentials and try again."
        return JSONResponse({"error": error_msg}, status_code=400)


@app.post("/ui/login")
async def ui_login(request: Request, api_key: str = Form(...), api_secret: str = Form(...)):
    # Set short-lived cookies with api_key and api_secret, then redirect to Kite login URL
    try:
        url = KiteService.login_url_from_env(api_key_override=api_key)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    response = RedirectResponse(url, status_code=303)
    # Short-lived cookies (10 minutes)
    max_age = 10 * 60
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    cookie_secure = (request.url.scheme == "https") or (forwarded_proto == "https")
    response.set_cookie("kite_api_key", api_key.strip(), max_age=max_age, httponly=True, secure=cookie_secure, samesite="lax")
    response.set_cookie("kite_api_secret", api_secret.strip(), max_age=max_age, httponly=True, secure=cookie_secure, samesite="lax")
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_days": 5,
            "default_exchange": os.environ.get("EXCHANGE", "NSE"),
            "default_tz": os.environ.get("MARKET_TZ", "Asia/Kolkata"),
            "kite_connected": bool((request.cookies.get("kite_access_token") or "").strip()),
            "pplx_present": bool((request.cookies.get("pplx_api_key") or "").strip()),
        },
    )


@app.post("/ui/backtest", response_class=HTMLResponse)
async def ui_backtest(
    request: Request,
    file: UploadFile = File(...),
    days: int = Form(...),
    exchange: str = Form("NSE"),
    tz: str = Form("Asia/Kolkata"),
    sl_pct: Optional[str] = Form(default=None),
    tp_pct: Optional[str] = Form(default=None),
    entry_time: Optional[str] = Form(default=None),
    entry_time2: Optional[str] = Form(default=None),
    entry_time3: Optional[str] = Form(default=None),
    cap_small: Optional[str] = Form(default=None),
    cap_mid: Optional[str] = Form(default=None),
    cap_large: Optional[str] = Form(default=None),
    cap_meta_csv: Optional[UploadFile] = File(default=None),
    breakeven_profit_pct: Optional[str] = Form(default=None),
    breakeven_at_sl: Optional[str] = Form(default=None),
):
    # Ensure required Kite credentials are available for engine via env
    try:
        cookie_api_key = (request.cookies.get("kite_api_key") or "").strip()
        cookie_access_token = (request.cookies.get("kite_access_token") or "").strip()
        if cookie_api_key and not os.environ.get("KITE_API_KEY"):
            os.environ["KITE_API_KEY"] = cookie_api_key
        # Access token can also be loaded by the engine from token file; setting env helps if present
        if cookie_access_token and not os.environ.get("KITE_ACCESS_TOKEN"):
            os.environ["KITE_ACCESS_TOKEN"] = cookie_access_token
    except Exception:
        # Non-fatal; engine will still try token file and raise meaningful errors
        pass

    content = await file.read()
    tmp_path = f"/tmp/{file.filename or 'input.csv'}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        allowed = [t for t in [entry_time, entry_time2, entry_time3] if t]
        cap_allowed = []
        if cap_small:
            cap_allowed.append("Small")
        if cap_mid:
            cap_allowed.append("Mid")
        if cap_large:
            cap_allowed.append("Large")
        if not cap_allowed:
            cap_allowed = ["Small", "Mid", "Large"]

        # Persist cap meta CSV if provided
        cap_meta_path = None
        if cap_meta_csv is not None:
            try:
                content = await cap_meta_csv.read()
                cap_meta_path = f"/tmp/{cap_meta_csv.filename or 'cap_meta.csv'}"
                with open(cap_meta_path, 'wb') as f:
                    f.write(content)
            except Exception:
                cap_meta_path = None

        df = run_backtest_from_csv(
            csv_path=tmp_path,
            num_days=days,
            exchange=exchange,
            timezone_name=tz,
            sl_pct=_to_opt_float(sl_pct),
            tp_pct=_to_opt_float(tp_pct),
            allowed_entry_times=allowed or None,
            allowed_cap_buckets=cap_allowed or None,
            symbol_cap_csv_path=cap_meta_path,
            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
            breakeven_at_sl=bool(breakeven_at_sl),
        )
        records = df.to_dict(orient="records")
        equity, stats = compute_equity_and_stats(df)
        insights = compute_insights(df)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "rows": records,
                "count": len(records),
                "stats": stats,
                "equity": equity,
                "insights": insights,
                "csv_path": tmp_path,
                "defaults": {
                    "days": days,
                    "exchange": exchange,
                    "tz": tz,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "breakeven_profit_pct": breakeven_profit_pct,
                    "breakeven_at_sl": bool(breakeven_at_sl),
                },
            },
        )
    except Exception as e:
        # Show error on the results page instead of 500
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "rows": [], "count": 0, "error": str(e)},
            status_code=400,
        )


# Perplexity API key via UI (stores in HttpOnly cookie)
@app.post("/ui/pplx_key")
async def ui_set_pplx_key(request: Request, pplx_api_key: str = Form(...)):
    try:
        key = (pplx_api_key or "").strip()
        if not key:
            raise ValueError("API key cannot be empty")
        resp = RedirectResponse(url="/", status_code=303)
        forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
        cookie_secure = (request.url.scheme == "https") or (forwarded_proto == "https")
        resp.set_cookie("pplx_api_key", key, max_age=12 * 60 * 60, httponly=True, secure=cookie_secure, samesite="lax")
        return resp
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# --- AI Strategy generation via Perplexity ---
@app.post("/ui/ai_strategy", response_class=HTMLResponse)
async def ui_ai_strategy(
    request: Request,
    prompt: str = Form(...),
    days: int = Form(...),
    exchange: str = Form("NSE"),
    tz: str = Form("Asia/Kolkata"),
    sl_pct: Optional[str] = Form(default=None),
    tp_pct: Optional[str] = Form(default=None),
    breakeven_profit_pct: Optional[str] = Form(default=None),
    breakeven_at_sl: Optional[str] = Form(default=None),
    timeframe: Optional[str] = Form(default=None),
    from_date: Optional[str] = Form(default=None),
    to_date: Optional[str] = Form(default=None),
):
    """
    Uses Perplexity API (API key from PPLX_API_KEY) to turn a natural-language strategy prompt
    into a CSV of entries with columns: stock, entry_date, entry_time. The CSV is then backtested.
    """
    try:
        # Ensure required Kite credentials are available for engine via env (mirror /ui/backtest)
        try:
            cookie_api_key = (request.cookies.get("kite_api_key") or "").strip()
            cookie_access_token = (request.cookies.get("kite_access_token") or "").strip()
            if cookie_api_key and not os.environ.get("KITE_API_KEY"):
                os.environ["KITE_API_KEY"] = cookie_api_key
            if cookie_access_token and not os.environ.get("KITE_ACCESS_TOKEN"):
                os.environ["KITE_ACCESS_TOKEN"] = cookie_access_token
        except Exception:
            pass

        pplx_key = os.environ.get("PPLX_API_KEY")
        # Allow key via cookie for convenience
        try:
            cookie_key = (request.cookies.get("pplx_api_key") or "").strip()
            if cookie_key:
                pplx_key = pplx_key or cookie_key
        except Exception:
            pass
        if not pplx_key:
            raise RuntimeError("Missing PPLX_API_KEY in environment")

        # Determine intraday interval from timeframe hint (default 15minute)
        tf_map = {
            "1m": "minute",
            "1min": "minute",
            "5m": "5minute",
            "5min": "5minute",
            "10m": "10minute",
            "15m": "15minute",
            "15min": "15minute",
            "30m": "30minute",
            "30min": "30minute",
            "60m": "60minute",
            "1h": "60minute",
            "hour": "60minute",
        }
        intraday_interval = "15minute"
        if timeframe:
            key = timeframe.strip().lower()
            intraday_interval = tf_map.get(key, intraday_interval)
        else:
            # Try to infer from prompt keywords
            pl = (prompt or "").lower()
            for k, v in tf_map.items():
                if k in pl:
                    intraday_interval = v
                    break

        # Enhance system instructions with optional date range
        date_hint = ""
        if from_date and to_date:
            date_hint = f" Limit entries to dates between {from_date} and {to_date}."

        system_instructions = (
            "You are a trading assistant. Given a user strategy idea, output a CSV ONLY with columns "
            "stock,entry_date,entry_time for Indian equities (NSE), with past historical realistic entries." 
            " Use format: stock as tradingsymbol (e.g., TCS, RELIANCE), entry_date as YYYY-MM-DD, entry_time as HH:MM (24h)." 
            " Output at least 25 rows spanning multiple dates. Do not include any commentary, only CSV rows with a header." 
            + date_hint
        )
        
        payload = {
            "model": os.environ.get("PPLX_MODEL", "sonar-pro"),
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {pplx_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text content
        content = ""
        try:
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
        except Exception:
            content = ""

        if not content:
            raise RuntimeError("Empty response from Perplexity")

        # Clean fenced code blocks if present
        if "```" in content:
            parts = content.split("```")
            # Prefer the first fenced block that looks like CSV
            csv_text = None
            for i in range(1, len(parts), 2):
                block = parts[i]
                if "," in block and ("stock" in block.lower() or "entry_date" in block.lower()):
                    csv_text = block
                    break
            content = csv_text or content

        # Persist CSV to temp after enforcing optional date range
        tmp_csv_path = "/tmp/ai_strategy.csv"
        try:
            import io as _io
            # Parse and normalize CSV
            _df = pd.read_csv(_io.StringIO(content.strip()))
            cols = {c.lower().strip(): c for c in _df.columns}
            # Standardize expected column names if variants are present
            if "stock" not in cols or "entry_date" not in cols or "entry_time" not in cols:
                # Try forgiving rename using lower-case
                _df.columns = [str(c).strip() for c in _df.columns]
                lower_map = {c.lower(): c for c in _df.columns}
                cols = lower_map
            # Filter date range if provided
            if from_date or to_date:
                try:
                    date_col = cols.get("entry_date", "entry_date")
                    _df[date_col] = pd.to_datetime(_df[date_col]).dt.date
                    _from = pd.to_datetime(from_date).date() if from_date else None
                    _to = pd.to_datetime(to_date).date() if to_date else None
                    if _from:
                        _df = _df[_df[date_col] >= _from]
                    if _to:
                        _df = _df[_df[date_col] <= _to]
                except Exception:
                    # If parsing fails, fall back to raw content
                    pass
            _df.to_csv(tmp_csv_path, index=False)
        except Exception:
            # Fallback: write raw content
            with open(tmp_csv_path, "w", encoding="utf-8") as f:
                f.write(content.strip())

        df = run_backtest_from_csv(
            csv_path=tmp_csv_path,
            num_days=days,
            exchange=exchange,
            timezone_name=tz,
            sl_pct=_to_opt_float(sl_pct),
            tp_pct=_to_opt_float(tp_pct),
            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
            breakeven_at_sl=bool(breakeven_at_sl),
        )
        # Re-compute with chosen intraday interval by applying row-wise if intraday_interval differs from default
        # Note: run_backtest_from_csv uses compute_entry_exit_for_row; we need a version that passes intraday_interval.
        try:
            # Lightweight re-run using the same parsed rows by re-reading CSV through engine utils is acceptable here.
            import pytz as _pytz
            rows = parse_chartink_csv(tmp_csv_path, tz=_pytz.timezone(tz))
            kite = KiteService.from_env()
            out = []
            for r in rows:
                try:
                    out.append(
                        compute_entry_exit_for_row(
                            kite=kite,
                            row=r,
                            num_days=days,
                            exchange=exchange,
                            tz=_pytz.timezone(tz),
                            sl_pct=_to_opt_float(sl_pct),
                            tp_pct=_to_opt_float(tp_pct),
                            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
                            breakeven_at_sl=bool(breakeven_at_sl),
                            intraday_interval=intraday_interval,
                        )
                    )
                except Exception as ee:
                    out.append({
                        "Stock": r.stock,
                        "Entry Date": r.entry_date.isoformat(),
                        "Entry Time": r.entry_time.strftime("%H:%M"),
                        "Entry Price": None,
                        "Exit Date": None,
                        "Exit Time": None,
                        "Exit Price": None,
                        "Return %": None,
                        "Exit Reason": None,
                        "Error": str(ee),
                    })
            import pandas as _pd
            df = _pd.DataFrame(out)
        except Exception:
            pass
        
        records = df.to_dict(orient="records")
        equity, stats = compute_equity_and_stats(df)
        insights = compute_insights(df)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "rows": records,
                "count": len(records),
                "stats": stats,
                "equity": equity,
                "insights": insights,
                "ai_prompt": prompt,
                "csv_path": tmp_csv_path,
                "defaults": {
                    "days": days,
                    "exchange": exchange,
                    "tz": tz,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "breakeven_profit_pct": breakeven_profit_pct,
                    "breakeven_at_sl": bool(breakeven_at_sl),
                    "timeframe": timeframe,
                    "from_date": from_date,
                    "to_date": to_date,
                },
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "rows": [], "count": 0, "error": str(e)},
            status_code=400,
        )


@app.post("/ui/rerun", response_class=HTMLResponse)
async def ui_rerun(
    request: Request,
    csv_path: str = Form(...),
    days: int = Form(...),
    exchange: str = Form("NSE"),
    tz: str = Form("Asia/Kolkata"),
    sl_pct: Optional[str] = Form(default=None),
    tp_pct: Optional[str] = Form(default=None),
    cap_small: Optional[str] = Form(default=None),
    cap_mid: Optional[str] = Form(default=None),
    cap_large: Optional[str] = Form(default=None),
    breakeven_profit_pct: Optional[str] = Form(default=None),
    breakeven_at_sl: Optional[str] = Form(default=None),
):
    try:
        allowed_caps = []
        if cap_small:
            allowed_caps.append("Small")
        if cap_mid:
            allowed_caps.append("Mid")
        if cap_large:
            allowed_caps.append("Large")
        if not allowed_caps:
            allowed_caps = ["Small", "Mid", "Large"]

        df = run_backtest_from_csv(
            csv_path=csv_path,
            num_days=days,
            exchange=exchange,
            timezone_name=tz,
            sl_pct=_to_opt_float(sl_pct),
            tp_pct=_to_opt_float(tp_pct),
            allowed_cap_buckets=allowed_caps or None,
            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
            breakeven_at_sl=bool(breakeven_at_sl),
        )
        records = df.to_dict(orient="records")
        equity, stats = compute_equity_and_stats(df)
        insights = compute_insights(df)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "rows": records,
                "count": len(records),
                "stats": stats,
                "equity": equity,
                "insights": insights,
                "csv_path": csv_path,
                "defaults": {
                    "days": days,
                    "exchange": exchange,
                    "tz": tz,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "breakeven_profit_pct": breakeven_profit_pct,
                    "breakeven_at_sl": bool(breakeven_at_sl),
                },
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "rows": [], "count": 0, "error": str(e), "csv_path": csv_path},
            status_code=400,
        )


def _get_pplx_key_from_request(request: Request) -> str:
    key = os.environ.get("PPLX_API_KEY", "").strip()
    try:
        cookie_key = (request.cookies.get("pplx_api_key") or "").strip()
        if cookie_key:
            key = key or cookie_key
    except Exception:
        pass
    return key


async def _pplx_chat(request: Request, messages: list[dict]) -> str:
    pplx_key = _get_pplx_key_from_request(request)
    if not pplx_key:
        raise RuntimeError("Missing PPLX_API_KEY in environment or cookie")
    payload = {
        "model": os.environ.get("PPLX_MODEL", "sonar-pro"),
        "messages": messages,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {pplx_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    content = ""
    try:
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
    except Exception:
        content = ""
    return content


@app.post("/ui/ai_summary")
async def ui_ai_summary(request: Request, json: str = Form(...)):
    try:
        # json is the backtest JSON payload as string
        system = (
            "You analyze trading backtests. Summarize performance in 4 short bullet points: "
            "(1) headline win rate and return quality, (2) risk (max drawdown, volatility) "
            "(3) drivers (cap buckets, entry times), (4) concrete next tests. Keep it under 80 words."
        )
        user = f"Backtest JSON:\n{json}"
        content = await _pplx_chat(request, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ui/ai_chat")
async def ui_ai_chat(request: Request, question: str = Form(...), json: str = Form(...)):
    try:
        system = (
            "You are a helpful analyst. Answer the question using only the provided backtest JSON. "
            "Be concise and propose at most 3 actionable re-run ideas (parameter changes)."
        )
        user = f"Question: {question}\n\nBacktest JSON:\n{json}"
        content = await _pplx_chat(request, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        return PlainTextResponse(content)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ui/export_csv")
async def ui_export_csv(
    csv_path: str = Form(...),
    days: int = Form(...),
    exchange: str = Form("NSE"),
    tz: str = Form("Asia/Kolkata"),
    sl_pct: Optional[str] = Form(default=None),
    tp_pct: Optional[str] = Form(default=None),
    breakeven_profit_pct: Optional[str] = Form(default=None),
    breakeven_at_sl: Optional[str] = Form(default=None),
):
    try:
        df = run_backtest_from_csv(
            csv_path=csv_path,
            num_days=days,
            exchange=exchange,
            timezone_name=tz,
            sl_pct=_to_opt_float(sl_pct),
            tp_pct=_to_opt_float(tp_pct),
            breakeven_profit_pct=_to_opt_float(breakeven_profit_pct),
            breakeven_at_sl=bool(breakeven_at_sl),
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=backtest.csv"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ui/export_notes")
async def ui_export_notes(request: Request, json: str = Form(...)):
    try:
        system = (
            "Write a brief backtest note: Summary, Risks, Suggested Improvements. 120-200 words."
        )
        user = f"Backtest JSON:\n{json}"
        content = await _pplx_chat(request, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        return StreamingResponse(io.BytesIO(content.encode("utf-8")), media_type="text/plain", headers={"Content-Disposition": "attachment; filename=ai_notes.txt"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
