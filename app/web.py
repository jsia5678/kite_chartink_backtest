from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import pytz
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .engine import run_backtest_from_csv, compute_equity_and_stats, compute_insights
from .kite_service import KiteService

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
            raise ValueError("Missing api_key. Pass ?api_key=... or set KITE_API_KEY")
        # Prefer provided api_secret, else read from env
        secret = api_secret or request.cookies.get("kite_api_secret") or os.environ.get("KITE_API_SECRET")
        if not secret:
            raise ValueError("KITE_API_SECRET not provided or set")
        # Trim accidental whitespace
        key = key.strip()
        secret = secret.strip()

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
        return resp
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


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
        },
    )


@app.post("/ui/backtest", response_class=HTMLResponse)
async def ui_backtest(request: Request, file: UploadFile = File(...), days: int = Form(...), exchange: str = Form("NSE"), tz: str = Form("Asia/Kolkata"), sl_pct: Optional[str] = Form(default=None), tp_pct: Optional[str] = Form(default=None), entry_time: Optional[str] = Form(default=None), entry_time2: Optional[str] = Form(default=None), entry_time3: Optional[str] = Form(default=None), cap_small: Optional[str] = Form(default=None), cap_mid: Optional[str] = Form(default=None), cap_large: Optional[str] = Form(default=None), cap_meta_csv: Optional[UploadFile] = File(default=None), breakeven_profit_pct: Optional[str] = Form(default=None), breakeven_at_sl: Optional[str] = Form(default=None)):
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
            {"request": request, "rows": records, "count": len(records), "stats": stats, "equity": equity, "insights": insights},
        )
    except Exception as e:
        # Show error on the results page instead of 500
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "rows": [], "count": 0, "error": str(e)},
            status_code=400,
        )


