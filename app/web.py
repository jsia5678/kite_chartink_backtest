from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd
import pytz
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .engine import run_backtest_from_csv
from .kite_service import KiteService

app = FastAPI(title="Kite Chartink Backtester")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/backtest")
async def backtest_endpoint(
    file: UploadFile = File(...),
    days: int = Form(...),
    exchange: str = Form(default=os.environ.get("EXCHANGE", "NSE")),
    tz: str = Form(default=os.environ.get("MARKET_TZ", "Asia/Kolkata")),
    output: str = Form(default="json"),
):
    # Save uploaded file to a temp buffer and run backtest
    content = await file.read()
    tmp = io.BytesIO(content)
    # Pandas can read from buffer, but our engine expects a path. Write temp.
    tmp_path = f"/tmp/{file.filename or 'input.csv'}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    df = run_backtest_from_csv(csv_path=tmp_path, num_days=days, exchange=exchange, timezone_name=tz)

    if output == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv")
    else:
        return JSONResponse(df.to_dict(orient="records"))


@app.get("/login_url")
async def login_url(api_key: str | None = None):
    try:
        url = KiteService.login_url_from_env(api_key_override=api_key)
        return {"login_url": url}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/auth/callback")
async def auth_callback(request_token: str, api_secret: str | None = None, api_key: str | None = None):
    # Exchange request_token for access_token
    key = api_key or os.environ.get("KITE_API_KEY")
    if not key:
        return JSONResponse({"error": "Missing api_key. Pass ?api_key=... or set KITE_API_KEY"}, status_code=400)
    # Prefer provided api_secret, else read from env
    secret = api_secret or os.environ.get("KITE_API_SECRET")
    if not secret:
        return JSONResponse({"error": "KITE_API_SECRET not provided or set"}, status_code=400)
    access_token = KiteService.exchange_request_token(api_key=key, api_secret=secret, request_token=request_token)
    # Optionally persist to token file
    token_path = os.environ.get("KITE_TOKEN_PATH", "/tmp/kite_token.json")
    try:
        with open(token_path, "w", encoding="utf-8") as f:
            f.write('{"access_token": "' + access_token + '"}')
    except Exception:
        pass
    return {"access_token": access_token, "saved_to": token_path}


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
async def ui_backtest(request: Request, file: UploadFile = File(...), days: int = Form(...), exchange: str = Form("NSE"), tz: str = Form("Asia/Kolkata")):
    content = await file.read()
    tmp_path = f"/tmp/{file.filename or 'input.csv'}"
    with open(tmp_path, "wb") as f:
        f.write(content)

    df = run_backtest_from_csv(csv_path=tmp_path, num_days=days, exchange=exchange, timezone_name=tz)
    records = df.to_dict(orient="records")
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "rows": records, "count": len(records)},
    )


