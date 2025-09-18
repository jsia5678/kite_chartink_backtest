# Kite Chartink Backtester

CLI and web service to backtest entries from a Chartink CSV using Kite Connect historical data.

## Quickstart (Local)

1. Create and activate a Python 3.10+ environment.
2. Install deps:

```
pip install -r requirements.txt
```

3. Set environment variables:

```
set KITE_API_KEY=your_key
set KITE_ACCESS_TOKEN=your_access_token
set EXCHANGE=NSE
set MARKET_TZ=Asia/Kolkata
```

Note: If on Unix shells use `export` instead of `set`.

4. Run CLI:

```
python backtest.py --file chartink.csv --days 5 --output results.csv
```

5. Run web server:

```
uvicorn app.web:app --reload
```

POST `/backtest` with form fields: `file` (CSV), `days` (int), optional `exchange`, `tz`, `output` = `json` or `csv`.

GET `/login_url` to obtain the Kite Connect login URL (requires `KITE_API_KEY`). After completing login, set the `KITE_REQUEST_TOKEN` and `KITE_API_SECRET` env vars so the app can generate `KITE_ACCESS_TOKEN` at startup.

### Getting an access token (two options)

1) Quick, manual (recommended for Railway):

- In the Kite developer console, set Redirect URL to your deployed URL plus `/auth/callback` (HTTPS). Example: `https://your-app.onrailway.app/auth/callback`.
- Start the app on Railway.
- Visit `GET /login_url`, it returns the login URL. Open it, complete login.
- After redirect, you’ll hit `/auth/callback?request_token=...`. The app exchanges it for `access_token` and saves it at `/tmp/kite_token.json` (path can be overridden by `KITE_TOKEN_PATH`).
- Add the printed `access_token` to Railway env as `KITE_ACCESS_TOKEN` for future runs (or keep `KITE_API_SECRET` set so the app can regenerate when you provide a fresh `request_token`).

2) Manual outside the app:

```python
from kiteconnect import KiteConnect
kite = KiteConnect(api_key="YOUR_API_KEY")
data = kite.generate_session("REQUEST_TOKEN", api_secret="YOUR_API_SECRET")
print(data["access_token"])  # set as KITE_ACCESS_TOKEN
```

Docs: [Kite Connect v3 login flow](https://kite.trade/docs/connect/v3/)

## CSV Format

Expected columns (case-insensitive, flexible):
- Stock / Symbol / Tradingsymbol
- Entry Date / Date (YYYY-MM-DD)
- Entry Time / Time (HH:MM or HH:MM:SS, local market tz)

## Logic
- Entry price = close of 1-hour candle at/just before the provided time on entry day.
- Exit price = daily close after N trading days (Mon-Fri, holidays ignored).

## Deploy to Railway
- Repo must include `requirements.txt` and `Procfile`.
- Add environment variables in Railway project settings:
  - `KITE_API_KEY`
  - `KITE_ACCESS_TOKEN` (or set `KITE_API_SECRET` + `KITE_REQUEST_TOKEN` to auto-generate once at startup)
  - optionally `EXCHANGE` (default: NSE), `MARKET_TZ` (default: Asia/Kolkata)

The web process is started by Procfile.

## Notes
- This uses Kite Connect historical API; ensure your account has access.
- Holiday calendars are not included; N trading days count excludes weekends only by default.
- Symbols are resolved via instruments; NSE `-EQ` suffix is attempted automatically if needed.

### Kite access token
Per the official client, you can generate an access token from a `request_token` using your `api_secret`:

```python
from kiteconnect import KiteConnect
kite = KiteConnect(api_key="YOUR_API_KEY")
data = kite.generate_session("REQUEST_TOKEN", api_secret="YOUR_API_SECRET")
print(data["access_token"])  # set as KITE_ACCESS_TOKEN
```

References: [pykiteconnect repository](https://github.com/zerodha/pykiteconnect)