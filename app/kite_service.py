from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional
import time
import json

import pandas as pd
import pytz

IMPORT_ERR: Optional[BaseException] = None
try:
    from kiteconnect import KiteConnect
except Exception as e:  # pragma: no cover
    KiteConnect = None  # type: ignore
    IMPORT_ERR = e


@dataclass
class Instrument:
    tradingsymbol: str
    exchange: str
    instrument_token: int


class KiteService:
    def __init__(self, api_key: str, access_token: str) -> None:
        if KiteConnect is None:
            raise RuntimeError(f"kiteconnect import error: {IMPORT_ERR}")
        self.client = KiteConnect(api_key=api_key)
        self.client.set_access_token(access_token)
        self._instruments_cache: Optional[pd.DataFrame] = None

    @classmethod
    def from_env(cls) -> "KiteService":
        api_key = os.environ.get("KITE_API_KEY")
        access_token = os.environ.get("KITE_ACCESS_TOKEN")
        api_secret = os.environ.get("KITE_API_SECRET")
        request_token = os.environ.get("KITE_REQUEST_TOKEN")
        token_path = os.environ.get("KITE_TOKEN_PATH", "/tmp/kite_token.json")
        if not api_key:
            raise RuntimeError("Missing KITE_API_KEY env var")

        # If access token not provided, try loading from token file
        if not access_token and token_path and os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    access_token = data.get("access_token")
            except Exception:
                pass

        # If still missing, try generating it using request token + secret
        if not access_token and request_token and api_secret:
            if KiteConnect is None:
                raise RuntimeError(f"kiteconnect import error: {IMPORT_ERR}")
            tmp_client = KiteConnect(api_key=api_key)
            data = tmp_client.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            # Save if a token path is configured
            if token_path:
                try:
                    with open(token_path, "w", encoding="utf-8") as f:
                        json.dump({"access_token": access_token}, f)
                except Exception:
                    pass

        if not access_token:
            raise RuntimeError("Missing KITE_ACCESS_TOKEN; or set KITE_REQUEST_TOKEN and KITE_API_SECRET to generate it")

        return cls(api_key=api_key, access_token=access_token)

    @staticmethod
    def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> str:
        if KiteConnect is None:
            raise RuntimeError(f"kiteconnect import error: {IMPORT_ERR}")
        client = KiteConnect(api_key=api_key)
        data = client.generate_session(request_token=request_token, api_secret=api_secret)
        return data["access_token"]

    def _load_instruments(self) -> pd.DataFrame:
        if self._instruments_cache is None:
            data = self.client.instruments()
            self._instruments_cache = pd.DataFrame(data)
        return self._instruments_cache

    def resolve_instrument_token(self, symbol: str, exchange: str) -> int:
        # Normalize symbol (strip common suffixes)
        base_symbol = symbol.strip().upper()
        if base_symbol.endswith(".NS"):
            base_symbol = base_symbol[:-3]

        df = self._load_instruments()
        match = df[(df["tradingsymbol"].str.upper() == base_symbol) & (df["exchange"] == exchange)]
        if match.empty:
            # Try appending -EQ for NSE
            if exchange == "NSE" and not symbol.endswith("-EQ"):
                alt = f"{base_symbol}-EQ"
                match = df[(df["tradingsymbol"].str.upper() == alt) & (df["exchange"] == exchange)]
        if match.empty:
            raise ValueError(f"Instrument not found for {exchange}:{symbol}")
        return int(match.iloc[0]["instrument_token"])  # type: ignore

    def fetch_ohlc(
        self,
        token: int,
        interval: str,
        start: dt.datetime,
        end: dt.datetime,
        tz: pytz.BaseTzInfo,
    ) -> pd.DataFrame:
        """Fetch historical data and return DataFrame indexed by localized timestamp.

        interval: "minute", "3minute", "5minute", "10minute", "15minute", "30minute", "60minute", "day"
        """
        # Respect rate limits with a simple retry strategy
        max_attempts = 3
        attempt = 0
        while True:
            try:
                records = self.client.historical_data(
                    instrument_token=token,
                    from_date=start.astimezone(pytz.UTC),
                    to_date=end.astimezone(pytz.UTC),
                    interval=interval,
                    continuous=False,
                    oi=False,
                )
                break
            except Exception as e:
                # Retry on evident rate-limit messages
                msg = str(e).lower()
                if ("429" in msg) or ("rate limit" in msg) or ("too many requests" in msg):
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(1.0 * attempt)
                else:
                    raise
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        # Kite returns datetimes in 'date' (tz-aware IST). Handle robustly.
        if "date" in df.columns:
            parsed = pd.to_datetime(df["date"], errors="coerce")
            if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
                idx = parsed.dt.tz_convert(tz)
            else:
                # Assume returned times are UTC if naive, then convert
                idx = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
        else:
            # Fallback synthetic index
            freq = "1D" if interval == "day" else "1H"
            idx = pd.date_range(start=start, end=end, freq=freq).tz_convert(tz)
        df.index = idx
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
        return df[["open", "high", "low", "close"]]

    @staticmethod
    def login_url_from_env(api_key_override: Optional[str] = None) -> str:
        if KiteConnect is None:
            raise RuntimeError(f"kiteconnect import error: {IMPORT_ERR}")
        api_key = api_key_override or os.environ.get("KITE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing api_key. Pass ?api_key=... or set KITE_API_KEY env var")
        client = KiteConnect(api_key=api_key)
        return client.login_url()


