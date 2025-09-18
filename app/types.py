from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class BacktestInputRow:
    stock: str
    entry_date: dt.date
    entry_time: dt.time


