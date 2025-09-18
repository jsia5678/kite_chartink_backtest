import argparse
import os
import sys
from typing import Optional

from app.engine import run_backtest_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chartink + Kite Connect backtester")
    parser.add_argument("--file", required=True, help="Path to Chartink CSV input")
    parser.add_argument("--days", required=True, type=int, help="Number of trading days to hold (N)")
    parser.add_argument("--exchange", default=os.environ.get("EXCHANGE", "NSE"), help="Exchange for symbols (default: NSE)")
    parser.add_argument("--tz", default=os.environ.get("MARKET_TZ", "Asia/Kolkata"), help="Timezone of dates/times in CSV (default: Asia/Kolkata)")
    parser.add_argument("--output", default="results.csv", help="Output CSV filepath (default: results.csv)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format for stdout (default: csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        results_df = run_backtest_from_csv(
            csv_path=args.file,
            num_days=args.days,
            exchange=args.exchange,
            timezone_name=args.tz,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Save to CSV file
    results_df.to_csv(args.output, index=False)

    # Also print to stdout in chosen format
    if args.format == "csv":
        print(results_df.to_csv(index=False))
    else:
        print(results_df.to_json(orient="records"))


if __name__ == "__main__":
    main()


