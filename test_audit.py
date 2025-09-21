#!/usr/bin/env python3
"""
Test script to demonstrate the trade audit functionality
"""

import pandas as pd
import datetime as dt
from app.trade_audit import TradeAuditor

def create_sample_trades():
    """Create sample trades for testing audit functionality"""
    trades = [
        {
            "Stock": "RELIANCE",
            "Entry Date": "2024-01-15",
            "Entry Time": "09:15",
            "Entry Price": 2500.0,
            "Exit Date": "2024-01-15",
            "Exit Time": "15:30",
            "Exit Price": 2550.0,
            "Return %": 2.0,
            "Exit Reason": "TP"
        },
        {
            "Stock": "TCS",
            "Entry Date": "2024-01-15",
            "Entry Time": "15:15",
            "Entry Price": 3500.0,
            "Exit Date": "2024-01-16",
            "Exit Time": "09:15",
            "Exit Price": 3600.0,
            "Return %": 2.86,
            "Exit Reason": "BTST_Open"
        },
        {
            "Stock": "INFY",
            "Entry Date": "2024-01-15",
            "Entry Time": "10:00",
            "Entry Price": 1500.0,
            "Exit Date": "2024-01-20",
            "Exit Time": "15:30",
            "Exit Price": 1600.0,
            "Return %": 6.67,
            "Exit Reason": "Swing_Time"
        },
        {
            "Stock": "HDFC",
            "Entry Date": "2024-01-15",
            "Entry Time": "09:30",
            "Entry Price": 2000.0,
            "Exit Date": "2024-01-15",
            "Exit Time": "15:30",
            "Exit Price": 1950.0,
            "Return %": -2.5,
            "Exit Reason": "SL"
        },
        {
            "Stock": "WIPRO",
            "Entry Date": "2024-01-15",
            "Entry Time": "15:15",
            "Entry Price": 400.0,
            "Exit Date": "2024-01-15",
            "Exit Time": "15:30",
            "Exit Price": 410.0,
            "Return %": 2.5,
            "Exit Reason": "Time"
        }
    ]
    return pd.DataFrame(trades)

def main():
    """Main test function"""
    print("🔍 Trade Audit System Test")
    print("=" * 50)
    
    # Create sample trades
    df = create_sample_trades()
    print(f"\n📊 Sample Trades ({len(df)} trades):")
    print(df[['Stock', 'Entry Time', 'Exit Time', 'Exit Reason', 'Return %']].to_string(index=False))
    
    # Initialize auditor
    auditor = TradeAuditor(timezone="Asia/Kolkata")
    
    # Run audit
    print(f"\n🔍 Running Trade Audit...")
    audited_df, audit_summary = auditor.audit_trade_log(df, num_days_param=5)
    
    # Display results
    print(f"\n📈 Audit Summary:")
    print(f"Total Trades: {audit_summary['total_trades']}")
    print(f"Passed: {audit_summary['passed_trades']}")
    print(f"Failed: {audit_summary['failed_trades']}")
    print(f"Pass Rate: {audit_summary['pass_rate']:.1f}%")
    
    print(f"\n📊 Strategy Distribution:")
    for strategy, count in audit_summary['strategy_distribution'].items():
        print(f"  {strategy}: {count} trades")
    
    if audit_summary['common_violations']:
        print(f"\n⚠️  Common Violations:")
        for violation, count in audit_summary['common_violations'][:3]:
            print(f"  • {violation} ({count} times)")
    
    print(f"\n✅ Audited Trades:")
    print(audited_df[['Stock', 'Strategy_Type', 'Audit_Status', 'Violations']].to_string(index=False))
    
    print(f"\n🎯 Key Features Demonstrated:")
    print("• Automatic strategy type detection (Intraday, BTST, Swing)")
    print("• Entry/exit timing validation")
    print("• Holding period validation")
    print("• Exit reason validation")
    print("• Trade corrections based on strategy rules")
    print("• Comprehensive audit reporting")

if __name__ == "__main__":
    main()
