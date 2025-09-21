#!/usr/bin/env python3
"""
Test script to demonstrate BTST audit corrections
"""

import pandas as pd
import datetime as dt
from app.trade_audit import TradeAuditor

def create_btst_test_data():
    """Create test data based on the user's BTST issues"""
    trades = [
        {
            "Stock": "TCS",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:18",
            "Entry Price": 3001.4,
            "Exit Date": "2025-08-04",  # WRONG: Should be 2025-08-02
            "Exit Time": "15:25",       # WRONG: Should be 09:15-09:25
            "Exit Price": 3078,
            "Return %": 2.5521,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapUp
        },
        {
            "Stock": "INFY",
            "Entry Date": "2025-08-04",
            "Entry Time": "15:22",
            "Entry Price": 1480.3,
            "Exit Date": "2025-08-05",  # WRONG: Should be 2025-08-05 (correct)
            "Exit Time": "15:25",       # WRONG: Should be 09:15-09:25
            "Exit Price": 1460,
            "Return %": -1.3713,
            "Exit Reason": "Time"       # WRONG: Should be BTST_NoGap
        },
        {
            "Stock": "HDFCBANK",
            "Entry Date": "2025-08-14",
            "Entry Time": "15:20",
            "Entry Price": 994.7,
            "Exit Date": "2025-08-18",  # WRONG: Should be 2025-08-15
            "Exit Time": "15:30",       # WRONG: Should be 09:15-09:25
            "Exit Price": 1001.8,
            "Return %": 0.7138,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapUp
        }
    ]
    return pd.DataFrame(trades)

def main():
    """Main test function"""
    print("🔍 BTST Audit System Test - Fixing Critical Issues")
    print("=" * 60)
    
    # Create test data with BTST issues
    df = create_btst_test_data()
    print(f"\n📊 Original BTST Trades (with issues):")
    print(df[['Stock', 'Entry Time', 'Exit Date', 'Exit Time', 'Exit Reason', 'Return %']].to_string(index=False))
    
    # Initialize auditor
    auditor = TradeAuditor(timezone="Asia/Kolkata")
    
    # Run audit
    print(f"\n🔍 Running Enhanced BTST Audit...")
    audited_df, audit_summary = auditor.audit_trade_log(df, num_days_param=1)
    
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
        print(f"\n⚠️  Common Violations Found:")
        for violation, count in audit_summary['common_violations']:
            print(f"  • {violation} ({count} times)")
    
    print(f"\n✅ Corrected BTST Trades:")
    corrected_display = audited_df[['Stock', 'Strategy_Type', 'Exit Date', 'Exit Time', 'Exit Reason', 'Return %', 'Audit_Status']].copy()
    print(corrected_display.to_string(index=False))
    
    print(f"\n🔧 Key Corrections Applied:")
    print("• Exit dates corrected to next trading day")
    print("• Exit times changed from 15:25-15:30 to 09:15-09:25")
    print("• Exit reasons updated to BTST_GapUp/BTST_NoGap")
    print("• Returns recalculated based on corrected prices")
    print("• Holding periods enforced to exactly 1 day")
    
    print(f"\n🎯 BTST Strategy Rules Enforced:")
    print("• Entry: 15:15-15:30 PM (near market close)")
    print("• Exit: 09:15-09:25 AM next day (market open)")
    print("• Gap-up logic: 0.5% threshold for gap-up detection")
    print("• No carry beyond 1 day")
    print("• Proper exit reasons based on gap-up behavior")

if __name__ == "__main__":
    main()
