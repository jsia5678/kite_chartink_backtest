#!/usr/bin/env python3
"""
Test script to demonstrate BTST gap-down functionality
"""

import pandas as pd
import datetime as dt
from app.trade_audit import TradeAuditor

def create_gap_down_test_data():
    """Create test data with gap-down scenarios"""
    trades = [
        {
            "Stock": "RELIANCE",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:18",
            "Entry Price": 2500.0,
            "Exit Date": "2025-08-02",  # Correct: next day
            "Exit Time": "15:25",       # WRONG: Should be 09:15-09:25
            "Exit Price": 2475.0,       # Gap-down: -1% (below 0.5% threshold)
            "Return %": -1.0,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapDown
        },
        {
            "Stock": "TCS",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:22",
            "Entry Price": 3000.0,
            "Exit Date": "2025-08-02",  # Correct: next day
            "Exit Time": "15:30",       # WRONG: Should be 09:15-09:25
            "Exit Price": 3030.0,       # Gap-up: +1% (above 0.5% threshold)
            "Return %": 1.0,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapUp
        },
        {
            "Stock": "INFY",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:25",
            "Entry Price": 1500.0,
            "Exit Date": "2025-08-02",  # Correct: next day
            "Exit Time": "15:25",       # WRONG: Should be 09:15-09:25
            "Exit Price": 1500.0,       # No gap: exactly same price
            "Return %": 0.0,
            "Exit Reason": "Time"       # WRONG: Should be BTST_NoGap
        },
        {
            "Stock": "HDFC",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:20",
            "Entry Price": 2000.0,
            "Exit Date": "2025-08-02",  # Correct: next day
            "Exit Time": "15:30",       # WRONG: Should be 09:15-09:25
            "Exit Price": 1980.0,       # Gap-down: -1% (below 0.5% threshold)
            "Return %": -1.0,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapDown
        },
        {
            "Stock": "WIPRO",
            "Entry Date": "2025-08-01",
            "Entry Time": "15:15",
            "Entry Price": 400.0,
            "Exit Date": "2025-08-02",  # Correct: next day
            "Exit Time": "15:25",       # WRONG: Should be 09:15-09:25
            "Exit Price": 404.0,        # Gap-up: +1% (above 0.5% threshold)
            "Return %": 1.0,
            "Exit Reason": "Time"       # WRONG: Should be BTST_GapUp
        }
    ]
    return pd.DataFrame(trades)

def main():
    """Main test function"""
    print("🔍 BTST Gap-Down Audit System Test")
    print("=" * 50)
    
    # Create test data with gap scenarios
    df = create_gap_down_test_data()
    print(f"\n📊 Original BTST Trades (with gap scenarios):")
    print(df[['Stock', 'Entry Time', 'Exit Time', 'Exit Price', 'Return %', 'Exit Reason']].to_string(index=False))
    
    # Initialize auditor
    auditor = TradeAuditor(timezone="Asia/Kolkata")
    
    # Run audit
    print(f"\n🔍 Running Enhanced BTST Gap-Down Audit...")
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
    
    print(f"\n✅ Corrected BTST Trades with Gap Logic:")
    corrected_display = audited_df[['Stock', 'Strategy_Type', 'Exit Time', 'Exit Reason', 'Return %', 'Audit_Status']].copy()
    print(corrected_display.to_string(index=False))
    
    print(f"\n🔧 Gap-Down Logic Applied:")
    print("• Gap-Up Detection: Exit at 09:15 if open > entry * 1.005 (+0.5%)")
    print("• Gap-Down Detection: Exit at 09:15 if open < entry * 0.995 (-0.5%)")
    print("• No Gap: Exit at 09:25 at entry price if no significant gap")
    print("• Cut Losses Quickly: Gap-down exits immediately at open")
    print("• Capture Profits: Gap-up exits immediately at open")
    
    print(f"\n🎯 BTST Gap Strategy Rules:")
    print("• Entry: 15:15-15:30 PM (near market close)")
    print("• Exit: 09:15-09:25 AM next day (market open)")
    print("• Gap-Up: Exit at 09:15 with profit (BTST_GapUp)")
    print("• Gap-Down: Exit at 09:15 with loss (BTST_GapDown)")
    print("• No Gap: Exit at 09:25 at entry price (BTST_NoGap)")
    print("• Threshold: 0.5% for gap detection")
    print("• Risk Management: Cut losses quickly on gap-down")

if __name__ == "__main__":
    main()
