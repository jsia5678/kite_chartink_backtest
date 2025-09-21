# Trade Audit System

## Overview

The Trade Audit System is a self-auditing tool that ensures trades follow defined strategy rules. It automatically detects strategy types, validates trade parameters, and corrects violations to maintain consistency and accuracy in backtesting results.

## Features

### 🔍 Strategy Type Detection
- **Intraday**: Same-day trades, no overnight carry
- **BTST (Buy Today Sell Tomorrow)**: Entry near close, exit at next day open
- **Swing**: Multi-day holding, exit at target/SL only

### ✅ Validation Rules

#### Entry & Exit Timing
- Validates entry/exit times against strategy-specific allowed windows
- Ensures trades follow proper timing patterns for each strategy type

#### Holding Period
- Checks holding duration against maximum allowed days
- Prevents overnight carry violations for intraday strategies

#### Exit Reasons
- Validates exit reasons match actual price movements
- Ensures TP/SL exits correspond to actual profit/loss

#### Return Calculation
- Recalculates returns based on corrected entry/exit prices
- Ensures accurate performance metrics

### 🔧 Automatic Corrections

#### BTST Strategy
- Corrects exit to next day open (09:15)
- Sets proper exit reason as "BTST_Open"

#### Intraday Strategy
- Ensures same-day exit at market close (15:30)
- Prevents overnight carry violations

#### Swing Strategy
- Allows multi-day holding within limits
- Validates exit conditions (TP/SL only)

## Usage

### Basic Usage

```python
from app.trade_audit import TradeAuditor

# Initialize auditor
auditor = TradeAuditor(timezone="Asia/Kolkata")

# Audit trade log
audited_df, audit_summary = auditor.audit_trade_log(df, num_days_param=5)

# Access audit results
print(f"Pass Rate: {audit_summary['pass_rate']:.1f}%")
print(f"Strategy Distribution: {audit_summary['strategy_distribution']}")
```

### Integration with Backtest Engine

The audit system is automatically integrated into the backtest workflow:

```python
# In run_backtest_from_csv function
if enable_audit:
    auditor = TradeAuditor(timezone=timezone_name)
    df_audited, audit_summary = auditor.audit_trade_log(df, num_days)
    df_audited.attrs['audit_summary'] = audit_summary
    df = df_audited
```

## Audit Results

### Audit Summary
- **Total Trades**: Number of trades processed
- **Pass Rate**: Percentage of trades that passed all validations
- **Strategy Distribution**: Count of trades by strategy type
- **Common Violations**: Most frequent rule violations

### Trade-Level Results
- **Strategy_Type**: Detected strategy (intraday/btst/swing)
- **Audit_Status**: PASS/FAIL/ERROR
- **Violations**: List of specific rule violations
- **Corrected Trade**: Trade data after applying corrections

## Strategy Rules

### Intraday Strategy
```python
StrategyRules(
    max_holding_days=1,
    allowed_entry_times=[09:15, 09:30, 10:00, 10:30, 11:00],
    allowed_exit_times=[15:15, 15:30],
    overnight_allowed=False,
    exit_at_close_required=True
)
```

### BTST Strategy
```python
StrategyRules(
    max_holding_days=2,
    allowed_entry_times=[15:15, 15:30],
    allowed_exit_times=[09:15, 09:30, 10:00],
    overnight_allowed=True,
    exit_at_close_required=False
)
```

### Swing Strategy
```python
StrategyRules(
    max_holding_days=30,
    allowed_entry_times=[09:15, 09:30, 10:00, 15:15, 15:30],
    allowed_exit_times=[09:15, 09:30, 10:00, 15:15, 15:30],
    overnight_allowed=True,
    exit_at_close_required=False
)
```

## Web Interface

The audit system is integrated into the web interface with:

- **Audit Summary Card**: Shows pass rate, strategy distribution, and common violations
- **Visual Indicators**: Color-coded status for easy identification
- **Detailed Violations**: Lists specific rule violations for failed trades

## Testing

Run the test script to see the audit system in action:

```bash
python test_audit.py
```

This will demonstrate:
- Strategy type detection
- Rule validation
- Automatic corrections
- Audit reporting

## Benefits

1. **Consistency**: Ensures all trades follow defined strategy rules
2. **Accuracy**: Corrects timing and exit condition violations
3. **Transparency**: Provides detailed audit reports
4. **Automation**: Self-auditing without manual intervention
5. **Reliability**: Maintains data integrity in backtesting results

## Configuration

The audit system can be configured by:

- Modifying strategy rules in `STRATEGY_RULES` dictionary
- Adjusting timezone settings
- Customizing validation logic
- Adding new strategy types

## Error Handling

The system gracefully handles:
- Missing trade data
- Invalid timestamps
- Price calculation errors
- Strategy detection failures

Failed audits are logged with detailed error messages while allowing the backtest to continue with original data.
