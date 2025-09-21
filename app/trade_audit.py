"""
Trade Audit Module - Self-auditing trade log system

This module ensures trades follow defined strategy rules by:
1. Detecting strategy type (Intraday, BTST, Swing)
2. Validating entry/exit timings
3. Checking holding period limits
4. Verifying exit reasons match actual triggers
5. Recalculating returns based on corrected prices
6. Flagging rule violations
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import pytz


class StrategyType(Enum):
    """Strategy types with their specific rules"""
    INTRADAY = "intraday"
    BTST = "btst"  # Buy Today Sell Tomorrow
    SWING = "swing"


@dataclass
class StrategyRules:
    """Strategy-specific rules and constraints"""
    strategy_type: StrategyType
    max_holding_days: int
    allowed_entry_times: List[dt.time]
    allowed_exit_times: List[dt.time]
    overnight_allowed: bool
    exit_at_close_required: bool
    description: str


class TradeAuditor:
    """Self-auditing trade log system"""
    
    # Strategy rules definitions
    STRATEGY_RULES = {
        StrategyType.INTRADAY: StrategyRules(
            strategy_type=StrategyType.INTRADAY,
            max_holding_days=1,
            allowed_entry_times=[dt.time(9, 15), dt.time(9, 30), dt.time(10, 0), dt.time(10, 30), dt.time(11, 0)],
            allowed_exit_times=[dt.time(15, 15), dt.time(15, 30)],
            overnight_allowed=False,
            exit_at_close_required=True,
            description="Intraday: No overnight carry, exit same day"
        ),
        StrategyType.BTST: StrategyRules(
            strategy_type=StrategyType.BTST,
            max_holding_days=2,
            allowed_entry_times=[dt.time(15, 15), dt.time(15, 30)],
            allowed_exit_times=[dt.time(9, 15), dt.time(9, 30), dt.time(10, 0)],
            overnight_allowed=True,
            exit_at_close_required=False,
            description="BTST: Buy today, sell tomorrow at open"
        ),
        StrategyType.SWING: StrategyRules(
            strategy_type=StrategyType.SWING,
            max_holding_days=30,  # Configurable
            allowed_entry_times=[dt.time(9, 15), dt.time(9, 30), dt.time(10, 0), dt.time(15, 15), dt.time(15, 30)],
            allowed_exit_times=[dt.time(9, 15), dt.time(9, 30), dt.time(10, 0), dt.time(15, 15), dt.time(15, 30)],
            overnight_allowed=True,
            exit_at_close_required=False,
            description="Swing: Multi-day holding, exit at target/SL only"
        )
    }
    
    def __init__(self, timezone: str = "Asia/Kolkata"):
        self.tz = pytz.timezone(timezone)
    
    def detect_strategy_type(self, entry_time: dt.time, exit_time: dt.time, 
                           holding_days: int, num_days_param: int) -> StrategyType:
        """
        Detect strategy type based on entry/exit patterns and holding period
        """
        # BTST: Entry near close (15:15-15:30), exit near open (9:15-10:00), 1-2 days
        if (self._is_time_in_range(entry_time, dt.time(15, 15), dt.time(15, 30)) and
            self._is_time_in_range(exit_time, dt.time(9, 15), dt.time(10, 0)) and
            holding_days <= 2):
            return StrategyType.BTST
        
        # Intraday: Same day entry/exit or very short holding
        if holding_days <= 1 and not self._is_overnight_carry(entry_time, exit_time):
            return StrategyType.INTRADAY
        
        # Swing: Longer holding period, multiple days allowed
        if holding_days > 1 or num_days_param > 2:
            return StrategyType.SWING
        
        # Default to intraday for ambiguous cases
        return StrategyType.INTRADAY
    
    def _is_time_in_range(self, time_to_check: dt.time, start: dt.time, end: dt.time) -> bool:
        """Check if time is within range (inclusive)"""
        return start <= time_to_check <= end
    
    def _is_overnight_carry(self, entry_time: dt.time, exit_time: dt.time) -> bool:
        """Check if trade involves overnight carry"""
        # If entry is after 15:00 and exit is before 10:00, it's likely overnight
        return (entry_time >= dt.time(15, 0) and exit_time <= dt.time(10, 0))
    
    def audit_trade(self, trade: Dict[str, Any], num_days_param: int) -> Dict[str, Any]:
        """
        Audit a single trade and return corrected version with audit flags
        """
        try:
            # Parse trade data
            entry_date = pd.to_datetime(trade.get("Entry Date")).date()
            entry_time = dt.datetime.strptime(trade.get("Entry Time", "09:15"), "%H:%M").time()
            exit_date = pd.to_datetime(trade.get("Exit Date")).date() if trade.get("Exit Date") else None
            exit_time = dt.datetime.strptime(trade.get("Exit Time", "15:30"), "%H:%M").time() if trade.get("Exit Time") else dt.time(15, 30)
            entry_price = trade.get("Entry Price")
            exit_price = trade.get("Exit Price")
            exit_reason = trade.get("Exit Reason", "Time")
            
            # Calculate holding period
            holding_days = (exit_date - entry_date).days if exit_date else 0
            
            # Detect strategy type
            strategy_type = self.detect_strategy_type(entry_time, exit_time, holding_days, num_days_param)
            strategy_rules = self.STRATEGY_RULES[strategy_type]
            
            # Initialize audit results
            audit_result = {
                "original_trade": trade.copy(),
                "strategy_type": strategy_type.value,
                "strategy_description": strategy_rules.description,
                "violations": [],
                "corrections": {},
                "audit_status": "PASS"
            }
            
            # Validate entry timing
            entry_violation = self._validate_entry_timing(entry_time, strategy_rules)
            if entry_violation:
                audit_result["violations"].append(entry_violation)
                audit_result["audit_status"] = "FAIL"
            
            # Validate exit timing
            exit_violation = self._validate_exit_timing(exit_time, strategy_rules, holding_days)
            if exit_violation:
                audit_result["violations"].append(exit_violation)
                audit_result["audit_status"] = "FAIL"
            
            # Validate holding period
            holding_violation = self._validate_holding_period(holding_days, strategy_rules, num_days_param)
            if holding_violation:
                audit_result["violations"].append(holding_violation)
                audit_result["audit_status"] = "FAIL"
            
            # Validate exit reason
            exit_reason_violation = self._validate_exit_reason(exit_reason, strategy_type, entry_price, exit_price)
            if exit_reason_violation:
                audit_result["violations"].append(exit_reason_violation)
                audit_result["audit_status"] = "FAIL"
            
            # Apply corrections based on strategy type
            corrected_trade = self._apply_strategy_corrections(trade, strategy_type, strategy_rules)
            audit_result["corrected_trade"] = corrected_trade
            
            # Recalculate return if prices were corrected
            if (corrected_trade.get("Entry Price") != entry_price or 
                corrected_trade.get("Exit Price") != exit_price):
                corrected_return = self._calculate_return(
                    corrected_trade.get("Entry Price"), 
                    corrected_trade.get("Exit Price")
                )
                corrected_trade["Return %"] = corrected_return
                audit_result["corrections"]["return_recalculated"] = True
            
            return audit_result
            
        except Exception as e:
            return {
                "original_trade": trade.copy(),
                "strategy_type": "UNKNOWN",
                "violations": [f"Audit error: {str(e)}"],
                "corrections": {},
                "audit_status": "ERROR"
            }
    
    def _validate_entry_timing(self, entry_time: dt.time, rules: StrategyRules) -> Optional[str]:
        """Validate entry timing against strategy rules"""
        if not any(self._is_time_in_range(entry_time, allowed, allowed) 
                  for allowed in rules.allowed_entry_times):
            return f"Entry time {entry_time.strftime('%H:%M')} not allowed for {rules.strategy_type.value} strategy"
        return None
    
    def _validate_exit_timing(self, exit_time: dt.time, rules: StrategyRules, holding_days: int) -> Optional[str]:
        """Validate exit timing against strategy rules"""
        if not any(self._is_time_in_range(exit_time, allowed, allowed) 
                  for allowed in rules.allowed_exit_times):
            return f"Exit time {exit_time.strftime('%H:%M')} not allowed for {rules.strategy_type.value} strategy"
        return None
    
    def _validate_holding_period(self, holding_days: int, rules: StrategyRules, num_days_param: int) -> Optional[str]:
        """Validate holding period against strategy rules"""
        max_allowed = min(rules.max_holding_days, num_days_param)
        if holding_days > max_allowed:
            return f"Holding period {holding_days} days exceeds max allowed {max_allowed} for {rules.strategy_type.value} strategy"
        return None
    
    def _validate_exit_reason(self, exit_reason: str, strategy_type: StrategyType, 
                            entry_price: float, exit_price: float) -> Optional[str]:
        """Validate exit reason matches actual price movement"""
        if not entry_price or not exit_price:
            return None  # Skip validation if prices missing
        
        actual_return = self._calculate_return(entry_price, exit_price)
        
        # Check if exit reason matches actual price movement
        if exit_reason == "TP" and actual_return <= 0:
            return f"Exit reason 'TP' but actual return is {actual_return:.2f}%"
        elif exit_reason == "SL" and actual_return >= 0:
            return f"Exit reason 'SL' but actual return is {actual_return:.2f}%"
        
        return None
    
    def _apply_strategy_corrections(self, trade: Dict[str, Any], strategy_type: StrategyType, 
                                  rules: StrategyRules) -> Dict[str, Any]:
        """Apply strategy-specific corrections to trade"""
        corrected = trade.copy()
        
        if strategy_type == StrategyType.BTST:
            # BTST: Ensure exit is at next day open
            entry_date = pd.to_datetime(trade.get("Entry Date")).date()
            next_day = entry_date + dt.timedelta(days=1)
            corrected["Exit Date"] = next_day.isoformat()
            corrected["Exit Time"] = "09:15"  # Market open
            corrected["Exit Reason"] = "BTST_Open"
            
        elif strategy_type == StrategyType.INTRADAY:
            # Intraday: Ensure same day exit
            corrected["Exit Date"] = trade.get("Entry Date")
            corrected["Exit Time"] = "15:30"  # Market close
            if trade.get("Exit Reason") not in ["TP", "SL"]:
                corrected["Exit Reason"] = "Intraday_Close"
                
        elif strategy_type == StrategyType.SWING:
            # Swing: Allow multi-day but validate exit conditions
            if trade.get("Exit Reason") not in ["TP", "SL"]:
                corrected["Exit Reason"] = "Swing_Time"
        
        return corrected
    
    def _calculate_return(self, entry_price: float, exit_price: float) -> float:
        """Calculate return percentage"""
        if not entry_price or not exit_price:
            return 0.0
        return ((exit_price - entry_price) / entry_price) * 100
    
    def audit_trade_log(self, df: pd.DataFrame, num_days_param: int) -> pd.DataFrame:
        """
        Audit entire trade log and return corrected version with audit flags
        """
        audit_results = []
        corrected_trades = []
        
        for _, trade in df.iterrows():
            trade_dict = trade.to_dict()
            audit_result = self.audit_trade(trade_dict, num_days_param)
            audit_results.append(audit_result)
            corrected_trades.append(audit_result["corrected_trade"])
        
        # Create audit summary
        audit_summary = self._create_audit_summary(audit_results)
        
        # Create corrected dataframe
        corrected_df = pd.DataFrame(corrected_trades)
        
        # Add audit columns
        audit_df = pd.DataFrame(audit_results)
        corrected_df["Strategy_Type"] = audit_df["strategy_type"]
        corrected_df["Audit_Status"] = audit_df["audit_status"]
        corrected_df["Violations"] = audit_df["violations"].apply(lambda x: "; ".join(x) if x else "")
        
        return corrected_df, audit_summary
    
    def _create_audit_summary(self, audit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of audit results"""
        total_trades = len(audit_results)
        passed_trades = sum(1 for r in audit_results if r["audit_status"] == "PASS")
        failed_trades = sum(1 for r in audit_results if r["audit_status"] == "FAIL")
        error_trades = sum(1 for r in audit_results if r["audit_status"] == "ERROR")
        
        strategy_counts = {}
        for result in audit_results:
            strategy = result.get("strategy_type", "UNKNOWN")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_trades": total_trades,
            "passed_trades": passed_trades,
            "failed_trades": failed_trades,
            "error_trades": error_trades,
            "pass_rate": (passed_trades / total_trades * 100) if total_trades > 0 else 0,
            "strategy_distribution": strategy_counts,
            "common_violations": self._get_common_violations(audit_results)
        }
    
    def _get_common_violations(self, audit_results: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Get most common violations"""
        violation_counts = {}
        for result in audit_results:
            for violation in result.get("violations", []):
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
