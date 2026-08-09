"""Trading calendar for MOEX."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np


class TradingCalendar:
    """MOEX trading calendar with holiday support.
    
    Provides functions to navigate trading days, skipping weekends
    and exchange holidays.
    """
    
    def __init__(
        self,
        min_date: str = "2010-01-01",
        max_date: str = "2030-12-31",
        holidays: Optional[pd.DatetimeIndex] = None,
    ):
        """Initialize trading calendar.
        
        Args:
            min_date: Minimum date in calendar.
            max_date: Maximum date in calendar.
            holidays: Custom list of exchange holidays.
        """
        self.min_date = pd.Timestamp(min_date)
        self.max_date = pd.Timestamp(max_date)
        
        # Generate all business days (Mon-Fri)
        self._all_business_days = pd.bdate_range(
            start=self.min_date, end=self.max_date, freq="C"
        )
        
        # Default Russian holidays (approximate, should be extended)
        default_holidays = self._generate_default_holidays()
        
        if holidays is not None:
            self.holidays = pd.DatetimeIndex(holidays).unique()
        else:
            self.holidays = default_holidays
        
        # Filter out holidays from business days
        self.trading_days = self._all_business_days[
            ~self._all_business_days.isin(self.holidays)
        ]
        
        # Create lookup set for fast membership testing
        self._trading_days_set = set(self.trading_days)
    
    def _generate_default_holidays(self) -> pd.DatetimeIndex:
        """Generate default Russian market holidays.
        
        Note: This is a simplified list. Production should use
        official MOEX holiday calendar.
        """
        holidays = []
        
        for year in range(2010, 2031):
            # New Year holidays (Jan 1-8 approximately)
            holidays.extend([
                f"{year}-01-01",
                f"{year}-01-02",
                f"{year}-01-03",
                f"{year}-01-04",
                f"{year}-01-05",
                f"{year}-01-06",
                f"{year}-01-07",
                f"{year}-01-08",
            ])
            
            # Defender of the Fatherland Day (Feb 23)
            holidays.append(f"{year}-02-23")
            
            # International Women's Day (Mar 8)
            holidays.append(f"{year}-03-08")
            
            # Labor Day (May 1)
            holidays.append(f"{year}-05-01")
            
            # Victory Day (May 9)
            holidays.append(f"{year}-05-09")
            
            # Russia Day (Jun 12)
            holidays.append(f"{year}-06-12")
            
            # Unity Day (Nov 4)
            holidays.append(f"{year}-11-04")
        
        # Convert to DatetimeIndex and filter to business days only
        holidays_idx = pd.to_datetime(holidays)
        holidays_idx = holidays_idx[holidays_idx >= self.min_date]
        holidays_idx = holidays_idx[holidays_idx <= self.max_date]
        
        return holidays_idx
    
    def is_trading_day(self, date: pd.Timestamp | str) -> bool:
        """Check if a date is a trading day.
        
        Args:
            date: Date to check.
            
        Returns:
            True if trading day, False otherwise.
        """
        ts = pd.Timestamp(date)
        return ts in self._trading_days_set
    
    def next_trading_day(self, date: pd.Timestamp | str, n: int = 1) -> pd.Timestamp:
        """Get the n-th next trading day.
        
        Args:
            date: Starting date.
            n: Number of trading days to advance.
            
        Returns:
            Next trading day timestamp.
        """
        ts = pd.Timestamp(date)
        idx = self.trading_days.searchsorted(ts, side="right")
        target_idx = idx + n - 1
        
        if target_idx >= len(self.trading_days):
            raise ValueError(f"Date {target_idx} is beyond calendar range")
        
        return self.trading_days[target_idx]
    
    def prev_trading_day(self, date: pd.Timestamp | str, n: int = 1) -> pd.Timestamp:
        """Get the n-th previous trading day.
        
        Args:
            date: Starting date.
            n: Number of trading days to go back.
            
        Returns:
            Previous trading day timestamp.
        """
        ts = pd.Timestamp(date)
        idx = self.trading_days.searchsorted(ts, side="left")
        target_idx = idx - n
        
        if target_idx < 0:
            raise ValueError(f"Date {target_idx} is before calendar range")
        
        return self.trading_days[target_idx]
    
    def add_trading_days(self, date: pd.Timestamp | str, n: int) -> pd.Timestamp:
        """Add n trading days to a date.
        
        Args:
            date: Starting date.
            n: Number of trading days to add (can be negative).
            
        Returns:
            Resulting trading day timestamp.
        """
        if n > 0:
            return self.next_trading_day(date, n)
        elif n < 0:
            return self.prev_trading_day(date, -n)
        else:
            return pd.Timestamp(date)
    
    def get_trading_days_between(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> pd.DatetimeIndex:
        """Get all trading days between two dates (inclusive).
        
        Args:
            start: Start date.
            end: End date.
            
        Returns:
            DatetimeIndex of trading days.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        mask = (self.trading_days >= start_ts) & (self.trading_days <= end_ts)
        return self.trading_days[mask]
    
    def count_trading_days(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> int:
        """Count trading days between two dates (inclusive).
        
        Args:
            start: Start date.
            end: End date.
            
        Returns:
            Number of trading days.
        """
        return len(self.get_trading_days_between(start, end))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert calendar to DataFrame.
        
        Returns:
            DataFrame with trade_date and is_trading_day columns.
        """
        df = pd.DataFrame({
            "trade_date": self.trading_days,
            "is_trading_day": True,
            "session_type": "regular",
            "source": "moex_calendar",
        })
        return df


def create_calendar(
    min_date: str = "2010-01-01",
    max_date: str = "2030-12-31",
    holidays_path: Optional[str] = None,
) -> TradingCalendar:
    """Create a trading calendar.
    
    Args:
        min_date: Minimum date in calendar.
        max_date: Maximum date in calendar.
        holidays_path: Optional path to CSV file with holidays.
        
    Returns:
        TradingCalendar instance.
    """
    holidays = None
    
    if holidays_path is not None:
        try:
            holidays_df = pd.read_csv(holidays_path, parse_dates=["date"])
            holidays = pd.to_datetime(holidays_df["date"])
        except FileNotFoundError:
            print(f"Holidays file not found: {holidays_path}, using defaults")
    
    return TradingCalendar(
        min_date=min_date,
        max_date=max_date,
        holidays=holidays,
    )
