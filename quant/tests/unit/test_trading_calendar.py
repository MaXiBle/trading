"""Tests for trading calendar."""

import pytest
import pandas as pd
from datetime import datetime

from src.data_ingestion.trading_calendar import TradingCalendar, create_calendar


class TestTradingCalendar:
    """Test cases for TradingCalendar class."""
    
    @pytest.fixture
    def calendar(self):
        """Create a test calendar."""
        return TradingCalendar(
            min_date="2020-01-01",
            max_date="2023-12-31",
        )
    
    def test_calendar_creation(self, calendar):
        """Test calendar is created successfully."""
        assert calendar.min_date == pd.Timestamp("2020-01-01")
        assert calendar.max_date == pd.Timestamp("2023-12-31")
        assert len(calendar.trading_days) > 0
    
    def test_is_trading_day_weekend(self, calendar):
        """Test that weekends are not trading days."""
        # Saturday Jan 4, 2020
        assert calendar.is_trading_day("2020-01-04") is False
        # Sunday Jan 5, 2020
        assert calendar.is_trading_day("2020-01-05") is False
    
    def test_is_trading_day_weekday(self, calendar):
        """Test that weekdays are trading days (unless holiday)."""
        # Monday Jan 6, 2020
        assert calendar.is_trading_day("2020-01-06") is True
    
    def test_is_trading_day_holiday(self, calendar):
        """Test that holidays are not trading days."""
        # New Year's Day
        assert calendar.is_trading_day("2020-01-01") is False
        # Victory Day
        assert calendar.is_trading_day("2020-05-09") is False
    
    def test_next_trading_day(self, calendar):
        """Test next trading day calculation."""
        # From Friday, next trading day should be Monday (or Tuesday if holiday)
        next_day = calendar.next_trading_day("2020-01-03")
        assert next_day.dayofweek < 5  # Not weekend
    
    def test_prev_trading_day(self, calendar):
        """Test previous trading day calculation."""
        prev_day = calendar.prev_trading_day("2020-01-06")
        assert prev_day.dayofweek < 5  # Not weekend
    
    def test_add_trading_days_positive(self, calendar):
        """Test adding positive trading days."""
        result = calendar.add_trading_days("2020-01-06", 5)
        assert result >= pd.Timestamp("2020-01-06")
    
    def test_add_trading_days_negative(self, calendar):
        """Test adding negative trading days."""
        result = calendar.add_trading_days("2020-01-17", -5)
        assert result <= pd.Timestamp("2020-01-17")
    
    def test_get_trading_days_between(self, calendar):
        """Test getting trading days between dates."""
        days = calendar.get_trading_days_between("2020-01-01", "2020-01-31")
        assert len(days) > 0
        assert all(d.dayofweek < 5 for d in days)  # All weekdays
    
    def test_count_trading_days(self, calendar):
        """Test counting trading days."""
        count = calendar.count_trading_days("2020-01-01", "2020-01-31")
        assert count > 0
        assert count < 31  # Less than total days in month
    
    def test_to_dataframe(self, calendar):
        """Test converting to DataFrame."""
        df = calendar.to_dataframe()
        assert "trade_date" in df.columns
        assert "is_trading_day" in df.columns
        assert len(df) == len(calendar.trading_days)


class TestCreateCalendar:
    """Test cases for create_calendar function."""
    
    def test_create_default_calendar(self):
        """Test creating calendar with defaults."""
        cal = create_calendar()
        assert cal is not None
        assert len(cal.trading_days) > 0
    
    def test_create_calendar_with_custom_dates(self):
        """Test creating calendar with custom date range."""
        cal = create_calendar(
            min_date="2022-01-01",
            max_date="2022-12-31",
        )
        assert cal.min_date == pd.Timestamp("2022-01-01")
        assert cal.max_date == pd.Timestamp("2022-12-31")
    
    def test_create_calendar_with_nonexistent_holidays_file(self):
        """Test creating calendar with missing holidays file."""
        cal = create_calendar(
            min_date="2022-01-01",
            max_date="2022-12-31",
            holidays_path="/nonexistent/path/holidays.csv",
        )
        assert cal is not None  # Should use default holidays
