"""Unit tests for platform components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.moex_iss_client import MOEXISSClient, MOEXISSConfig
from data_quality.checks import DataQualityChecker, get_moex_trading_calendar
from corporate_actions.handler import CorporateActionsHandler, CorporateAction
from splitting.walk_forward import WalkForwardSplitter
from backtest.engine import DailyBacktester
from metrics.performance import MetricsCalculator, RankICCalculator, QuintileAnalyzer


class TestMOEXISSClient:
    """Tests for MOEX ISS client."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = MOEXISSConfig()
        assert config.base_url == "https://iss.moex.com/iss"
        assert "TQBR" in config.boards
        assert config.interval == 24
    
    def test_validate_dataframe_valid(self):
        """Test validation with valid data."""
        df = pd.DataFrame({
            "TRADEDATE": ["2024-01-10", "2024-01-11"],
            "OPEN": [100, 102],
            "LOW": [99, 101],
            "HIGH": [102, 103],
            "CLOSE": [101, 102],
            "VOLUME": [1000000, 1200000],
            "VALUE": [101000000, 122400000],
            "WAPRICE": [100.5, 101.8]
        })
        
        is_valid, issues = MOEXISSClient.validate_dataframe(df)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_dataframe_invalid(self):
        """Test validation with invalid data."""
        df = pd.DataFrame({
            "TRADEDATE": ["2024-01-10", "2024-01-10"],  # duplicate
            "OPEN": [100, -50],  # negative
            "LOW": [99, 101],
            "HIGH": [102, 103],
            "CLOSE": [101, 102],
            "VOLUME": [-1000000, 1200000],  # negative volume
            "VALUE": [101000000, 122400000],
            "WAPRICE": [100.5, 101.8]
        })
        
        is_valid, issues = MOEXISSClient.validate_dataframe(df)
        assert is_valid is False
        assert len(issues) > 0


class TestDataQualityChecker:
    """Tests for data quality checks."""
    
    def test_check_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            "TRADEDATE": ["2024-01-10", "2024-01-11", "2024-01-11"],
            "CLOSE": [100, 101, 102]
        })
        
        checker = DataQualityChecker()
        issues = checker.check_duplicates(df, "TEST")
        assert len(issues) > 0
    
    def test_check_no_duplicates(self):
        """Test with no duplicates."""
        df = pd.DataFrame({
            "TRADEDATE": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "CLOSE": [100, 101, 102]
        })
        
        checker = DataQualityChecker()
        issues = checker.check_duplicates(df, "TEST")
        assert len(issues) == 0
    
    def test_price_consistency(self):
        """Test price consistency check."""
        df = pd.DataFrame({
            "TRADEDATE": ["2024-01-10"],
            "OPEN": [105],  # Outside HIGH
            "LOW": [99],
            "HIGH": [102],
            "CLOSE": [101]
        })
        
        checker = DataQualityChecker()
        issues = checker.check_price_consistency(df, "TEST")
        assert len(issues) > 0


class TestCorporateActionsHandler:
    """Tests for corporate actions handler."""
    
    def test_add_action(self, tmp_path):
        """Test adding corporate action."""
        handler = CorporateActionsHandler(data_dir=str(tmp_path))
        
        action = CorporateAction(
            secid="SBER",
            ex_date="2024-06-20",
            action_type="dividend",
            value=33.0,
            source="test"
        )
        
        handler.add_action(action)
        assert len(handler.actions_df) == 1
        assert handler.actions_df.iloc[0]["secid"] == "SBER"
    
    def test_adjustment_factor_calculation(self, tmp_path):
        """Test adjustment factor calculation."""
        handler = CorporateActionsHandler(data_dir=str(tmp_path))
        
        prices = pd.DataFrame({
            "TRADEDATE": ["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15"],
            "CLOSE": [100, 101, 99, 100]
        })
        
        result = handler.calculate_adjustment_factors(prices, "SBER")
        assert "adj_factor" in result.columns
        assert (result["adj_factor"] == 1.0).all()  # No actions, factor should be 1


class TestWalkForwardSplitter:
    """Tests for walk-forward splitter."""
    
    def test_split_creation(self):
        """Test split creation."""
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]  # weekdays only
        
        splitter = WalkForwardSplitter(
            train_window_years=3,
            validation_months=3,
            test_months=3,
            roll_period_months=3,
            embargo_days=5
        )
        
        splits = splitter.get_splits(dates)
        assert len(splits) > 0
        
        # Check first split has all required fields
        first_split = splits[0]
        assert "fold" in first_split
        assert "train_dates" in first_split
        assert "val_dates" in first_split
        assert "test_dates" in first_split
    
    def test_no_overlap(self):
        """Test that train/val/test don't overlap."""
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]
        
        splitter = WalkForwardSplitter(embargo_days=5)
        splits = splitter.get_splits(dates)
        
        for split in splits:
            train_set = set(split["train_dates"])
            val_set = set(split["val_dates"])
            test_set = set(split["test_dates"])
            
            assert len(train_set & val_set) == 0
            assert len(train_set & test_set) == 0
            assert len(val_set & test_set) == 0
    
    def test_embargo_respected(self):
        """Test that embargo period is respected."""
        dates = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        dates = dates[dates.dayofweek < 5]
        
        splitter = WalkForwardSplitter(embargo_days=5)
        splits = splitter.get_splits(dates)
        
        if splits:
            split = splits[0]
            if len(split["train_dates"]) > 0 and len(split["val_dates"]) > 0:
                max_train = max(split["train_dates"])
                min_val = min(split["val_dates"])
                gap_days = (min_val - max_train).days
                
                assert gap_days >= splitter.embargo_days


class TestDailyBacktester:
    """Tests for daily backtester."""
    
    def test_initialization(self):
        """Test backtester initialization."""
        backtester = DailyBacktester(
            initial_capital=1_000_000,
            cost_bps_one_way=12,
            max_position_weight=0.10
        )
        
        assert backtester.initial_capital == 1_000_000
        assert backtester.cost_bps_one_way == 12
        assert backtester.max_position_weight == 0.10
        assert backtester.cash == 1_000_000
    
    def test_apply_constraints(self):
        """Test position constraints."""
        backtester = DailyBacktester(max_position_weight=0.50)  # Relaxed for test
        
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        constrained = backtester.apply_constraints(weights)
        
        # All weights should be <= max_position_weight after constraint
        assert (constrained <= 0.50).all()
        # Sum should be close to 1.0
        assert abs(constrained.sum() - 1.0) < 0.01
    
    def test_turnover_calculation(self):
        """Test turnover calculation."""
        backtester = DailyBacktester()
        
        old_weights = pd.Series({"A": 0.5, "B": 0.5})
        new_weights = pd.Series({"A": 0.5, "B": 0.5})
        
        turnover = backtester.calculate_turnover(old_weights, new_weights)
        assert turnover == 0.0
        
        new_weights = pd.Series({"A": 1.0, "B": 0.0})
        turnover = backtester.calculate_turnover(old_weights, new_weights)
        assert turnover == 0.5  # 50% turnover


class TestMetricsCalculator:
    """Tests for metrics calculator."""
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator(risk_free_rate=0.0)
        
        # Positive constant returns should give high Sharpe (no volatility)
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)  # Small positive mean
        sharpe = calc.calculate_sharpe(returns)
        assert sharpe > 0  # Should have positive Sharpe with positive mean
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        
        # Monotonically increasing equity
        equity = pd.Series([100, 110, 120, 130])
        dd = calc.calculate_max_drawdown(equity)
        assert dd == 0.0
        
        # Equity with drawdown
        equity = pd.Series([100, 110, 100, 90, 95])
        dd = calc.calculate_max_drawdown(equity)
        assert dd < 0
        assert abs(dd) > 0.1  # More than 10% drawdown


class TestRankICCalculator:
    """Tests for Rank IC calculator."""
    
    def test_perfect_correlation(self):
        """Test perfect rank correlation."""
        predictions = pd.Series([1, 2, 3, 4, 5])
        actuals = pd.Series([1, 2, 3, 4, 5])
        
        ic = RankICCalculator.calculate_rank_ic(predictions, actuals)
        assert abs(ic - 1.0) < 0.01
    
    def test_negative_correlation(self):
        """Test negative rank correlation."""
        predictions = pd.Series([1, 2, 3, 4, 5])
        actuals = pd.Series([5, 4, 3, 2, 1])
        
        ic = RankICCalculator.calculate_rank_ic(predictions, actuals)
        assert abs(ic - (-1.0)) < 0.01
    
    def test_zero_correlation(self):
        """Test zero correlation with random data."""
        np.random.seed(42)
        predictions = pd.Series(np.random.randn(100))
        actuals = pd.Series(np.random.randn(100))
        
        ic = RankICCalculator.calculate_rank_ic(predictions, actuals)
        assert abs(ic) < 0.3  # Should be close to 0


class TestQuintileAnalyzer:
    """Tests for quintile analyzer."""
    
    def test_quintile_assignment(self):
        """Test quintile assignment."""
        scores = pd.Series(range(1, 101))  # 1 to 100
        quintiles = QuintileAnalyzer.assign_quintiles(scores)
        
        assert len(quintiles) == 100
        assert set(quintiles.unique()) == {1, 2, 3, 4, 5}
        # Each quintile should have ~20 stocks
        for q in range(1, 6):
            count = (quintiles == q).sum()
            assert 15 <= count <= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
