"""Backtest engine for the research platform."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtester."""
    initial_capital: float = 1_000_000
    long_only: bool = True
    gross_exposure: float = 1.0
    max_position_pct: float = 0.10
    rebalance_frequency_days: int = 5
    cost_bps: float = 12.0  # one-way


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    portfolio_returns: pd.Series
    portfolio_values: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    
    def summary(self) -> str:
        """Generate summary statistics."""
        lines = [
            "Backtest Summary",
            "=" * 40,
            f"Total Return: {self.metrics.get('total_return', 0):.2%}",
            f"Annualized Return: {self.metrics.get('annualized_return', 0):.2%}",
            f"Volatility (Ann.): {self.metrics.get('volatility_annual', 0):.2%}",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.3f}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 0):.2%}",
            f"Total Trades: {self.metrics.get('total_trades', 0)}",
        ]
        return "\n".join(lines)


class Backtester:
    """Simple event-driven backtester for long-only strategies."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize backtester.
        
        Args:
            config: Backtest configuration.
        """
        self.config = config or BacktestConfig()
        self.results: Optional[BacktestResult] = None
    
    def run(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run backtest.
        
        Args:
            signals: DataFrame of signals (Date × Ticker) with values in [0, 1].
            returns: DataFrame of forward returns (Date × Ticker).
            prices: Optional DataFrame of prices for trade simulation.
            
        Returns:
            BacktestResult with portfolio returns and metrics.
        """
        # Align signals and returns
        common_dates = signals.index.intersection(returns.index)
        signals = signals.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Normalize signals to portfolio weights
        weights = self._normalize_signals(signals)
        
        # Calculate portfolio returns
        portfolio_returns = (weights * returns).sum(axis=1)
        
        # Apply transaction costs
        if self.config.cost_bps > 0:
            turnover = self._calculate_turnover(weights)
            costs = turnover * (self.config.cost_bps / 10000)
            portfolio_returns = portfolio_returns - costs
        
        # Calculate cumulative returns and portfolio values
        cum_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = self.config.initial_capital * cum_returns
        
        # Generate positions DataFrame
        positions = weights * portfolio_values.values[:, np.newaxis]
        positions = pd.DataFrame(
            positions,
            index=weights.index,
            columns=weights.columns,
        )
        
        # Generate trades DataFrame (simplified)
        trades = self._generate_trades(weights, prices)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns, portfolio_values)
        
        self.results = BacktestResult(
            portfolio_returns=portfolio_returns,
            portfolio_values=portfolio_values,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )
        
        return self.results
    
    def _normalize_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Normalize signals to portfolio weights.
        
        Args:
            signals: Raw signals.
            
        Returns:
            Normalized weights summing to gross_exposure.
        """
        # Zero out negative signals for long-only
        if self.config.long_only:
            signals = signals.clip(lower=0)
        
        # Sum across assets for each date
        signal_sum = signals.sum(axis=1)
        
        # Avoid division by zero
        signal_sum = signal_sum.replace(0, np.nan)
        
        # Normalize to gross exposure
        weights = signals.div(signal_sum, axis=0) * self.config.gross_exposure
        
        # Apply max position constraint with iterative clip and renormalize
        if self.config.max_position_pct < 1.0:
            for _ in range(10):  # Max iterations to converge
                # Clip to max position
                weights = weights.clip(upper=self.config.max_position_pct)
                # Renormalize
                weight_sum = weights.sum(axis=1)
                weight_sum = weight_sum.replace(0, np.nan)
                weights = weights.div(weight_sum, axis=0) * self.config.gross_exposure
                # Check if all weights are within bounds
                if (weights <= self.config.max_position_pct + 1e-9).all().all():
                    break
        
        # Fill NaN with 0 (no position)
        weights = weights.fillna(0)
        
        return weights
    
    def _calculate_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """Calculate turnover between periods.
        
        Turnover = 0.5 * sum(|w_new - w_old|)
        """
        weight_diff = weights.diff().abs().sum(axis=1)
        turnover = 0.5 * weight_diff.fillna(0)
        return turnover
    
    def _generate_trades(
        self,
        weights: pd.DataFrame,
        prices: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate trades DataFrame."""
        # Simplified: just record weight changes as trades
        trades = weights.diff().fillna(0)
        
        if prices is not None:
            # Could add more detailed trade simulation here
            pass
        
        return trades
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        values: pd.Series,
    ) -> dict:
        """Calculate performance metrics."""
        trading_days_per_year = 252
        
        # Total return
        total_return = values.iloc[-1] / values.iloc[0] - 1
        
        # Annualized return
        n_years = len(returns) / trading_days_per_year
        if n_years > 0:
            annualized_return = (values.iloc[-1] / values.iloc[0]) ** (1 / n_years) - 1
        else:
            annualized_return = 0.0
        
        # Volatility
        volatility = returns.std()
        volatility_annual = volatility * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio (assuming 10% risk-free rate)
        risk_free_rate = 0.10
        excess_return = annualized_return - risk_free_rate
        if volatility_annual > 0:
            sharpe_ratio = excess_return / volatility_annual
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Total trades
        total_trades = (returns != 0).sum()
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility_annual": volatility_annual,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": int(total_trades),
            "trading_days": len(returns),
        }


def run_dummy_backtest(config: Optional[BacktestConfig] = None) -> BacktestResult:
    """Run a dummy backtest to verify the platform.
    
    Creates synthetic signals and returns for testing.
    
    Args:
        config: Backtest configuration.
        
    Returns:
        BacktestResult from dummy run.
    """
    np.random.seed(42)
    
    # Generate synthetic data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    tickers = ["TICKER_" + str(i) for i in range(10)]
    
    # Random returns with slight positive drift
    returns_data = np.random.randn(len(dates), len(tickers)) * 0.02 + 0.0005
    returns = pd.DataFrame(returns_data, index=dates, columns=tickers)
    
    # Random signals
    signals_data = np.random.rand(len(dates), len(tickers))
    signals = pd.DataFrame(signals_data, index=dates, columns=tickers)
    
    # Run backtest
    backtester = Backtester(config)
    result = backtester.run(signals, returns)
    
    logger.info("Dummy backtest completed:")
    logger.info(result.summary())
    
    return result
