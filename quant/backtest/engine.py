"""Daily backtester with costs, turnover, and constraints."""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class DailyBacktester:
    """
    Event-driven daily backtester for long-only strategies.
    
    Features:
    - Daily rebalance
    - Transaction costs
    - Turnover calculation
    - Position/sector constraints
    - Gross/net exposure tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        cost_bps_one_way: float = 12.0,
        max_position_weight: float = 0.10,
        max_sector_weight: float = 0.30,
        target_gross_exposure: float = 1.0,
        long_only: bool = True
    ):
        self.initial_capital = initial_capital
        self.cost_bps_one_way = cost_bps_one_way
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.target_gross_exposure = target_gross_exposure
        self.long_only = long_only
        
        # State variables
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # secid -> shares
        self.prices: Dict[str, float] = {}  # secid -> current price
        self.sectors: Dict[str, str] = {}  # secid -> sector
        
        # History
        self.equity_curve = []
        self.trades_log = []
        self.daily_returns = []
    
    def apply_constraints(
        self,
        target_weights: pd.Series
    ) -> pd.Series:
        """Apply position and sector constraints to target weights."""
        weights = target_weights.copy()
        
        # Long-only constraint
        if self.long_only:
            weights = weights.clip(lower=0)
        
        # Normalize to target exposure
        current_sum = weights.sum()
        if current_sum > 0:
            weights = weights * (self.target_gross_exposure / current_sum)
        
        # Position limit
        weights = weights.clip(upper=self.max_position_weight)
        
        # Re-normalize after clipping
        current_sum = weights.sum()
        if current_sum > 0:
            weights = weights * (self.target_gross_exposure / current_sum)
        
        # Sector limit (if sector info available)
        if self.sectors:
            sector_groups = weights.to_frame().assign(
                sector=lambda df: [self.sectors.get(secid, "unknown") 
                                   for secid in df.index]
            )
            
            for sector in sector_groups["sector"].unique():
                sector_mask = sector_groups["sector"] == sector
                sector_weights = sector_groups.loc[sector_mask, weights.name]
                
                if sector_weights.sum() > self.max_sector_weight:
                    scale = self.max_sector_weight / sector_weights.sum()
                    weights.loc[sector_mask] *= scale
        
        return weights
    
    def calculate_turnover(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series
    ) -> float:
        """Calculate portfolio turnover."""
        # Turnover = 0.5 * sum(|w_new - w_old|)
        all_tickers = set(old_weights.index) | set(new_weights.index)
        
        total_diff = 0.0
        for ticker in all_tickers:
            w_old = old_weights.get(ticker, 0.0)
            w_new = new_weights.get(ticker, 0.0)
            total_diff += abs(w_new - w_old)
        
        return 0.5 * total_diff
    
    def execute_rebalance(
        self,
        date: str,
        target_weights: pd.Series,
        prices: pd.Series,
        sectors: Optional[Dict[str, str]] = None
    ):
        """Execute portfolio rebalance."""
        # Update sector mapping
        if sectors:
            self.sectors.update(sectors)
        
        # Update current prices
        self.prices = prices.to_dict()
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for secid, shares in self.positions.items():
            if secid in prices:
                portfolio_value += shares * prices[secid]
        
        # Apply constraints
        constrained_weights = self.apply_constraints(target_weights)
        
        # Calculate old weights
        old_weights = pd.Series({
            secid: (shares * self.prices.get(secid, 0)) / portfolio_value
            for secid, shares in self.positions.items()
            if self.prices.get(secid, 0) > 0
        })
        
        # Calculate turnover
        turnover = self.calculate_turnover(old_weights, constrained_weights)
        
        # Calculate transaction costs
        trade_value = portfolio_value * turnover
        transaction_cost = trade_value * (self.cost_bps_one_way / 10000)
        
        # Deduct costs from cash
        self.cash -= transaction_cost
        
        # Log trade
        self.trades_log.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "cost_bps": self.cost_bps_one_way
        })
        
        # Update positions
        self.positions = {}
        for secid, weight in constrained_weights.items():
            if weight > 0 and secid in prices:
                target_value = portfolio_value * weight
                shares = target_value / prices[secid]
                self.positions[secid] = shares
        
        # Record equity
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "equity": portfolio_value,
            "gross_exposure": sum(
                shares * self.prices.get(secid, 0)
                for secid, shares in self.positions.items()
            ) / portfolio_value if portfolio_value > 0 else 0
        })
    
    def update_prices(self, date: str, prices: pd.Series):
        """Update prices and calculate daily return."""
        self.prices = prices.to_dict()
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for secid, shares in self.positions.items():
            if secid in prices:
                portfolio_value += shares * prices[secid]
        
        # Get previous equity
        if self.equity_curve:
            prev_equity = self.equity_curve[-1]["equity"]
            daily_return = (portfolio_value - prev_equity) / prev_equity if prev_equity > 0 else 0
            
            self.daily_returns.append({
                "date": date,
                "return": daily_return,
                "equity": portfolio_value
            })
    
    def get_results(self) -> pd.DataFrame:
        """Get backtest results as DataFrame."""
        equity_df = pd.DataFrame(self.equity_curve)
        returns_df = pd.DataFrame(self.daily_returns)
        trades_df = pd.DataFrame(self.trades_log)
        
        return {
            "equity_curve": equity_df,
            "daily_returns": returns_df,
            "trades_log": trades_df
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.daily_returns:
            return {}
        
        returns_series = pd.Series([r["return"] for r in self.daily_returns])
        
        # Annualized return
        total_return = (1 + returns_series).prod() - 1
        n_years = len(returns_series) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility and Sharpe
        vol_daily = returns_series.std()
        vol_annual = vol_daily * np.sqrt(252)
        sharpe = (ann_return - 0.08) / vol_annual if vol_annual > 0 else 0  # 8% risk-free
        
        # Max drawdown
        equity_series = pd.Series([r["equity"] for r in self.daily_returns])
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()
        
        # Average turnover
        if self.trades_log:
            avg_turnover = pd.Series([t["turnover"] for t in self.trades_log]).mean()
            total_costs = sum(t["transaction_cost"] for t in self.trades_log)
        else:
            avg_turnover = 0
            total_costs = 0
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility_annual": vol_annual,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_turnover": avg_turnover,
            "total_transaction_costs": total_costs,
            "final_equity": equity_series.iloc[-1] if len(equity_series) > 0 else self.initial_capital
        }


def simple_backtest(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 12.0
) -> Dict[str, Any]:
    """
    Simple backtest helper function.
    
    Args:
        signals_df: MultiIndex DataFrame (date, secid) with signal scores
        prices_df: MultiIndex DataFrame (date, secid) with prices
        horizon: Holding period in days
        cost_bps: Transaction cost in basis points
    
    Returns:
        Dictionary with backtest results
    """
    backtester = DailyBacktester(cost_bps_one_way=cost_bps)
    
    dates = sorted(signals_df.index.get_level_values("TRADEDATE").unique())
    
    prev_weights = pd.Series()
    
    for date in dates:
        # Get signals for this date
        date_signals = signals_df.xs(date, level="TRADEDATE")
        
        # Convert signals to weights (top quintile equal weight)
        scores = date_signals.iloc[:, 0] if hasattr(date_signals, 'columns') else date_signals
        threshold = scores.quantile(0.80)
        selected = scores[scores >= threshold].index.tolist()
        
        if not selected:
            continue
        
        target_weights = pd.Series(
            {secid: 1.0 / len(selected) for secid in selected}
        )
        
        # Get prices for this date
        try:
            date_prices = prices_df.xs(date, level="TRADEDATE")["CLOSE"]
        except KeyError:
            continue
        
        # Execute rebalance
        backtester.execute_rebalance(
            date=date,
            target_weights=target_weights,
            prices=date_prices
        )
    
    results = backtester.get_results()
    metrics = backtester.calculate_metrics()
    
    return {
        "metrics": metrics,
        "equity_curve": results["equity_curve"],
        "trades_log": results["trades_log"]
    }


if __name__ == "__main__":
    # Test backtester
    backtester = DailyBacktester(initial_capital=1_000_000, cost_bps_one_way=12)
    
    # Simulate simple strategy
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    dates = dates[dates.dayofweek < 5]  # weekdays
    
    tickers = ["SBER", "GAZP", "LKOH"]
    
    for i, date in enumerate(dates[:20]):
        # Simulate prices
        prices = pd.Series({
            "SBER": 300 + i * 0.5,
            "GAZP": 170 + i * 0.3,
            "LKOH": 7000 + i * 10
        })
        
        # Simple equal weight
        target_weights = pd.Series({ticker: 0.33 for ticker in tickers})
        
        backtester.execute_rebalance(
            date=str(date.date()),
            target_weights=target_weights,
            prices=prices
        )
    
    metrics = backtester.calculate_metrics()
    print("\nBacktest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
