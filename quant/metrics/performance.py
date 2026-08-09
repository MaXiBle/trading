"""Performance metrics and reporting for backtest results."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


class MetricsCalculator:
    """Calculate comprehensive performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.08):
        self.risk_free_rate = risk_free_rate
    
    def calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        
        return (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
    
    def calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        ann_return = returns.mean() * 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) < 2:
            return 0.0
        
        downside_dev = downside_returns.std() * np.sqrt(252)
        
        return (ann_return - self.risk_free_rate) / downside_dev if downside_dev > 0 else 0.0
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_calmar(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio (return / max_drawdown)."""
        ann_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(equity_curve))
        
        return ann_return / max_dd if max_dd > 0 else 0.0
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 10:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) < 10:
            return 0.0
        
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness."""
        if len(returns) < 3:
            return 0.0
        
        return returns.skew()
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis."""
        if len(returns) < 4:
            return 0.0
        
        return returns.kurtosis()
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) < 1:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    def calculate_best_worst(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate best and worst returns."""
        if len(returns) < 1:
            return {"best": 0.0, "worst": 0.0}
        
        return {
            "best": returns.max(),
            "worst": returns.min()
        }
    
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 63
    ) -> pd.DataFrame:
        """Calculate rolling Sharpe and volatility."""
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: self.calculate_sharpe(x)
        )
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        rolling_ret = returns.rolling(window).mean() * 252
        
        return pd.DataFrame({
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_return": rolling_ret
        })


class RankICCalculator:
    """Calculate Rank IC metrics for cross-sectional models."""
    
    @staticmethod
    def calculate_rank_ic(
        predictions: pd.Series,
        actuals: pd.Series
    ) -> float:
        """
        Calculate Spearman rank correlation between predictions and actuals.
        
        Both series should be indexed by (date, secid) or aligned.
        """
        if len(predictions) != len(actuals) or len(predictions) < 5:
            return 0.0
        
        # Remove NaN values
        mask = ~(predictions.isna() | actuals.isna())
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) < 5:
            return 0.0
        
        # Calculate Spearman correlation
        corr, _ = stats.spearmanr(pred_clean, actual_clean)
        
        return corr if not np.isnan(corr) else 0.0
    
    @staticmethod
    def calculate_rank_ic_by_date(
        predictions: pd.Series,
        actuals: pd.Series
    ) -> pd.Series:
        """Calculate Rank IC for each date."""
        # Assume MultiIndex with date as first level
        dates = predictions.index.get_level_values("TRADEDATE").unique()
        
        rank_ics = []
        for date in dates:
            try:
                pred_date = predictions.xs(date, level="TRADEDATE")
                actual_date = actuals.xs(date, level="TRADEDATE")
                
                ic = RankICCalculator.calculate_rank_ic(pred_date, actual_date)
                rank_ics.append((date, ic))
            except KeyError:
                continue
        
        return pd.Series(dict(rank_ics))
    
    @staticmethod
    def calculate_ic_statistics(rank_ics: pd.Series) -> Dict[str, float]:
        """Calculate IC statistics."""
        if len(rank_ics) < 5:
            return {}
        
        mean_ic = rank_ics.mean()
        std_ic = rank_ics.std()
        t_stat = mean_ic / (std_ic / np.sqrt(len(rank_ics)))
        
        # Information coefficient
        ir = mean_ic * np.sqrt(len(rank_ics))
        
        return {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_statistic": t_stat,
            "information_ratio": ir,
            "positive_ic_ratio": (rank_ics > 0).mean(),
            "n_periods": len(rank_ics)
        }


class QuintileAnalyzer:
    """Analyze performance by quintile groups."""
    
    @staticmethod
    def assign_quintiles(scores: pd.Series) -> pd.Series:
        """Assign stocks to quintiles based on scores."""
        return pd.qcut(scores.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    
    @staticmethod
    def calculate_quintile_returns(
        scores: pd.Series,
        forward_returns: pd.Series,
        n_quintiles: int = 5
    ) -> pd.DataFrame:
        """Calculate average forward return for each quintile."""
        quintiles = QuintileAnalyzer.assign_quintiles(scores)
        
        combined = pd.DataFrame({
            "quintile": quintiles,
            "forward_return": forward_returns
        }).dropna()
        
        quintile_returns = combined.groupby("quintile")["forward_return"].agg([
            "mean", "std", "count"
        ])
        
        # Add spread (Q5 - Q1)
        if 1 in quintile_returns.index and 5 in quintile_returns.index:
            spread_mean = quintile_returns.loc[5, "mean"] - quintile_returns.loc[1, "mean"]
            spread_std = np.sqrt(
                quintile_returns.loc[5, "std"]**2 + 
                quintile_returns.loc[1, "std"]**2
            )
            
            spread_row = pd.Series({
                "mean": spread_mean,
                "std": spread_std,
                "count": min(
                    quintile_returns.loc[5, "count"],
                    quintile_returns.loc[1, "count"]
                )
            })
            
            quintile_returns.loc[6] = spread_row  # 6 = spread
        
        return quintile_returns


def generate_backtest_report(
    equity_curve: pd.DataFrame,
    daily_returns: pd.DataFrame,
    trades_log: pd.DataFrame,
    rank_ics: Optional[pd.Series] = None,
    cost_bps_list: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Generate comprehensive backtest report."""
    
    metrics_calc = MetricsCalculator()
    
    # Basic metrics
    returns_series = daily_returns["return"] if "return" in daily_returns.columns else pd.Series()
    equity_series = equity_curve["equity"] if "equity" in equity_curve.columns else pd.Series()
    
    report = {
        "summary_metrics": {
            "total_return": (equity_series.iloc[-1] / equity_series.iloc[0] - 1) if len(equity_series) > 1 else 0,
            "annualized_return": metrics_calc.calculate_sharpe(returns_series) * 0.08 + 0.08 if len(returns_series) > 1 else 0,
            "volatility_annual": returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0,
            "sharpe_ratio": metrics_calc.calculate_sharpe(returns_series),
            "sortino_ratio": metrics_calc.calculate_sortino(returns_series),
            "max_drawdown": metrics_calc.calculate_max_drawdown(equity_series),
            "calmar_ratio": metrics_calc.calculate_calmar(returns_series, equity_series),
            "win_rate": metrics_calc.calculate_win_rate(returns_series),
            "skewness": metrics_calc.calculate_skewness(returns_series),
            "kurtosis": metrics_calc.calculate_kurtosis(returns_series),
            "var_95": metrics_calc.calculate_var(returns_series),
            "cvar_95": metrics_calc.calculate_cvar(returns_series)
        },
        "turnover_metrics": {
            "avg_turnover": trades_log["turnover"].mean() if "turnover" in trades_log.columns else 0,
            "total_transaction_costs": trades_log["transaction_cost"].sum() if "transaction_cost" in trades_log.columns else 0,
            "n_trades": len(trades_log)
        }
    }
    
    # Rank IC metrics
    if rank_ics is not None and len(rank_ics) > 0:
        report["rank_ic"] = RankICCalculator.calculate_ic_statistics(rank_ics)
    
    # Cost sensitivity analysis
    if cost_bps_list:
        cost_sensitivity = []
        base_costs = trades_log["transaction_cost"].sum() if "transaction_cost" in trades_log.columns else 0
        
        for bps in cost_bps_list:
            scale = bps / 12.0  # Assuming base is 12 bps
            adjusted_costs = base_costs * scale
            cost_sensitivity.append({
                "cost_bps": bps,
                "estimated_total_cost": adjusted_costs,
                "cost_drag_pct": adjusted_costs / equity_series.iloc[0] * 100 if len(equity_series) > 0 else 0
            })
        
        report["cost_sensitivity"] = cost_sensitivity
    
    return report


if __name__ == "__main__":
    # Test metrics calculator
    np.random.seed(42)
    test_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
    test_equity = (1 + test_returns).cumprod() * 1_000_000
    
    calc = MetricsCalculator()
    
    print("\nPerformance Metrics:")
    print(f"  Sharpe: {calc.calculate_sharpe(test_returns):.3f}")
    print(f"  Sortino: {calc.calculate_sortino(test_returns):.3f}")
    print(f"  Max DD: {calc.calculate_max_drawdown(test_equity):.3f}")
    print(f"  Win Rate: {calc.calculate_win_rate(test_returns):.3f}")
    print(f"  Skewness: {calc.calculate_skewness(test_returns):.3f}")
    print(f"  Kurtosis: {calc.calculate_kurtosis(test_returns):.3f}")
