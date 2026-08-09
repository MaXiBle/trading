"""Data quality checks for MOEX daily bars."""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class DataQualityChecker:
    """Perform data quality checks on daily bars."""
    
    def __init__(self):
        self.issues = []
    
    def check_duplicates(self, df: pd.DataFrame, secid: str) -> List[str]:
        """Check for duplicate dates."""
        issues = []
        if df.duplicated(subset=["TRADEDATE"]).any():
            dup_dates = df[df.duplicated(subset=["TRADEDATE"], keep=False)]["TRADEDATE"].unique()
            issues.append(f"{secid}: Duplicate dates found: {dup_dates[:5]}...")
        return issues
    
    def check_negative_values(self, df: pd.DataFrame, secid: str) -> List[str]:
        """Check for negative volumes and prices."""
        issues = []
        
        if (df["VOLUME"] < 0).any():
            neg_vol_count = (df["VOLUME"] < 0).sum()
            issues.append(f"{secid}: Negative volumes found ({neg_vol_count} rows)")
        
        price_cols = ["OPEN", "LOW", "HIGH", "CLOSE"]
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                neg_count = (df[col] <= 0).sum()
                issues.append(f"{secid}: Non-positive {col} found ({neg_count} rows)")
        
        return issues
    
    def check_price_consistency(self, df: pd.DataFrame, secid: str) -> List[str]:
        """Check that LOW <= CLOSE <= HIGH and LOW <= OPEN <= HIGH."""
        issues = []
        
        # Check LOW <= HIGH
        if (df["LOW"] > df["HIGH"]).any():
            invalid_count = (df["LOW"] > df["HIGH"]).sum()
            issues.append(f"{secid}: LOW > HIGH found ({invalid_count} rows)")
        
        # Check OPEN within range
        if ((df["OPEN"] < df["LOW"]) | (df["OPEN"] > df["HIGH"])).any():
            invalid_count = ((df["OPEN"] < df["LOW"]) | (df["OPEN"] > df["HIGH"])).sum()
            issues.append(f"{secid}: OPEN outside [LOW, HIGH] ({invalid_count} rows)")
        
        # Check CLOSE within range
        if ((df["CLOSE"] < df["LOW"]) | (df["CLOSE"] > df["HIGH"])).any():
            invalid_count = ((df["CLOSE"] < df["LOW"]) | (df["CLOSE"] > df["HIGH"])).sum()
            issues.append(f"{secid}: CLOSE outside [LOW, HIGH] ({invalid_count} rows)")
        
        return issues
    
    def check_volume_anomalies(self, df: pd.DataFrame, secid: str) -> List[str]:
        """Check for volume anomalies."""
        issues = []
        
        if len(df) < 20:
            return issues
        
        # Check for zero volume with non-zero value
        zero_vol_mask = (df["VOLUME"] == 0) & (df["VALUE"] > 0)
        if zero_vol_mask.any():
            issues.append(f"{secid}: Zero volume with positive VALUE found")
        
        # Check for extreme volume spikes (>10x median)
        vol_median = df["VOLUME"].median()
        if vol_median > 0:
            extreme_spike = df["VOLUME"] > (10 * vol_median)
            if extreme_spike.any():
                spike_count = extreme_spike.sum()
                issues.append(f"{secid}: Extreme volume spikes (>10x median) found ({spike_count} rows)")
        
        return issues
    
    def check_return_anomalies(
        self,
        df: pd.DataFrame,
        secid: str,
        max_daily_return: float = 0.50
    ) -> List[str]:
        """Check for extreme daily returns that may indicate data errors."""
        issues = []
        
        if len(df) < 2:
            return issues
        
        df_sorted = df.sort_values("TRADEDATE").copy()
        daily_ret = df_sorted["CLOSE"].pct_change()
        
        extreme_returns = daily_ret.abs() > max_daily_return
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            extreme_dates = df_sorted.loc[extreme_returns, "TRADEDATE"].head(5).tolist()
            issues.append(
                f"{secid}: Extreme daily returns >{max_daily_return*100}% "
                f"found ({extreme_count} rows): {extreme_dates}"
            )
        
        return issues
    
    def check_trading_calendar(
        self,
        df: pd.DataFrame,
        secid: str,
        expected_calendar: pd.DatetimeIndex
    ) -> List[str]:
        """Check for missing trading days."""
        issues = []
        
        df_dates = pd.to_datetime(df["TRADEDATE"])
        df_dates_set = set(df_dates)
        expected_set = set(expected_calendar)
        
        missing_days = expected_set - df_dates_set
        extra_days = df_dates_set - expected_set
        
        if missing_days:
            # Only report if more than 5% of days are missing
            if len(missing_days) > len(expected_calendar) * 0.05:
                issues.append(
                    f"{secid}: Missing {len(missing_days)} expected trading days"
                )
        
        if extra_days:
            issues.append(
                f"{secid}: Found {len(extra_days)} unexpected trading days"
            )
        
        return issues
    
    def check_gap_detection(
        self,
        df: pd.DataFrame,
        secid: str,
        max_gap_days: int = 30
    ) -> List[str]:
        """Detect large gaps in trading dates."""
        issues = []
        
        if len(df) < 2:
            return issues
        
        df_sorted = df.sort_values("TRADEDATE").copy()
        dates = pd.to_datetime(df_sorted["TRADEDATE"])
        date_diffs = dates.diff().dt.days
        
        large_gaps = date_diffs > max_gap_days
        if large_gaps.any():
            gap_count = large_gaps.sum()
            gap_dates = dates[large_gaps].head(5).tolist()
            issues.append(
                f"{secid}: Large trading gaps >{max_gap_days} days found "
                f"({gap_count} occurrences): {gap_dates}"
            )
        
        return issues
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        secid: str,
        expected_calendar: pd.DatetimeIndex = None
    ) -> Dict[str, Any]:
        """Run all data quality checks."""
        all_issues = []
        
        all_issues.extend(self.check_duplicates(df, secid))
        all_issues.extend(self.check_negative_values(df, secid))
        all_issues.extend(self.check_price_consistency(df, secid))
        all_issues.extend(self.check_volume_anomalies(df, secid))
        all_issues.extend(self.check_return_anomalies(df, secid))
        
        if expected_calendar is not None:
            all_issues.extend(self.check_trading_calendar(df, secid, expected_calendar))
        
        all_issues.extend(self.check_gap_detection(df, secid))
        
        return {
            "secid": secid,
            "rows": len(df),
            "date_range": (df["TRADEDATE"].min(), df["TRADEDATE"].max()),
            "valid": len(all_issues) == 0,
            "issue_count": len(all_issues),
            "issues": all_issues
        }


def get_moex_trading_calendar(from_date: str, till_date: str) -> pd.DatetimeIndex:
    """
    Generate approximate MOEX trading calendar.
    
    Note: This is a simplified version. For production, use official MOEX calendar.
    """
    full_range = pd.date_range(from_date, till_date, freq="D")
    
    # Remove weekends
    trading_days = full_range[full_range.dayofweek < 5]
    
    # Remove major Russian holidays (simplified)
    holidays = []
    for year in range(2015, 2027):
        holidays.extend([
            f"{year}-01-01",  # New Year
            f"{year}-01-02",
            f"{year}-01-03",
            f"{year}-01-04",
            f"{year}-01-05",
            f"{year}-01-07",  # Orthodox Christmas
            f"{year}-01-08",
            f"{year}-02-23",  # Defender of the Fatherland
            f"{year}-03-08",  # International Women's Day
            f"{year}-05-01",  # Labor Day
            f"{year}-05-09",  # Victory Day
            f"{year}-06-12",  # Russia Day
            f"{year}-11-04",  # Unity Day
        ])
    
    holiday_dates = pd.to_datetime(holidays)
    trading_days = trading_days[~trading_days.isin(holiday_dates)]
    
    return trading_days


if __name__ == "__main__":
    # Test with sample data
    test_data = {
        "TRADEDATE": ["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15", "2024-01-16"],
        "OPEN": [100, 102, 101, 103, 104],
        "LOW": [99, 101, 100, 102, 103],
        "HIGH": [102, 103, 102, 104, 105],
        "CLOSE": [101, 102, 101, 103, 104],
        "VOLUME": [1000000, 1200000, 800000, 1500000, 1100000],
        "VALUE": [101000000, 122400000, 80800000, 154500000, 114400000],
        "WAPRICE": [100.5, 101.8, 100.9, 102.7, 103.8]
    }
    
    df = pd.DataFrame(test_data)
    checker = DataQualityChecker()
    result = checker.run_all_checks(df, "TEST")
    
    print(f"Test result: valid={result['valid']}, issues={result['issues']}")
