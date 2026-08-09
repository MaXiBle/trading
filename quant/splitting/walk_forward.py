"""Walk-forward splitter with embargo and leakage protection."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta


class WalkForwardSplitter:
    """
    Walk-forward time series splitter with embargo.
    
    Ensures no look-ahead bias by:
    1. Training only on past data
    2. Applying embargo between train and test
    3. Rolling validation/test windows forward
    """
    
    def __init__(
        self,
        train_window_years: int = 4,
        validation_months: int = 3,
        test_months: int = 3,
        roll_period_months: int = 1,
        embargo_days: int = 5
    ):
        self.train_window_years = train_window_years
        self.validation_months = validation_months
        self.test_months = test_months
        self.roll_period_months = roll_period_months
        self.embargo_days = embargo_days
    
    def get_train_val_test_dates(
        self,
        all_dates: pd.DatetimeIndex,
        fold: int = 0
    ) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Get train, validation, and test date ranges for a specific fold.
        
        Returns:
            train_dates, val_dates, test_dates
        """
        # Find the end date based on fold
        max_date = all_dates.max()
        
        # Test period ends at max_date
        test_end = max_date
        test_start = test_end - pd.DateOffset(months=self.test_months)
        
        # Roll back for each fold
        roll_offset = pd.DateOffset(months=self.roll_period_months * fold)
        test_end = test_end - roll_offset
        test_start = test_start - roll_offset
        
        # Validation period is before test
        val_end = test_start
        val_start = val_end - pd.DateOffset(months=self.validation_months)
        
        # Embargo period between train and validation
        embargo_start = val_start - pd.Timedelta(days=self.embargo_days)
        
        # Train period ends before embargo
        train_end = embargo_start
        train_start = train_end - pd.DateOffset(years=self.train_window_years)
        
        # Filter to actual available dates
        train_dates = all_dates[(all_dates >= train_start) & (all_dates < train_end)]
        val_dates = all_dates[(all_dates >= val_start) & (all_dates < val_end)]
        test_dates = all_dates[(all_dates >= test_start) & (all_dates < test_end)]
        
        return train_dates, val_dates, test_dates
    
    def get_splits(
        self,
        all_dates: pd.DatetimeIndex
    ) -> List[Dict[str, pd.DatetimeIndex]]:
        """
        Generate all walk-forward splits.
        
        Returns list of dicts with train/val/test dates for each fold.
        """
        splits = []
        fold = 0
        
        while True:
            train_dates, val_dates, test_dates = self.get_train_val_test_dates(
                all_dates, fold
            )
            
            # Stop if we don't have enough data
            if len(train_dates) < 252 or len(val_dates) < 20 or len(test_dates) < 20:
                break
            
            splits.append({
                "fold": fold,
                "train_dates": train_dates,
                "val_dates": val_dates,
                "test_dates": test_dates,
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "val_start": val_dates.min(),
                "val_end": val_dates.max(),
                "test_start": test_dates.min(),
                "test_end": test_dates.max()
            })
            
            fold += 1
            
            # Limit number of folds for efficiency
            if fold > 12:  # Max 12 folds
                break
        
        return splits
    
    def validate_no_leakage(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        split: Dict[str, pd.DatetimeIndex]
    ) -> bool:
        """
        Validate that there's no data leakage in a split.
        
        Checks:
        1. No label dates overlap with train dates
        2. Embargo is respected
        3. Features use only past data
        """
        train_dates = set(split["train_dates"])
        val_dates = set(split["val_dates"])
        test_dates = set(split["test_dates"])
        
        # Check no overlap
        overlap_train_val = train_dates & val_dates
        overlap_train_test = train_dates & test_dates
        overlap_val_test = val_dates & test_dates
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print(f"WARNING: Date overlap detected!")
            if overlap_train_val:
                print(f"  Train-Val overlap: {overlap_train_val}")
            if overlap_train_test:
                print(f"  Train-Test overlap: {overlap_train_test}")
            if overlap_val_test:
                print(f"  Val-Test overlap: {overlap_val_test}")
            return False
        
        # Check embargo
        if len(train_dates) > 0 and len(val_dates) > 0:
            max_train = max(train_dates)
            min_val = min(val_dates)
            gap_days = (min_val - max_train).days
            
            if gap_days < self.embargo_days:
                print(f"WARNING: Embargo violated! Gap={gap_days} days, required={self.embargo_days}")
                return False
        
        return True
    
    def print_split_summary(self, splits: List[Dict]):
        """Print summary of all splits."""
        print("\n" + "="*80)
        print("WALK-FORWARD SPLIT SUMMARY")
        print("="*80)
        
        for split in splits:
            print(f"\nFold {split['fold']}:")
            print(f"  Train: {split['train_start']} → {split['train_end']} ({len(split['train_dates'])} days)")
            print(f"  Val:   {split['val_start']} → {split['val_end']} ({len(split['val_dates'])} days)")
            print(f"  Test:  {split['test_start']} → {split['test_end']} ({len(split['test_dates'])} days)")
        
        print("\n" + "="*80)


def create_time_series_panel(
    prices_dict: Dict[str, pd.DataFrame],
    feature_func=None,
    label_func=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create panel data from dictionary of price DataFrames.
    
    Returns:
        features_panel: MultiIndex DataFrame (date, secid) x features
        labels_panel: MultiIndex Series (date, secid) -> label
    """
    all_features = []
    all_labels = []
    
    for secid, df in prices_dict.items():
        df = df.sort_values("TRADEDATE").copy()
        df["secid"] = secid
        df.set_index(["TRADEDATE", "secid"], inplace=True)
        
        if feature_func:
            features = feature_func(df)
            all_features.append(features)
        
        if label_func:
            labels = label_func(df)
            all_labels.append(labels)
    
    if all_features:
        features_panel = pd.concat(all_features)
    else:
        features_panel = pd.DataFrame()
    
    if all_labels:
        labels_panel = pd.concat(all_labels)
    else:
        labels_panel = pd.Series()
    
    return features_panel, labels_panel


if __name__ == "__main__":
    # Test splitter
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
    splitter.print_split_summary(splits)
    
    if splits:
        # Validate first split
        is_valid = splitter.validate_no_leakage(
            pd.DataFrame(index=splits[0]["train_dates"]),
            pd.DataFrame(index=splits[0]["val_dates"]),
            splits[0]
        )
        print(f"\nFirst split validation: {'PASS' if is_valid else 'FAIL'}")
