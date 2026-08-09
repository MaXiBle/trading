"""Corporate actions handling for MOEX stocks."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from pydantic import BaseModel


class CorporateAction(BaseModel):
    """Represents a corporate action event."""
    secid: str
    ex_date: str
    action_type: str  # dividend, split, merge, rights, delisting
    value: Optional[float] = None
    adjustment_factor: Optional[float] = None
    source: str = "manual"
    published_at: Optional[str] = None


class CorporateActionsHandler:
    """Handle corporate actions and price adjustments."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.actions_file = self.data_dir / "corporate_actions.csv"
        self.actions_df: Optional[pd.DataFrame] = None
    
    def load_actions(self) -> pd.DataFrame:
        """Load corporate actions from file."""
        if self.actions_file.exists():
            self.actions_df = pd.read_csv(self.actions_file)
        else:
            self.actions_df = pd.DataFrame(columns=[
                "secid", "ex_date", "action_type", "value", 
                "adjustment_factor", "source", "published_at"
            ])
        return self.actions_df
    
    def save_actions(self, df: pd.DataFrame):
        """Save corporate actions to file."""
        df.to_csv(self.actions_file, index=False)
        self.actions_df = df
    
    def add_action(self, action: CorporateAction):
        """Add a new corporate action."""
        if self.actions_df is None:
            self.load_actions()
        
        new_row = pd.DataFrame([action.model_dump()])
        self.actions_df = pd.concat([self.actions_df, new_row], ignore_index=True)
        self.save_actions(self.actions_df)
    
    def calculate_adjustment_factors(
        self,
        prices_df: pd.DataFrame,
        secid: str
    ) -> pd.DataFrame:
        """
        Calculate cumulative adjustment factors for a security.
        
        Returns DataFrame with adjustment_factor column to apply to historical prices.
        """
        if self.actions_df is None or len(self.actions_df) == 0:
            prices_df["adj_factor"] = 1.0
            return prices_df
        
        sec_actions = self.actions_df[
            (self.actions_df["secid"] == secid) &
            (self.actions_df["action_type"].isin(["dividend", "split"]))
        ].copy()
        
        if len(sec_actions) == 0:
            prices_df["adj_factor"] = 1.0
            return prices_df
        
        sec_actions = sec_actions.sort_values("ex_date", ascending=False)
        
        prices_df = prices_df.copy()
        prices_df["TRADEDATE"] = pd.to_datetime(prices_df["TRADEDATE"])
        prices_df["adj_factor"] = 1.0
        
        current_factor = 1.0
        for _, action in sec_actions.iterrows():
            ex_date = pd.to_datetime(action["ex_date"])
            
            if action["action_type"] == "dividend":
                div_value = action["value"] or 0
                # Find price before ex-date
                pre_ex_prices = prices_df[prices_df["TRADEDATE"] < ex_date]
                if len(pre_ex_prices) > 0:
                    last_price = pre_ex_prices.iloc[-1]["CLOSE"]
                    adj = (last_price - div_value) / last_price
                    current_factor *= adj
            
            elif action["action_type"] == "split":
                split_ratio = action["value"] or 1.0
                current_factor *= split_ratio
        
            prices_df.loc[prices_df["TRADEDATE"] < ex_date, "adj_factor"] = current_factor
        
        return prices_df
    
    def detect_dividend_gaps(
        self,
        prices_df: pd.DataFrame,
        secid: str,
        min_gap_percent: float = 0.03
    ) -> List[Dict[str, Any]]:
        """
        Detect potential dividend gaps in price data.
        
        This helps identify unrecorded corporate actions.
        """
        if len(prices_df) < 5:
            return []
        
        df = prices_df.sort_values("TRADEDATE").copy()
        df["daily_ret"] = df["CLOSE"].pct_change()
        
        # Look for large negative gaps that could be dividends
        large_gaps = df[df["daily_ret"] < -min_gap_percent].copy()
        
        detected_actions = []
        for idx, row in large_gaps.iterrows():
            if idx == 0:
                continue
            
            prev_idx = idx - 1
            prev_date = df.iloc[prev_idx]["TRADEDATE"]
            gap_date = row["TRADEDATE"]
            gap_size = abs(row["daily_ret"])
            
            detected_actions.append({
                "secid": secid,
                "ex_date": gap_date,
                "action_type": "suspected_dividend",
                "estimated_value": df.iloc[prev_idx]["CLOSE"] * gap_size,
                "gap_percent": gap_size * 100,
                "confidence": "low"
            })
        
        return detected_actions
    
    def get_ex_dividend_dates(
        self,
        secid: str,
        from_date: str,
        till_date: str
    ) -> pd.DataFrame:
        """Get all ex-dividend dates for a security in a date range."""
        if self.actions_df is None or len(self.actions_df) == 0:
            return pd.DataFrame()
        
        mask = (
            (self.actions_df["secid"] == secid) &
            (self.actions_df["action_type"] == "dividend") &
            (self.actions_df["ex_date"] >= from_date) &
            (self.actions_df["ex_date"] <= till_date)
        )
        
        return self.actions_df[mask].copy()
    
    def create_manual_actions_template(self) -> pd.DataFrame:
        """Create template CSV for manual corporate actions entry."""
        template = pd.DataFrame(columns=[
            "secid", "ex_date", "action_type", "value",
            "adjustment_factor", "source", "published_at"
        ])
        
        example_data = [
            ["SBER", "2024-06-20", "dividend", 33.0, None, "moex_disclosure", "2024-05-15"],
            ["GAZP", "2024-07-15", "dividend", 0.0, None, "moex_disclosure", "2024-06-28"],
        ]
        
        for row in example_data:
            template.loc[len(template)] = row
        
        return template


def build_dividend_database() -> CorporateActionsHandler:
    """
    Build initial dividend database for top MOEX stocks.
    
    This should be populated from MOEX disclosure or manual research.
    """
    handler = CorporateActionsHandler()
    
    # Placeholder - in production, this would scrape MOEX disclosure
    # or use a paid data source
    print("Corporate actions database initialized.")
    print("Populate data/corporate_actions/corporate_actions.csv manually or via scraper.")
    
    return handler


if __name__ == "__main__":
    handler = CorporateActionsHandler()
    handler.load_actions()
    
    if len(handler.actions_df) == 0:
        print("No corporate actions loaded. Creating template...")
        template = handler.create_manual_actions_template()
        template.to_csv("data/processed/corporate_actions_template.csv", index=False)
        print("Template saved to data/processed/corporate_actions_template.csv")
    else:
        print(f"Loaded {len(handler.actions_df)} corporate actions")
        print(handler.actions_df.head())
