"""MOEX ISS API client for downloading daily candles."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import pandas as pd
from pydantic import BaseModel


class MOEXISSConfig(BaseModel):
    """Configuration for MOEX ISS client."""
    base_url: str = "https://iss.moex.com/iss"
    boards: List[str] = ["TQBR"]
    engines: List[str] = ["stock"]
    markets: List[str] = ["shares"]
    interval: int = 24  # daily
    rate_limit_delay: float = 0.5
    retry_attempts: int = 3
    cache_dir: str = "data/raw/moex_cache"


class MOEXISSClient:
    """Client for MOEX ISS API with pagination, retry, and caching."""
    
    def __init__(self, config: Optional[MOEXISSConfig] = None):
        self.config = config or MOEXISSConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def _build_url(
        self,
        secid: str,
        from_date: str,
        till_date: str,
        start: int = 0,
        limit: int = 100
    ) -> str:
        """Build ISS API URL for a security."""
        params = {
            "from": from_date,
            "till": till_date,
            "interval": self.config.interval,
            "start": start,
            "limit": limit,
        }
        
        url = (
            f"{self.config.base_url}/history/engines/"
            f"{self.config.engines[0]}/markets/"
            f"{self.config.markets[0]}/boards/"
            f"{self.config.boards[0]}/securities/{secid}.json"
        )
        
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{url}?{query_string}"
    
    def _fetch_with_retry(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch URL with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == self.config.retry_attempts - 1:
                    print(f"Failed to fetch {url} after {self.config.retry_attempts} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # exponential backoff
        return None
    
    def fetch_security_history(
        self,
        secid: str,
        from_date: str,
        till_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch complete history for a security with pagination."""
        all_rows = []
        start = 0
        limit = 100
        
        while True:
            url = self._build_url(secid, from_date, till_date, start, limit)
            data = self._fetch_with_retry(url)
            
            if data is None:
                break
            
            # Extract history data
            history_data = data.get("history", [])
            if not history_data:
                break
            
            all_rows.extend(history_data)
            
            # Check if we got fewer rows than limit (end of data)
            if len(history_data) < limit:
                break
            
            start += limit
            time.sleep(self.config.rate_limit_delay)
        
        if not all_rows:
            return None
        
        df = pd.DataFrame(all_rows)
        return df
    
    def fetch_multiple_securities(
        self,
        secids: List[str],
        from_date: str,
        till_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Fetch history for multiple securities."""
        results = {}
        
        for i, secid in enumerate(secids):
            cache_file = self.cache_dir / f"{secid}_{from_date}_{till_date}.parquet"
            
            if use_cache and cache_file.exists():
                try:
                    results[secid] = pd.read_parquet(cache_file)
                    print(f"[{i+1}/{len(secids)}] Loaded {secid} from cache")
                    continue
                except Exception as e:
                    print(f"Cache read failed for {secid}: {e}")
            
            print(f"[{i+1}/{len(secids)}] Fetching {secid}...")
            df = self.fetch_security_history(secid, from_date, till_date)
            
            if df is not None and len(df) > 0:
                results[secid] = df
                if use_cache:
                    df.to_parquet(cache_file, index=False)
                    print(f"  Cached {len(df)} rows")
            else:
                print(f"  No data for {secid}")
            
            time.sleep(self.config.rate_limit_delay)
        
        return results
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate downloaded dataframe."""
        issues = []
        
        required_columns = [
            "TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE",
            "VOLUME", "VALUE", "WAPRICE"
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for negative prices/volumes
        if (df["VOLUME"] < 0).any():
            issues.append("Negative volumes detected")
        
        price_cols = ["OPEN", "LOW", "HIGH", "CLOSE"]
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")
        
        # Check for duplicates
        if df.duplicated(subset=["TRADEDATE"]).any():
            issues.append("Duplicate dates detected")
        
        return len(issues) == 0, issues


def get_liquid_tickers() -> List[str]:
    """Return list of liquid TQBR tickers for initial testing."""
    # Top liquid stocks on MOEX TQBR
    return [
        "SBER", "GAZP", "LKOH", "ROSN", "GMKN",
        "YNDX", "VTBR", "POLY", "MGNT", "NVTK",
        "SNGSP", "HYDR", "TATN", "TRNFP", "AFKS",
        "PIKK", "MRKK", "AFLT", "ALRS", "CHMF",
        "FEES", "IRAO", "KMAZ", "LSRG", "MTLR",
        "MVID", "NLMK", "PHOR", "PLNZ", "RUAL",
        "SELG", "SGZH", "TCSG", "UPRO", "URKA"
    ]


if __name__ == "__main__":
    # Test download for a few tickers
    client = MOEXISSClient()
    tickers = get_liquid_tickers()[:5]  # test with first 5
    
    print(f"Fetching data for {tickers}...")
    results = client.fetch_multiple_securities(
        tickers,
        "2020-01-01",
        "2024-12-31"
    )
    
    for secid, df in results.items():
        is_valid, issues = client.validate_dataframe(df)
        print(f"\n{secid}: {len(df)} rows, valid={is_valid}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
