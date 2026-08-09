"""MOEX ISS data ingestion module.

Downloads historical market data from Moscow Exchange Information Statistical Server.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ..utils.config import IngestionConfig, resolve_path

logger = logging.getLogger(__name__)


class MOEXISSClient:
    """Client for MOEX Information Statistical Server."""
    
    def __init__(self, config: IngestionConfig):
        """Initialize MOEX ISS client.
        
        Args:
            config: Ingestion configuration.
        """
        self.base_url = config.moex_iss.base_url
        self.history_interval = config.moex_iss.history_interval
        self.start_date = config.moex_iss.start_date
        self.retry_attempts = config.moex_iss.retry_attempts
        self.retry_delay_sec = config.moex_iss.retry_delay_sec
        self.request_timeout_sec = config.moex_iss.request_timeout_sec
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MOEX-Research-Platform/0.1.0",
        })
    
    def _make_request(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Make HTTP request with retry logic.
        
        Args:
            url: Request URL.
            params: Query parameters.
            
        Returns:
            JSON response or None if all retries failed.
        """
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.request_timeout_sec,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay_sec * (attempt + 1))
        
        logger.error(f"All retry attempts failed for URL: {url}")
        return None
    
    def get_securities_list(self, board: str = "TQBR") -> pd.DataFrame:
        """Get list of securities for a specific board.
        
        Args:
            board: Trading board (default: TQBR).
            
        Returns:
            DataFrame with securities information.
        """
        url = f"{self.base_url}/engines/stock/markets/shares/boards/{board}/securities.json"
        
        data = self._make_request(url)
        if not data:
            return pd.DataFrame()
        
        # Parse MOEX ISS response format
        columns = data["securities"]["columns"]
        rows = data["securities"]["data"]
        
        df = pd.DataFrame(rows, columns=[col["name"] for col in columns])
        df["source"] = "moex_iss"
        df["ingested_at"] = datetime.now()
        
        return df
    
    def get_history(
        self,
        security: str,
        board: str = "TQBR",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get historical data for a security.
        
        Args:
            security: Security ID (ticker).
            board: Trading board.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).
            
        Returns:
            DataFrame with OHLCV data.
        """
        params = {
            "interval": self.history_interval,
            "from": from_date or self.start_date,
        }
        
        if to_date:
            params["to"] = to_date
        
        url = f"{self.base_url}/engines/stock/markets/shares/boards/{board}/securities/{security}/candles.json"
        
        data = self._make_request(url, params)
        if not data:
            return pd.DataFrame()
        
        # Check if candles data exists
        if "candles" not in data or "data" not in data["candles"]:
            logger.warning(f"No candle data for {security}")
            return pd.DataFrame()
        
        columns = data["candles"]["columns"]
        rows = data["candles"]["data"]
        
        df = pd.DataFrame(rows, columns=[col["name"] for col in columns])
        
        if df.empty:
            return df
        
        # Convert begin column to datetime
        df["begin"] = pd.to_datetime(df["begin"])
        df["secid"] = security
        df["board"] = board
        df["source"] = "moex_iss"
        df["ingested_at"] = datetime.now()
        
        # Rename begin to trade_date for consistency
        df = df.rename(columns={"begin": "trade_date"})
        
        return df
    
    def get_all_histories(
        self,
        securities: list[str],
        board: str = "TQBR",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """Download historical data for multiple securities.
        
        Args:
            securities: List of security IDs.
            board: Trading board.
            from_date: Start date.
            to_date: End date.
            save_dir: Directory to save parquet files.
            
        Returns:
            Dictionary mapping security ID to DataFrame.
        """
        results = {}
        
        for i, secid in enumerate(securities):
            logger.info(f"[{i+1}/{len(securities)}] Downloading {secid}...")
            
            df = self.get_history(secid, board, from_date, to_date)
            
            if not df.empty:
                results[secid] = df
                
                if save_dir is not None:
                    save_path = save_dir / f"{secid}.parquet"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(save_path, index=False)
                    logger.debug(f"Saved {save_path}")
            
            # Small delay to be nice to the server
            if i < len(securities) - 1:
                time.sleep(0.5)
        
        return results


def ingest_moex_data(
    config: IngestionConfig,
    output_dir: Optional[Path] = None,
    securities: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Ingest MOEX historical data.
    
    Args:
        config: Ingestion configuration.
        output_dir: Output directory for raw data.
        securities: Optional list of securities to download. If None, downloads all.
        
    Returns:
        Dictionary of downloaded DataFrames.
    """
    client = MOEXISSClient(config)
    
    # Get securities list
    logger.info("Fetching securities list...")
    securities_df = client.get_securities_list()
    
    if securities_df.empty:
        logger.error("Failed to fetch securities list")
        return {}
    
    # Filter to active securities
    active_securities = securities_df[
        securities_df.get("BOARDID", securities_df.columns[0]) == "TQBR"
    ]
    
    if securities is None:
        # Get all active security IDs
        securities = active_securities["SECID"].tolist()
    
    logger.info(f"Downloading data for {len(securities)} securities...")
    
    if output_dir is None:
        output_dir = resolve_path("data/raw/moex/history/TQBR")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save securities master
    securities_path = output_dir.parent / "securities_master.parquet"
    securities_df.to_parquet(securities_path, index=False)
    logger.info(f"Saved securities master to {securities_path}")
    
    # Download histories
    results = client.get_all_histories(
        securities=securities,
        save_dir=output_dir,
    )
    
    logger.info(f"Successfully downloaded {len(results)} securities")
    
    return results
