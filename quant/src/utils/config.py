"""Configuration loader and validator for the platform."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class MarketConfig(BaseModel):
    """Market configuration."""
    exchange: str = "MOEX"
    board: str = "TQBR"
    currency: str = "RUB"
    timezone: str = "Europe/Moscow"


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""
    class MOEXISS(BaseModel):
        base_url: str = "https://iss.moex.com/iss"
        history_interval: int = 24
        start_date: str = "2010-01-01"
        retry_attempts: int = 3
        retry_delay_sec: int = 5
        request_timeout_sec: int = 30
    
    moex_iss: MOEXISS = Field(default_factory=MOEXISS)


class CalendarConfig(BaseModel):
    """Trading calendar configuration."""
    min_date: str = "2010-01-01"
    max_date: str = "2030-12-31"
    exchange_holidays_path: Optional[str] = None


class UniverseConfig(BaseModel):
    """Universe builder configuration."""
    min_history_days: int = 252
    min_adv_63: float = 5_000_000
    max_universe_size: int = 50
    liquidity_lookback_days: int = 63


class CorporateActionsConfig(BaseModel):
    """Corporate actions configuration."""
    adjust_prices: bool = True
    adjust_volume: bool = True
    min_adjust_factor: float = 0.001
    max_adjust_factor: float = 1000.0


class DataQualityConfig(BaseModel):
    """Data quality checks configuration."""
    price_floor: float = 0.01
    price_ceiling: float = 1_000_000
    volume_floor: float = 0
    max_daily_return: float = 0.99
    min_traded_value: float = 100_000


class SplitterConfig(BaseModel):
    """Walk-forward splitter configuration."""
    train_window_days: int = 1260
    validation_window_days: int = 63
    test_window_days: int = 21
    step_days: int = 21
    embargo_days: int = 5


class CostModelConfig(BaseModel):
    """Cost model configuration."""
    commission_bps: float = 5
    spread_bps: float = 5
    slippage_bps: float = 2
    total_cost_bps: float = 12


class BacktestConfig(BaseModel):
    """Backtester configuration."""
    initial_capital: float = 1_000_000
    long_only: bool = True
    gross_exposure: float = 1.0
    max_position_pct: float = 0.10
    rebalance_frequency_days: int = 5


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    horizons: list[int] = [1, 5, 20]
    risk_free_rate: float = 0.10
    trading_days_per_year: int = 252


class PathsConfig(BaseModel):
    """Data paths configuration."""
    raw_data: str = "data/raw"
    curated_data: str = "data/curated"
    features: str = "data/features"
    labels: str = "data/labels"
    predictions: str = "data/predictions"
    backtests: str = "data/backtests"
    reports: str = "reports"


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""
    random_seed: int = 42
    save_git_commit: bool = True
    save_config_hash: bool = True


class PlatformConfig(BaseModel):
    """Main platform configuration."""
    experiment: str = "phase0_platform"
    data_version: str = "v1"
    market: MarketConfig = Field(default_factory=MarketConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    corporate_actions: CorporateActionsConfig = Field(default_factory=CorporateActionsConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


def load_config(config_path: str | Path) -> PlatformConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Validated PlatformConfig object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return PlatformConfig(**config_dict)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def resolve_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a path relative to project root or base directory.
    
    Args:
        path: Relative or absolute path string.
        base_dir: Base directory for relative paths. Defaults to project root.
        
    Returns:
        Absolute Path object.
    """
    if base_dir is None:
        base_dir = get_project_root()
    
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    
    return resolved.resolve()
