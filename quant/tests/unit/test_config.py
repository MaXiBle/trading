"""Tests for configuration loading."""

import pytest
import tempfile
from pathlib import Path

from src.utils.config import (
    load_config,
    get_project_root,
    resolve_path,
    PlatformConfig,
)


class TestPlatformConfig:
    """Test cases for PlatformConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PlatformConfig()
        
        assert config.experiment == "phase0_platform"
        assert config.data_version == "v1"
        assert config.market.exchange == "MOEX"
        assert config.market.board == "TQBR"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "experiment": "test_experiment",
            "data_version": "v2",
            "market": {
                "exchange": "TEST",
                "board": "TEST",
            },
        }
        
        config = PlatformConfig(**data)
        
        assert config.experiment == "test_experiment"
        assert config.data_version == "v2"
        assert config.market.exchange == "TEST"


class TestLoadConfig:
    """Test cases for load_config function."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
experiment: test_exp
data_version: v1
market:
  exchange: TEST
  board: TQBR
""")
            f.flush()
            
            config = load_config(f.name)
            
            assert config.experiment == "test_exp"
            assert config.data_version == "v1"
            assert config.market.exchange == "TEST"
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
    
    def test_load_full_config(self):
        """Test loading a complete configuration."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("""
experiment: full_test
data_version: v3
market:
  exchange: MOEX
  board: TQBR
  currency: RUB
ingestion:
  moex_iss:
    base_url: https://test.moex.com
    start_date: "2015-01-01"
splitter:
  train_window_days: 500
  validation_window_days: 30
  embargo_days: 3
cost_model:
  total_cost_bps: 15
backtest:
  initial_capital: 5000000
  max_position_pct: 0.05
""")
            f.flush()
            
            config = load_config(f.name)
            
            assert config.experiment == "full_test"
            assert config.ingestion.moex_iss.start_date == "2015-01-01"
            assert config.splitter.train_window_days == 500
            assert config.cost_model.total_cost_bps == 15
            assert config.backtest.initial_capital == 5000000
        
        # Cleanup
        Path(f.name).unlink()


class TestPathResolution:
    """Test cases for path resolution functions."""
    
    def test_get_project_root(self):
        """Test getting project root directory."""
        root = get_project_root()
        assert root.exists()
        assert root.is_dir()
    
    def test_resolve_absolute_path(self):
        """Test resolving an absolute path."""
        abs_path = "/tmp/test"
        resolved = resolve_path(abs_path)
        assert resolved == Path(abs_path)
    
    def test_resolve_relative_path(self):
        """Test resolving a relative path."""
        rel_path = "data/raw"
        resolved = resolve_path(rel_path)
        assert resolved.is_absolute()
        assert resolved.exists() or not resolved.exists()  # Path may or may not exist
