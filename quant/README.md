# MOEX Quant Platform

Research and backtest platform for Russian equities (MOEX).

## Architecture

```
quant/
├── configs/              # YAML configuration files
├── data_ingestion/       # MOEX ISS API client
├── data_quality/         # Data validation checks
├── corporate_actions/    # Dividend/split handling
├── features/             # Feature engineering
├── labels/               # Label construction
├── splitting/            # Walk-forward splitter with embargo
├── models/               # ML models (LightGBM, etc.)
├── backtest/             # Event-driven backtester
├── metrics/              # Performance metrics & Rank IC
├── research/             # Jupyter notebooks
├── pipelines/            # End-to-end pipelines
├── reports/              # Generated reports
├── tests/                # Unit tests
└── data/
    ├── raw/              # Raw downloaded data
    └── processed/        # Cleaned Parquet files
```

## Installation

```bash
cd quant
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

```python
from data_ingestion.moex_iss_client import MOEXISSClient, get_liquid_tickers

client = MOEXISSClient()
tickers = get_liquid_tickers()[:10]  # Top 10 liquid stocks

results = client.fetch_multiple_securities(
    tickers,
    "2020-01-01",
    "2024-12-31"
)
```

### 2. Validate Data Quality

```python
from data_quality.checks import DataQualityChecker

checker = DataQualityChecker()
for secid, df in results.items():
    result = checker.run_all_checks(df, secid)
    print(f"{secid}: valid={result['valid']}, issues={result['issue_count']}")
```

### 3. Run Tests

```bash
pytest tests/test_platform.py -v
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
backtest:
  horizon_days: 5          # Trading horizon
  cost_bps_one_way: 12     # Transaction costs
  
model:
  baseline:
    n_estimators: 300
    learning_rate: 0.03

splitting:
  embargo_days: 5          # Leakage protection
```

## Key Features

### Point-in-Time Data
All features use only historically available data. No look-ahead bias.

### Walk-Forward Validation
- Train on past data only
- Embargo period between train and test
- Rolling validation windows

### Cost Model
- Configurable transaction costs (bps)
- Turnover calculation
- Market impact (optional)

### Metrics
- Sharpe, Sortino, Calmar ratios
- Max drawdown
- Rank IC (Spearman correlation)
- Quintile analysis
- Cost sensitivity

## Development Workflow

1. **Platform First**: Build common infrastructure before models
2. **Baseline**: Establish simple cross-sectional baseline
3. **Incremental Improvements**: Each module must prove value over baseline
4. **Reproducibility**: Fix random seeds, log git commits, version data

## Testing Checklist

- [ ] No look-ahead bias in features
- [ ] Embargo respected in splits
- [ ] Costs included in backtest
- [ ] Reproducible results (same seed = same output)
- [ ] Unit tests pass

## Next Steps (Phase 1+)

After platform is ready:

1. **BASE-001**: Build feature engineering pipeline
2. **BASE-002**: Implement label construction
3. **BASE-003**: Walk-forward model training
4. **BASE-004**: Portfolio construction
5. **BASE-005**: Generate baseline report

## License

Internal research use only.
