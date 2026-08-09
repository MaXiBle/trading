#!/usr/bin/env python3
"""Backtest pipeline.

Usage:
    python pipelines/run_backtest.py --config configs/phase0_platform.yaml --mode dummy
    python pipelines/run_backtest.py --config configs/phase0_platform.yaml --mode full
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.engine import run_dummy_backtest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase0_platform.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dummy", "full"],
        default="dummy",
        help="Backtest mode: dummy (synthetic data) or full (real data)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting backtest in {args.mode} mode...")

    if args.mode == "dummy":
        # Run dummy backtest with synthetic data
        logger.info("Running dummy backtest...")

        result = run_dummy_backtest()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "data" / "backtests" / "phase0_dummy" / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([result.metrics])
        metrics_df.to_parquet(output_dir / "metrics.parquet", index=False)
        logger.info(f"Saved metrics to {output_dir / 'metrics.parquet'}")

        # Save returns
        result.portfolio_returns.to_frame().to_parquet(
            output_dir / "portfolio_returns.parquet"
        )
        logger.info(f"Saved returns to {output_dir / 'portfolio_returns.parquet'}")

        # Save values
        result.portfolio_values.to_frame().to_parquet(
            output_dir / "portfolio_values.parquet"
        )
        logger.info(f"Saved values to {output_dir / 'portfolio_values.parquet'}")

        # Print summary
        print("\n" + "=" * 60)
        print(result.summary())
        print("=" * 60 + "\n")

        logger.info("Dummy backtest completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

    elif args.mode == "full":
        # Full backtest requires real data (not implemented in Phase 0)
        logger.warning("Full backtest mode requires Phase 1+ data")
        logger.info("Please run 'make build_curated' and implement signal generation first")
        sys.exit(1)


if __name__ == "__main__":
    main()
