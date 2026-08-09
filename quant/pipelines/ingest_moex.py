#!/usr/bin/env python3
"""MOEX data ingestion pipeline.

Usage:
    python pipelines/ingest_moex.py --config configs/phase0_platform.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data_ingestion.moex_ingestion import ingest_moex_data


def main():
    parser = argparse.ArgumentParser(description="Ingest MOEX historical data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase0_platform.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--securities",
        type=str,
        nargs="+",
        default=None,
        help="List of securities to download (default: all)",
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
    
    # Load configuration
    config_path = project_root / args.config
    logger = logging.getLogger(__name__)
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / config.paths.raw_data / "moex" / "history" / config.market.board
    
    # Run ingestion
    logger.info(f"Starting MOEX data ingestion...")
    logger.info(f"Output directory: {output_dir}")
    
    results = ingest_moex_data(
        config=config.ingestion,
        output_dir=output_dir,
        securities=args.securities,
    )
    
    if results:
        logger.info(f"Successfully ingested {len(results)} securities")
        
        # Summary
        total_rows = sum(len(df) for df in results.values())
        logger.info(f"Total rows: {total_rows}")
        
        date_ranges = {}
        for secid, df in results.items():
            if not df.empty and "trade_date" in df.columns:
                min_date = df["trade_date"].min()
                max_date = df["trade_date"].max()
                date_ranges[secid] = (min_date, max_date)
        
        logger.info("Date ranges:")
        for secid, (min_date, max_date) in list(date_ranges.items())[:5]:
            logger.info(f"  {secid}: {min_date.date()} to {max_date.date()}")
        if len(date_ranges) > 5:
            logger.info(f"  ... and {len(date_ranges) - 5} more")
    else:
        logger.warning("No data was ingested")
        sys.exit(1)


if __name__ == "__main__":
    main()
