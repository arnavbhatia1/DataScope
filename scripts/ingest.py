#!/usr/bin/env python3
"""
CLI: Data ingestion

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --days 14
  python scripts/ingest.py --source synthetic
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.utils.logger import get_logger

logger = get_logger('ingest')


def main():
    parser = argparse.ArgumentParser(description='MarketPulse Data Ingestion')
    parser.add_argument('--days', type=int, default=7, help='Lookback days')
    parser.add_argument('--source', type=str, default=None,
                        help='Force source: reddit, stocktwits, news, synthetic')
    args = parser.parse_args()

    config = load_config()

    # Override mode if source specified
    if args.source == 'synthetic':
        config['data']['mode'] = 'synthetic'
    elif args.source:
        config['data']['mode'] = 'live'

    config.setdefault('ingestion', {}).setdefault('date_range', {})['default_lookback_days'] = args.days

    mgr = IngestionManager(config)
    df = mgr.ingest()

    # Save raw data
    raw_dir = config.get('data', {}).get('storage', {}).get('raw_dir', 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, 'ingested_data.csv')
    df.to_csv(output_path, index=False)

    summary = mgr.get_source_summary()
    print(f"\n{'='*50}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Total posts: {summary['total_posts']}")
    print(f"  Sources: {summary['sources_used']}")
    print(f"  Unavailable: {summary.get('sources_unavailable', [])}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
