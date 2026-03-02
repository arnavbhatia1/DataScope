#!/usr/bin/env python3
"""
CLI: Run labeling pipeline on ingested data

Usage:
  python scripts/label.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.config import load_config
from src.labeling.aggregator import LabelAggregator
from src.labeling.quality import LabelQualityAnalyzer
from src.labeling.functions import LABELING_FUNCTIONS
from src.utils.logger import get_logger

logger = get_logger('label')


def main():
    config = load_config()

    # Load ingested data
    raw_dir = config.get('data', {}).get('storage', {}).get('raw_dir', 'data/raw')
    input_path = os.path.join(raw_dir, 'ingested_data.csv')

    if not os.path.exists(input_path):
        print(f"No ingested data found at {input_path}. Run 'python scripts/ingest.py' first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} posts from {input_path}")

    # Run aggregator
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    # Quality report
    analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS, agg)
    report = analyzer.aggregate_quality_report(df)

    # Save labeled data
    labeled_dir = config.get('data', {}).get('storage', {}).get('labeled_dir', 'data/labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    output_path = os.path.join(labeled_dir, 'labeled_data.csv')
    df.to_csv(output_path, index=False)

    labeled_count = df['programmatic_label'].notna().sum()
    print(f"\n{'='*50}")
    print(f"LABELING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total posts: {len(df)}")
    print(f"  Labeled: {labeled_count} ({labeled_count/len(df):.1%})")
    print(f"  Coverage: {report['total_coverage']:.1%}")
    print(f"  Conflict rate: {report['conflict_rate']:.1%}")
    print(f"  Avg votes/post: {report['avg_votes_per_post']:.1f}")
    print(f"  Distribution: {report['label_distribution']}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
