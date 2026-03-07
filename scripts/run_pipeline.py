#!/usr/bin/env python3
"""
Run the complete MarketPulse pipeline end-to-end.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --days 14
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.models.pipeline import SentimentPipeline
from src.extraction.ticker_extractor import TickerExtractor
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
from src.storage.db import init_db, save_posts, save_ticker_cache, log_training_run
from src.utils.logger import get_logger
import uuid

logger = get_logger('pipeline')


def main():
    parser = argparse.ArgumentParser(description='MarketPulse Sentiment Intelligence Pipeline')
    parser.add_argument('--days', type=int, default=7, help='Lookback days')
    args = parser.parse_args()

    config = load_config()
    config.setdefault('ingestion', {}).setdefault('date_range', {})['default_lookback_days'] = args.days

    print("=" * 60)
    print("  MARKETPULSE -- Sentiment Intelligence Pipeline")
    print("=" * 60)

    init_db()

    # -- Step 1: Ingest ------------------------------------------------
    print("\n[1/5] INGESTING DATA...")
    mgr = IngestionManager(config)
    df = mgr.ingest()
    summary = mgr.get_source_summary()
    print(f"  -> {summary['total_posts']} posts from {summary['sources_used']}")

    # Save raw
    raw_dir = config.get('data', {}).get('storage', {}).get('raw_dir', 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    df.to_csv(os.path.join(raw_dir, 'ingested_data.csv'), index=False)

    # -- Step 2: Label -------------------------------------------------
    print("\n[2/5] RUNNING LABELING PIPELINE...")
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)
    labeled = df[df['programmatic_label'].notna()]
    print(f"  -> {len(labeled)}/{len(df)} posts labeled ({len(labeled)/len(df):.1%})")

    # Save labeled data
    labeled_dir = config.get('data', {}).get('storage', {}).get('labeled_dir', 'data/labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    df.to_csv(os.path.join(labeled_dir, 'labeled_data.csv'), index=False)

    # -- Step 3: Extract Entities & Map Labels -------------------------
    print("\n[3/5] EXTRACTING TICKER ENTITIES...")
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)
    df['sentiment'] = df['programmatic_label']
    df['confidence'] = df['label_confidence'].fillna(0.0)
    posts_with_tickers = sum(1 for t in df['tickers'] if t)
    print(f"  -> Tickers found in {posts_with_tickers}/{len(df)} posts")

    # -- Step 4: Analyze & Store ---------------------------------------
    print("\n[4/5] ANALYZING TICKER SENTIMENT...")
    tsa = TickerSentimentAnalyzer()
    ticker_results = tsa.analyze(df)
    market_summary = tsa.get_market_summary(ticker_results)
    total_tickers = market_summary['total_tickers']
    total_mentions = market_summary['total_mentions']
    print(f"  -> {total_tickers} tickers tracked, {total_mentions} total mentions")

    top_to_show = sorted(ticker_results.values(), key=lambda t: t['mention_count'], reverse=True)[:5]
    for t in top_to_show:
        sym = t['symbol']
        company = t['company']
        label = f"{sym} ({company})" if sym else company
        print(f"     {label}: {t['dominant_sentiment']} ({t['mention_count']} mentions, {t['avg_confidence']:.0%} conf)")

    save_posts(df)
    save_ticker_cache(ticker_results)

    # -- Step 5: Train Model -------------------------------------------
    print("\n[5/5] TRAINING MODEL...")
    model_dir = config.get('data', {}).get('storage', {}).get('model_dir', 'data/models')
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')

    if len(labeled) >= 200:
        pipeline = SentimentPipeline(config=config)
        report = pipeline.train(labeled['text'].tolist(), labeled['programmatic_label'].tolist())
        pipeline.save(model_dir)

        f1 = 0.0
        val = report.get('validation_metrics') or report.get('training_metrics')
        if val:
            f1 = val.get('weighted_f1', 0.0)
        log_training_run(str(uuid.uuid4()), len(labeled), f1)
        print(f"  -> Model trained. Validation F1: {f1:.3f}")
        print(f"  -> Saved to {model_dir}/")
    else:
        print(f"  -> Skipped (only {len(labeled)} labeled posts, need 200+)")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Posts analyzed:   {summary['total_posts']}")
    print(f"  Tickers tracked:  {total_tickers}")
    print("=" * 60)


if __name__ == '__main__':
    main()
