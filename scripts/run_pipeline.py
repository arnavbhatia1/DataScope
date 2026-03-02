#!/usr/bin/env python3
"""
Run the complete MarketPulse pipeline end-to-end.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --synthetic
  python scripts/run_pipeline.py --days 14
  python scripts/run_pipeline.py --thesis   # also run thesis experiment
"""

import argparse
import ast
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.labeling.quality import LabelQualityAnalyzer
from src.labeling.functions import LABELING_FUNCTIONS
from src.models.pipeline import SentimentPipeline
from src.models.versioning import ModelVersion
from src.evaluation.classification import evaluate_classification
from src.evaluation.extraction import evaluate_extraction
from src.evaluation.label_quality import run_thesis_experiment
from src.extraction.ticker_extractor import TickerExtractor
from src.extraction.normalizer import EntityNormalizer
from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
from src.utils.logger import get_logger

logger = get_logger('pipeline')


def _parse_tickers_gold(value):
    """Safely parse gold ticker annotations from CSV string representation."""
    if pd.isna(value):
        return []
    try:
        # ast.literal_eval safely parses Python literal structures only.
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        return []


def main():
    parser = argparse.ArgumentParser(description='MarketPulse Sentiment Intelligence Pipeline')
    parser.add_argument('--synthetic', action='store_true', help='Force synthetic data')
    parser.add_argument('--days', type=int, default=7, help='Lookback days')
    parser.add_argument('--thesis', action='store_true',
                        help='Run thesis experiment (gold vs programmatic vs noisy vs random)')
    args = parser.parse_args()

    config = load_config()
    if args.synthetic:
        config['data']['mode'] = 'synthetic'
    config.setdefault('ingestion', {}).setdefault('date_range', {})['default_lookback_days'] = args.days

    num_steps = 8 if args.thesis else 7

    print("=" * 60)
    print("  MARKETPULSE -- Sentiment Intelligence Pipeline")
    print("=" * 60)

    # -- Step 1: Ingest ------------------------------------------------
    print(f"\n[1/{num_steps}] INGESTING DATA...")
    mgr = IngestionManager(config)
    df = mgr.ingest()
    summary = mgr.get_source_summary()
    print(f"  -> {summary['total_posts']} posts from {summary['sources_used']}")

    # Save raw
    raw_dir = config.get('data', {}).get('storage', {}).get('raw_dir', 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    df.to_csv(os.path.join(raw_dir, 'ingested_data.csv'), index=False)

    # -- Step 2: Label -------------------------------------------------
    print(f"\n[2/{num_steps}] RUNNING LABELING PIPELINE...")
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)
    labeled = df[df['programmatic_label'].notna()]
    print(f"  -> {len(labeled)}/{len(df)} posts labeled ({len(labeled)/len(df):.1%})")

    # -- Step 3: Extract Entities --------------------------------------
    print(f"\n[3/{num_steps}] EXTRACTING ENTITIES...")
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(lambda t: te.extract(t))
    posts_with_tickers = sum(1 for t in df['tickers'] if t)
    print(f"  -> Tickers found in {posts_with_tickers}/{len(df)} posts")

    # Save labeled data with tickers
    labeled_dir = config.get('data', {}).get('storage', {}).get('labeled_dir', 'data/labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    df.to_csv(os.path.join(labeled_dir, 'labeled_data.csv'), index=False)

    # -- Step 4: Label Quality -----------------------------------------
    print(f"\n[4/{num_steps}] ASSESSING LABEL QUALITY...")
    analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS, agg)
    quality_report = analyzer.aggregate_quality_report(df)
    print(f"  -> Coverage: {quality_report['total_coverage']:.1%}, "
          f"Conflicts: {quality_report['conflict_rate']:.1%}")

    # Compare to gold if available
    gold_dir = config.get('data', {}).get('storage', {}).get('gold_dir', 'data/gold')
    gold_path = os.path.join(gold_dir, 'gold_standard.csv')
    gold_df = None
    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path)
        gold_report = analyzer.compare_to_gold(df, gold_df)
        print(f"  -> Gold agreement: {gold_report['agreement_rate']:.1%}")

    # -- Step 5: Train Model -------------------------------------------
    print(f"\n[5/{num_steps}] TRAINING MODEL...")
    model_config = config.get('model', {})
    pipeline = SentimentPipeline(model_config)
    train_report = pipeline.train(labeled['text'].tolist(), labeled['programmatic_label'].tolist())
    val_f1 = None
    if 'validation_metrics' in train_report:
        val_f1 = train_report['validation_metrics']['weighted_f1']
        print(f"  -> Validation F1: {val_f1:.3f}")
    print(f"  -> Features: {train_report['num_features']}")

    # Save model
    model_dir = config.get('data', {}).get('storage', {}).get('model_dir', 'data/models')
    pipeline.save(model_dir)
    mv = ModelVersion(model_dir)
    mv.save_version(pipeline, 'programmatic', train_report.get('validation_metrics', {}))

    # Feature importance
    features = pipeline.get_feature_importance(top_n=5)
    print(f"  -> Top features:")
    for cls, feats in features.items():
        top3 = ', '.join(f[0] for f in feats[:3])
        print(f"     {cls}: {top3}")

    # -- Step 6: Ticker Sentiment Analysis -----------------------------
    print(f"\n[6/{num_steps}] ANALYZING TICKER SENTIMENT...")
    tsa = TickerSentimentAnalyzer()
    ticker_results = tsa.analyze(df)
    market_summary = tsa.get_market_summary(ticker_results)

    total_tickers = market_summary['total_tickers']
    total_mentions = market_summary['total_mentions']
    dist = market_summary['ticker_sentiment_distribution']
    print(f"  -> {total_tickers} tickers tracked, {total_mentions} total mentions")

    if dist:
        dist_str = ', '.join(f"{k}: {v}" for k, v in sorted(dist.items()))
        print(f"  -> Market sentiment: {{{dist_str}}}")

    # Print top tickers
    top_bullish = market_summary.get('top_bullish', [])
    top_bearish = market_summary.get('top_bearish', [])
    top_to_show = sorted(ticker_results.values(), key=lambda t: t['mention_count'], reverse=True)[:5]
    for t in top_to_show:
        sym = t['symbol']
        company = t['company']
        sentiment = t['dominant_sentiment']
        mentions = t['mention_count']
        conf = t['avg_confidence']
        label = f"{sym} ({company})" if sym else company
        print(f"     {label}: {sentiment} ({mentions} mentions, {conf:.0%} conf)")

    # -- Step 7: Save Artifacts ----------------------------------------
    print(f"\n[7/{num_steps}] SAVING ARTIFACTS...")

    # Extraction eval on gold
    normalizer = EntityNormalizer()
    if gold_df is not None and 'tickers_gold' in gold_df.columns:
        pred_entities = []
        gold_entities = []
        for _, row in gold_df.iterrows():
            pred_entities.append(set(te.extract(row['text'])))
            tickers = _parse_tickers_gold(row.get('tickers_gold'))
            gold_entities.append(set(tickers))

        has_gold = [(p, g) for p, g in zip(pred_entities, gold_entities) if g]
        if has_gold:
            fp, fg = zip(*has_gold)
            ext_result = evaluate_extraction(list(fp), list(fg), normalizer)
            print(f"  -> Extraction F1: {ext_result['metrics']['f1']:.3f} "
                  f"(normalization lift: +{ext_result['normalization_lift']['f1_lift']:.3f})")

    print(f"  -> Model saved to {model_dir}/")
    print(f"  -> Labeled data saved to {labeled_dir}/labeled_data.csv")

    # -- Optional: Thesis Experiment -----------------------------------
    thesis = None
    if args.thesis:
        print(f"\n[8/{num_steps}] RUNNING THESIS EXPERIMENT...")
        print("  Training same model on: gold, programmatic, noisy, random labels...")
        if gold_df is not None:
            thesis = run_thesis_experiment(df, gold_df, config)
            print(f"\n  RESULTS:")
            for _, row in thesis['results_table'].iterrows():
                marker = " <-- OUR PIPELINE" if row['label_source'] == 'programmatic' else ""
                print(f"    {row['label_source']:>15}: F1 = {row['weighted_f1']:.3f}{marker}")
            print(f"\n  Thesis validated: {thesis['thesis_validated']}")
            print(f"  Programmatic vs Gold gap: {thesis['programmatic_vs_gold_gap']:.3f}")
        else:
            print("  -> Skipped (no gold standard)")

    # -- Summary -------------------------------------------------------
    # Build top bullish/bearish ticker symbol lists
    bullish_symbols = [t['symbol'] or t['company'] for t in top_bullish[:3]]
    bearish_symbols = [t['symbol'] or t['company'] for t in top_bearish[:3]]

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Posts analyzed:   {summary['total_posts']}")
    print(f"  Tickers tracked:  {total_tickers}")
    if bullish_symbols:
        print(f"  Top bullish:      {', '.join(bullish_symbols)}")
    if bearish_symbols:
        print(f"  Top bearish:      {', '.join(bearish_symbols)}")
    if val_f1 is not None:
        print(f"  Model F1:         {val_f1:.3f}")
    if thesis is not None:
        print(f"  Thesis validated: {thesis['thesis_validated']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
