#!/usr/bin/env python3
"""
Run the complete MarketPulse pipeline end-to-end.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --synthetic
  python scripts/run_pipeline.py --days 14
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
    parser = argparse.ArgumentParser(description='MarketPulse Full Pipeline')
    parser.add_argument('--synthetic', action='store_true', help='Force synthetic data')
    parser.add_argument('--days', type=int, default=7, help='Lookback days')
    args = parser.parse_args()

    config = load_config()
    if args.synthetic:
        config['data']['mode'] = 'synthetic'
    config.setdefault('ingestion', {}).setdefault('date_range', {})['default_lookback_days'] = args.days

    print("=" * 60)
    print("  MARKETPULSE -- Full Pipeline")
    print("  Data-Centric Sentiment Intelligence")
    print("=" * 60)

    # -- Step 1: Ingest ------------------------------------------------
    print("\n[1/7] INGESTING DATA...")
    mgr = IngestionManager(config)
    df = mgr.ingest()
    summary = mgr.get_source_summary()
    print(f"  -> {summary['total_posts']} posts from {summary['sources_used']}")

    # Save raw
    raw_dir = config.get('data', {}).get('storage', {}).get('raw_dir', 'data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    df.to_csv(os.path.join(raw_dir, 'ingested_data.csv'), index=False)

    # -- Step 2: Label -------------------------------------------------
    print("\n[2/7] RUNNING LABELING PIPELINE...")
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)
    labeled = df[df['programmatic_label'].notna()]
    print(f"  -> {len(labeled)}/{len(df)} posts labeled ({len(labeled)/len(df):.1%})")

    # Save labeled
    labeled_dir = config.get('data', {}).get('storage', {}).get('labeled_dir', 'data/labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    df.to_csv(os.path.join(labeled_dir, 'labeled_data.csv'), index=False)

    # -- Step 3: Label Quality -----------------------------------------
    print("\n[3/7] ASSESSING LABEL QUALITY...")
    analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS, agg)
    quality_report = analyzer.aggregate_quality_report(df)
    print(f"  -> Coverage: {quality_report['total_coverage']:.1%}")
    print(f"  -> Conflict rate: {quality_report['conflict_rate']:.1%}")
    print(f"  -> Avg votes/post: {quality_report['avg_votes_per_post']:.1f}")
    print(f"  -> Distribution: {quality_report['label_distribution']}")

    # Compare to gold
    gold_dir = config.get('data', {}).get('storage', {}).get('gold_dir', 'data/gold')
    gold_path = os.path.join(gold_dir, 'gold_standard.csv')
    gold_df = None
    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path)
        gold_report = analyzer.compare_to_gold(df, gold_df)
        print(f"  -> Gold agreement: {gold_report['agreement_rate']:.1%}")

    # -- Step 4: Train Model -------------------------------------------
    print("\n[4/7] TRAINING MODEL ON PROGRAMMATIC LABELS...")
    model_config = config.get('model', {})
    pipeline = SentimentPipeline(model_config)
    train_report = pipeline.train(labeled['text'].tolist(), labeled['programmatic_label'].tolist())
    if 'validation_metrics' in train_report:
        print(f"  -> Validation F1: {train_report['validation_metrics']['weighted_f1']:.3f}")
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

    # -- Step 5: Thesis Experiment -------------------------------------
    print("\n[5/7] RUNNING THESIS EXPERIMENT...")
    print("  Training same model on: gold, programmatic, noisy, random labels...")
    thesis = None
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

    # -- Step 6: Entity Extraction -------------------------------------
    print("\n[6/7] EXTRACTING ENTITIES...")
    te = TickerExtractor()
    normalizer = EntityNormalizer()
    df['extracted_entities'] = df['text'].apply(lambda t: te.extract(t))
    entity_counts = {}
    for entities in df['extracted_entities']:
        for e in entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1
    top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:10]
    print(f"  -> Entities extracted from {sum(1 for e in df['extracted_entities'] if e)}/{len(df)} posts")
    if top_entities:
        print(f"  -> Top mentioned: {', '.join(f'{e[0]}({e[1]})' for e in top_entities[:5])}")

    # Extraction eval on gold
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

    # -- Step 7: Summary -----------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Posts ingested:    {summary['total_posts']}")
    print(f"  Posts labeled:     {len(labeled)} ({len(labeled)/len(df):.1%})")
    print(f"  Label coverage:    {quality_report['total_coverage']:.1%}")
    if 'validation_metrics' in train_report:
        print(f"  Model F1:          {train_report['validation_metrics']['weighted_f1']:.3f}")
    if thesis is not None:
        print(f"  Thesis validated:  {thesis['thesis_validated']}")
    print(f"\n  Core thesis: 'Same model, different data quality.'")
    print(f"  A logistic regression on high-quality programmatic labels")
    print(f"  is a production-ready system.")
    print("=" * 60)


if __name__ == '__main__':
    main()
