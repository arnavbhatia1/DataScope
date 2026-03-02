#!/usr/bin/env python3
"""
CLI: Run full evaluation

Usage:
  python scripts/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import pandas as pd
from src.utils.config import load_config
from src.models.pipeline import SentimentPipeline
from src.evaluation.classification import evaluate_classification
from src.evaluation.extraction import evaluate_extraction
from src.extraction.ticker_extractor import TickerExtractor
from src.extraction.normalizer import EntityNormalizer
from src.utils.logger import get_logger

logger = get_logger('evaluate')


def _parse_tickers_gold(value):
    """Safely parse gold ticker annotations from CSV string representation."""
    if pd.isna(value):
        return []
    try:
        # ast.literal_eval safely parses Python literal structures (lists, strings, etc.)
        # It does NOT execute arbitrary code -- it only handles literals.
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        return []


def main():
    config = load_config()

    # Load model
    model_dir = config.get('data', {}).get('storage', {}).get('model_dir', 'data/models')
    pipeline = SentimentPipeline()
    pipeline.load(model_dir)
    print("Model loaded")

    # Load gold standard
    gold_dir = config.get('data', {}).get('storage', {}).get('gold_dir', 'data/gold')
    gold_path = os.path.join(gold_dir, 'gold_standard.csv')
    if not os.path.exists(gold_path):
        print(f"No gold standard at {gold_path}.")
        sys.exit(1)
    gold = pd.read_csv(gold_path)
    print(f"Gold standard: {len(gold)} posts")

    # Classification evaluation
    predictions = pipeline.predict(gold['text'].tolist())
    pred_labels = [p['label'] for p in predictions]
    pred_confs = [p['confidence'] for p in predictions]

    cls_result = evaluate_classification(
        gold['sentiment_gold'].tolist(),
        pred_labels,
        texts=gold['text'].tolist(),
        confidence_scores=pred_confs,
    )

    print(f"\n{'='*50}")
    print(f"CLASSIFICATION EVALUATION")
    print(f"{'='*50}")
    print(f"  Accuracy: {cls_result['accuracy']:.3f}")
    print(f"  Weighted F1: {cls_result['weighted_f1']:.3f}")
    print(f"  Errors: {len(cls_result['errors'])}/{len(gold)}")
    if cls_result['summary']['most_confused_pair']:
        pair = cls_result['summary']['most_confused_pair']
        print(f"  Most confused: {pair[0]} ({pair[1]} times)")

    print(f"\n  Per-class:")
    for cls, metrics in cls_result['per_class'].items():
        print(f"    {cls:>10}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    # Entity extraction evaluation
    te = TickerExtractor()
    normalizer = EntityNormalizer()

    pred_entities = []
    gold_entities = []

    for _, row in gold.iterrows():
        pred_entities.append(set(te.extract(row['text'])))
        tickers = _parse_tickers_gold(row.get('tickers_gold'))
        gold_entities.append(set(tickers))

    # Only evaluate posts that have gold entities
    has_gold = [(p, g) for p, g in zip(pred_entities, gold_entities) if g]
    if has_gold:
        filtered_preds, filtered_golds = zip(*has_gold)
        ext_result = evaluate_extraction(list(filtered_preds), list(filtered_golds), normalizer)

        print(f"\n{'='*50}")
        print(f"ENTITY EXTRACTION EVALUATION")
        print(f"{'='*50}")
        print(f"  With normalization:")
        print(f"    Precision: {ext_result['metrics']['precision']:.3f}")
        print(f"    Recall: {ext_result['metrics']['recall']:.3f}")
        print(f"    F1: {ext_result['metrics']['f1']:.3f}")
        print(f"  Without normalization:")
        print(f"    F1: {ext_result['metrics_without_normalization']['f1']:.3f}")
        print(f"  Normalization lift: +{ext_result['normalization_lift']['f1_lift']:.3f}")
    else:
        print("\n  No gold entity annotations found for extraction eval.")

    print(f"{'='*50}")


if __name__ == '__main__':
    main()
