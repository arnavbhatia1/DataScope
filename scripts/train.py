#!/usr/bin/env python3
"""
CLI: Train sentiment model

Usage:
  python scripts/train.py
  python scripts/train.py --source programmatic
  python scripts/train.py --source gold
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils.config import load_config
from src.models.pipeline import SentimentPipeline
from src.utils.logger import get_logger

logger = get_logger('train')


def main():
    parser = argparse.ArgumentParser(description='MarketPulse Model Training')
    parser.add_argument('--source', type=str, default='programmatic',
                        choices=['programmatic', 'gold'],
                        help='Label source for training')
    args = parser.parse_args()

    config = load_config()
    model_config = config.get('model', {})

    if args.source == 'programmatic':
        labeled_dir = config.get('data', {}).get('storage', {}).get('labeled_dir', 'data/labeled')
        data_path = os.path.join(labeled_dir, 'labeled_data.csv')
        if not os.path.exists(data_path):
            print(f"No labeled data at {data_path}. Run 'python scripts/label.py' first.")
            sys.exit(1)
        df = pd.read_csv(data_path)
        df = df[df['programmatic_label'].notna()]
        texts = df['text'].tolist()
        labels = df['programmatic_label'].tolist()
    else:
        gold_dir = config.get('data', {}).get('storage', {}).get('gold_dir', 'data/gold')
        gold_path = os.path.join(gold_dir, 'gold_standard.csv')
        if not os.path.exists(gold_path):
            print(f"No gold standard at {gold_path}.")
            sys.exit(1)
        df = pd.read_csv(gold_path)
        texts = df['text'].tolist()
        labels = df['sentiment_gold'].tolist()

    print(f"Training on {len(texts)} samples from {args.source} labels")

    # Train
    pipeline = SentimentPipeline(model_config)
    report = pipeline.train(texts, labels)

    # Save
    model_dir = config.get('data', {}).get('storage', {}).get('model_dir', 'data/models')
    pipeline.save(model_dir)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"  Source: {args.source}")
    print(f"  Samples: {len(texts)}")
    if 'validation_metrics' in report:
        print(f"  Validation F1: {report['validation_metrics']['weighted_f1']:.3f}")
        print(f"  Validation Accuracy: {report['validation_metrics']['accuracy']:.3f}")
    print(f"  Features: {report['num_features']}")
    print(f"  Classes: {report['classes']}")
    print(f"  Model saved to: {model_dir}")

    # Feature importance
    features = pipeline.get_feature_importance(top_n=5)
    print(f"\n  Top features per class:")
    for cls, feats in features.items():
        top3 = ', '.join(f[0] for f in feats[:3])
        print(f"    {cls}: {top3}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
