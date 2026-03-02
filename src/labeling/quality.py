"""
Label Quality Assessment

Measures how good our programmatic labels are before we train any model.
This is the most Snorkel-relevant part of the project.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.labeling.functions import LABELING_FUNCTIONS, METADATA_FUNCTIONS, ABSTAIN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelQualityAnalyzer:
    """
    Analyzes the quality of programmatic labels across labeling functions.
    Computes coverage, conflicts, accuracy, and comparison to gold standard.
    """

    def __init__(self, labeling_functions=None, aggregator=None):
        """
        Initialize the analyzer.

        Args:
            labeling_functions: List of labeling functions to analyze.
                               Defaults to LABELING_FUNCTIONS.
            aggregator: LabelAggregator instance (optional, for reference).
        """
        self.labeling_functions = labeling_functions or LABELING_FUNCTIONS
        self.aggregator = aggregator

    def per_function_report(self, df):
        """
        For each labeling function, compute:
        - coverage: % of posts it voted on (vs abstained)
        - label_distribution: what labels it assigns and in what proportion
        - conflict_rate: how often it disagrees with final aggregated label
        - accuracy_on_gold: if gold labels exist, precision and recall

        Args:
            df: DataFrame with 'text' column and optionally
                'programmatic_label' and 'sentiment_gold' columns.

        Returns:
            DataFrame with one row per function and quality metrics.
        """
        rows = []

        for func in self.labeling_functions:
            votes = []
            conflicts = 0
            total_voted = 0
            label_counts = Counter()
            gold_correct = 0
            gold_total = 0

            for _, row in df.iterrows():
                result = func(row['text'])
                votes.append(result)

                if result != ABSTAIN:
                    total_voted += 1
                    label_counts[result] += 1

                    # Track conflicts with programmatic label if available
                    if ('programmatic_label' in df.columns and
                        pd.notna(row.get('programmatic_label'))):
                        if result != row['programmatic_label']:
                            conflicts += 1

                    # Track agreement with gold labels if available
                    if 'sentiment_gold' in df.columns:
                        gold_total += 1
                        if result == row.get('sentiment_gold'):
                            gold_correct += 1

            coverage = total_voted / len(df) if len(df) > 0 else 0.0
            conflict_rate = conflicts / total_voted if total_voted > 0 else 0.0
            accuracy_on_gold = gold_correct / gold_total if gold_total > 0 else None

            rows.append({
                'function': func.__name__,
                'coverage': float(coverage),
                'num_voted': int(total_voted),
                'conflict_rate': float(conflict_rate),
                'label_distribution': dict(label_counts),
                'accuracy_on_gold': float(accuracy_on_gold) if accuracy_on_gold else None,
            })

        logger.info(f"Per-function report: {len(rows)} functions analyzed")
        return pd.DataFrame(rows)

    def aggregate_quality_report(self, df):
        """
        Overall labeling pipeline quality metrics.

        Computes:
        - total_coverage: % of posts that received at least one vote
        - avg_votes_per_post: mean number of functions that voted per post
        - conflict_rate: % of posts where functions disagreed
        - confidence_distribution: stats on confidence scores
        - uncertain_count: posts that couldn't be confidently labeled
        - label_distribution: final label distribution after aggregation

        Args:
            df: DataFrame with programmatic labeling results.

        Returns:
            Dict with aggregate quality metrics.
        """
        has_label = (df['programmatic_label'].notna()
                     if 'programmatic_label' in df.columns
                     else pd.Series([False] * len(df)))
        total_coverage = float(has_label.mean())

        # Avg votes per post from vote_breakdown
        avg_votes = 0.0
        if 'vote_breakdown' in df.columns:
            vote_counts = df['vote_breakdown'].apply(
                lambda x: sum(x.values()) if isinstance(x, dict) else 0
            )
            avg_votes = float(vote_counts.mean())

        # Conflict rate
        conflict_rate = float(df['label_conflict'].mean()
                              if 'label_conflict' in df.columns else 0.0)

        # Confidence distribution
        confidence_vals = (df['label_confidence'].dropna()
                          if 'label_confidence' in df.columns
                          else pd.Series())
        confidence_stats = {
            'mean': float(confidence_vals.mean()) if len(confidence_vals) > 0 else 0.0,
            'median': float(confidence_vals.median()) if len(confidence_vals) > 0 else 0.0,
            'std': float(confidence_vals.std()) if len(confidence_vals) > 0 else 0.0,
            'min': float(confidence_vals.min()) if len(confidence_vals) > 0 else 0.0,
            'max': float(confidence_vals.max()) if len(confidence_vals) > 0 else 0.0,
        }

        # Uncertain posts (those without a label or below threshold)
        uncertain_count = int((~has_label).sum())

        # Label distribution
        label_dist = {}
        if 'programmatic_label' in df.columns:
            label_dist = df['programmatic_label'].value_counts().to_dict()
            label_dist = {k: int(v) for k, v in label_dist.items()}

        report = {
            'total_coverage': total_coverage,
            'avg_votes_per_post': avg_votes,
            'conflict_rate': conflict_rate,
            'confidence_distribution': confidence_stats,
            'uncertain_count': uncertain_count,
            'label_distribution': label_dist,
            'total_posts': int(len(df)),
            'labeled_posts': int(has_label.sum()),
        }

        logger.info(f"Aggregate quality report: {total_coverage:.1%} coverage, "
                    f"{avg_votes:.1f} avg votes/post, {conflict_rate:.1%} conflicts")
        return report

    def compare_to_gold(self, df, gold_df):
        """
        Compare programmatic labels to gold standard labels.

        Returns per-class P/R/F1, confusion matrix, and disagreement list.

        Args:
            df: DataFrame with programmatic labeling results.
            gold_df: DataFrame with gold standard labels. Must have columns:
                    'post_id', 'sentiment_gold', 'ambiguity_score' (optional).

        Returns:
            Dict with:
            - agreement_rate: % of posts where programmatic == gold
            - classification_report: per-class precision/recall/f1 from sklearn
            - confusion_matrix: DataFrame confusion matrix
            - agreement_by_ambiguity: agreement rate per ambiguity score
            - disagreements: list of disagreement details
            - total_compared: number of posts compared
            - total_gold: total posts in gold set
        """
        # Merge on post_id
        if 'post_id' not in df.columns or 'post_id' not in gold_df.columns:
            logger.warning("post_id column not found in one or both dataframes")
            return {
                'agreement_rate': 0.0,
                'per_class': {},
                'confusion_matrix': None,
                'disagreements': [],
                'total_compared': 0,
                'total_gold': len(gold_df) if len(gold_df) > 0 else 0,
            }

        merged = df.merge(
            gold_df[['post_id', 'sentiment_gold', 'ambiguity_score', 'notes']],
            on='post_id',
            how='inner'
        )

        if len(merged) == 0:
            logger.warning("No overlapping posts between programmatic and gold")
            return {
                'agreement_rate': 0.0,
                'per_class': {},
                'confusion_matrix': None,
                'disagreements': [],
                'total_compared': 0,
                'total_gold': len(gold_df),
            }

        # Filter to posts that got a programmatic label
        labeled = merged[merged['programmatic_label'].notna()].copy()

        if len(labeled) == 0:
            logger.warning("No programmatic labels to compare to gold")
            return {
                'agreement_rate': 0.0,
                'per_class': {},
                'confusion_matrix': None,
                'disagreements': [],
                'total_compared': 0,
                'total_gold': len(gold_df),
            }

        y_true = labeled['sentiment_gold'].tolist()
        y_pred = labeled['programmatic_label'].tolist()

        # Classification report
        labels = sorted(set(y_true + y_pred))
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        # Overall agreement rate
        agreement_rate = float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))

        # Agreement by ambiguity score (if available)
        agreement_by_ambiguity = {}
        if 'ambiguity_score' in labeled.columns:
            for score in sorted(labeled['ambiguity_score'].dropna().unique()):
                subset = labeled[labeled['ambiguity_score'] == score]
                if len(subset) > 0:
                    agree = float((subset['programmatic_label'] == subset['sentiment_gold']).mean())
                    agreement_by_ambiguity[int(score)] = agree

        # Disagreement details
        disagreements = []
        for _, row in labeled.iterrows():
            if row['programmatic_label'] != row['sentiment_gold']:
                disagreements.append({
                    'post_id': row['post_id'],
                    'text': row.get('text', '')[:200] if 'text' in row else '',
                    'programmatic_label': row['programmatic_label'],
                    'gold_label': row['sentiment_gold'],
                    'confidence': float(row['label_confidence']) if pd.notna(row.get('label_confidence')) else None,
                    'ambiguity_score': int(row['ambiguity_score']) if pd.notna(row.get('ambiguity_score')) else None,
                    'notes': row.get('notes', ''),
                })

        logger.info(f"Gold comparison: {agreement_rate:.1%} agreement rate, "
                    f"{len(labeled)} posts compared")

        return {
            'agreement_rate': agreement_rate,
            'classification_report': report,
            'confusion_matrix': cm_df,
            'agreement_by_ambiguity': agreement_by_ambiguity,
            'disagreements': disagreements,
            'total_compared': len(labeled),
            'total_gold': len(gold_df),
        }

    def label_quality_experiment(self, df, gold_df, X_test, y_test_gold):
        """
        THE THESIS EXPERIMENT.

        Train the same model (TF-IDF + LogReg) on:
        1. Gold labels (best possible data)
        2. Programmatic labels (our pipeline output)
        3. Noisy labels (random noise injected into gold)
        4. Random labels (completely random baseline)

        Evaluate all four on the same gold test set.

        Expected results:
        - Gold: F1 ~0.85+
        - Programmatic: F1 ~0.75-0.82 (close to gold!)
        - Noisy: F1 ~0.55-0.65 (significantly worse)
        - Random: F1 ~0.25 (chance level)

        This proves: data quality > model complexity.

        Args:
            df: DataFrame with texts and programmatic labels.
            gold_df: DataFrame with gold standard labels.
            X_test: Test texts (from gold set).
            y_test_gold: Gold test labels.

        Returns:
            Dict with comparison results and visualization data.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        results = {}

        # Get texts and labels for training
        gold_texts = gold_df['text'].tolist()
        gold_labels = gold_df['sentiment_gold'].tolist()
        prog_texts = df[df['programmatic_label'].notna()]['text'].tolist()
        prog_labels = df[df['programmatic_label'].notna()]['programmatic_label'].tolist()

        # Ensure we have enough data
        if len(gold_texts) < 10 or len(prog_texts) < 10:
            logger.warning("Insufficient training data for thesis experiment")
            return {
                'results': {},
                'visualization_data': {},
                'thesis_validated': False,
                'error': 'Insufficient training data'
            }

        # Common hyperparameters
        config = {
            'max_features': 500,
            'ngram_range': (1, 2),
            'min_df': 2,
        }
        model_config = {
            'C': 1.0,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
        }

        # 1. Train on gold labels
        logger.info("Training model on gold labels...")
        vectorizer_gold = TfidfVectorizer(**config)
        X_train_gold = vectorizer_gold.fit_transform(gold_texts)
        X_test_vec = vectorizer_gold.transform(X_test)
        model_gold = LogisticRegression(**model_config)
        model_gold.fit(X_train_gold, gold_labels)
        f1_gold = float(f1_score(y_test_gold, model_gold.predict(X_test_vec), average='weighted', zero_division=0))
        results['gold'] = {'f1': f1_gold}

        # 2. Train on programmatic labels
        logger.info("Training model on programmatic labels...")
        vectorizer_prog = TfidfVectorizer(**config)
        X_train_prog = vectorizer_prog.fit_transform(prog_texts)
        X_test_vec_prog = vectorizer_prog.transform(X_test)
        model_prog = LogisticRegression(**model_config)
        model_prog.fit(X_train_prog, prog_labels)
        f1_prog = float(f1_score(y_test_gold, model_prog.predict(X_test_vec_prog), average='weighted', zero_division=0))
        results['programmatic'] = {'f1': f1_prog}

        # 3. Train on noisy labels (30% random noise injected into gold)
        logger.info("Training model on noisy labels...")
        np.random.seed(42)
        noisy_labels = gold_labels.copy()
        num_noise = max(1, int(len(noisy_labels) * 0.3))
        noise_indices = np.random.choice(len(noisy_labels), num_noise, replace=False)
        all_labels = list(set(gold_labels))
        for idx in noise_indices:
            # Replace with random label
            noisy_labels[idx] = np.random.choice([l for l in all_labels if l != noisy_labels[idx]])

        vectorizer_noisy = TfidfVectorizer(**config)
        X_train_noisy = vectorizer_noisy.fit_transform(gold_texts)
        X_test_vec_noisy = vectorizer_noisy.transform(X_test)
        model_noisy = LogisticRegression(**model_config)
        model_noisy.fit(X_train_noisy, noisy_labels)
        f1_noisy = float(f1_score(y_test_gold, model_noisy.predict(X_test_vec_noisy), average='weighted', zero_division=0))
        results['noisy'] = {'f1': f1_noisy}

        # 4. Train on random labels
        logger.info("Training model on random labels...")
        random_labels = [np.random.choice(all_labels) for _ in gold_texts]
        vectorizer_random = TfidfVectorizer(**config)
        X_train_random = vectorizer_random.fit_transform(gold_texts)
        X_test_vec_random = vectorizer_random.transform(X_test)
        model_random = LogisticRegression(**model_config)
        model_random.fit(X_train_random, random_labels)
        f1_random = float(f1_score(y_test_gold, model_random.predict(X_test_vec_random), average='weighted', zero_division=0))
        results['random'] = {'f1': f1_random}

        # Check thesis: is programmatic > noisy?
        thesis_validated = f1_prog > f1_noisy

        # Gap between programmatic and gold
        gap = f1_gold - f1_prog

        logger.info(f"Thesis experiment results: "
                    f"Gold={f1_gold:.3f}, Prog={f1_prog:.3f}, "
                    f"Noisy={f1_noisy:.3f}, Random={f1_random:.3f}, "
                    f"Thesis validated={thesis_validated}")

        return {
            'results': results,
            'visualization_data': {
                'labels': ['Gold', 'Programmatic', 'Noisy', 'Random'],
                'f1_scores': [f1_gold, f1_prog, f1_noisy, f1_random],
            },
            'thesis_validated': bool(thesis_validated),
            'programmatic_vs_gold_gap': float(gap),
        }
