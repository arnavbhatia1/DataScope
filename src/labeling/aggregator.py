"""
Label Aggregation Strategies

Combines votes from multiple labeling functions into a single
label per post. Implements three strategies of increasing sophistication.
"""

import json
from collections import Counter
from src.labeling.functions import (
    LABELING_FUNCTIONS, METADATA_FUNCTIONS, ABSTAIN
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Per-function weights reflecting estimated precision
FUNCTION_WEIGHTS = {
    'lf_options_directional': 3.0,
    'lf_price_target_mention': 2.5,
    'lf_news_language': 2.5,
    'lf_keyword_bullish': 2.0,
    'lf_keyword_bearish': 2.0,
    'lf_keyword_neutral': 2.0,
    'lf_keyword_meme': 2.0,
    'lf_sarcasm_indicators': 2.0,
    'lf_question_structure': 1.5,
    'lf_all_caps_ratio': 1.5,
    'lf_loss_reporting': 1.5,
    'lf_self_deprecating': 1.5,
    'lf_emoji_bullish': 1.0,
    'lf_emoji_bearish': 1.0,
    'lf_emoji_meme': 1.0,
    'lf_short_post': 1.0,
}


def _parse_metadata(metadata):
    """Safely parse metadata from string to dict if needed."""
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        # Try JSON first (safe), then fall back to treating as unparseable
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # Try replacing single quotes with double quotes for Python dict literals
        try:
            parsed = json.loads(metadata.replace("'", '"'))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class LabelAggregator:

    def __init__(self, strategy="confidence_weighted", config=None):
        self.strategy = strategy
        self.config = config or {}
        labeling_cfg = self.config.get('labeling', {})
        self.confidence_threshold = labeling_cfg.get('confidence_threshold', 0.6)
        self.min_votes = labeling_cfg.get('min_votes', 2)

    def aggregate_single(self, text, metadata=None):
        """
        Run all labeling functions on a single post.

        Returns dict with: final_label, confidence, votes, num_votes,
        num_abstains, has_conflict, competing_labels.
        """
        votes = {}

        # Run text-only functions
        for func in LABELING_FUNCTIONS:
            result = func(text)
            votes[func.__name__] = result

        # Run metadata-aware functions if metadata provided
        if metadata:
            for func in METADATA_FUNCTIONS:
                result = func(text, metadata=metadata)
                votes[func.__name__] = result

        # Filter out abstains
        active_votes = {k: v for k, v in votes.items() if v != ABSTAIN}
        num_votes = len(active_votes)
        num_abstains = len(votes) - num_votes

        if num_votes == 0:
            return {
                'final_label': None,
                'confidence': 0.0,
                'votes': votes,
                'num_votes': 0,
                'num_abstains': num_abstains,
                'has_conflict': False,
                'competing_labels': {},
            }

        # Count votes per label
        label_counts = Counter(active_votes.values())
        has_conflict = len(label_counts) > 1

        # Apply confidence-weighted strategy
        final_label, confidence = self._confidence_weighted(active_votes)

        # Enforce min_votes
        if num_votes < self.min_votes:
            final_label = None
            confidence = 0.0

        return {
            'final_label': final_label,
            'confidence': confidence,
            'votes': votes,
            'num_votes': num_votes,
            'num_abstains': num_abstains,
            'has_conflict': has_conflict,
            'competing_labels': dict(label_counts),
        }

    def aggregate_batch(self, df):
        """
        Label entire DataFrame. Add columns:
        programmatic_label, label_confidence, label_coverage,
        label_conflict, vote_breakdown.
        """
        results = []
        for _, row in df.iterrows():
            metadata = row.get('metadata') if 'metadata' in df.columns else None
            metadata = _parse_metadata(metadata)
            result = self.aggregate_single(row['text'], metadata=metadata)
            results.append(result)

        df = df.copy()
        df['programmatic_label'] = [r['final_label'] for r in results]
        df['label_confidence'] = [r['confidence'] for r in results]
        df['label_coverage'] = [r['num_votes'] > 0 for r in results]
        df['label_conflict'] = [r['has_conflict'] for r in results]
        df['vote_breakdown'] = [r['competing_labels'] for r in results]

        labeled_count = df['programmatic_label'].notna().sum()
        logger.info(f"Labeled {labeled_count}/{len(df)} posts "
                     f"({labeled_count/len(df):.1%} coverage)")

        return df

    def _weighted_vote(self, active_votes):
        """
        Each function has a weight. Sum weights per label. Highest wins.
        """
        weighted_scores = Counter()
        total_weight = 0.0

        for func_name, label in active_votes.items():
            weight = FUNCTION_WEIGHTS.get(func_name, 1.0)
            weighted_scores[label] += weight
            total_weight += weight

        winner = weighted_scores.most_common(1)[0][0]
        confidence = weighted_scores[winner] / total_weight if total_weight > 0 else 0.0
        return winner, confidence

    def _confidence_weighted(self, active_votes):
        """
        Like weighted vote, but only assign label if confidence > threshold.
        Posts below threshold get label=None (uncertain).
        """
        winner, confidence = self._weighted_vote(active_votes)

        if confidence < self.confidence_threshold:
            return None, confidence

        return winner, confidence
