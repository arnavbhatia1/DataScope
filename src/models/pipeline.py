"""
Production ML Pipeline for Sentiment Classification

TF-IDF + Logistic Regression. Supports training, evaluation,
inference, and model persistence.
"""

import re
import os
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPipeline:

    def __init__(self, config=None):
        self.config = config or {}
        self.vectorizer = None
        self.model = None
        self.metadata = {}
        self.is_trained = False

    def preprocess(self, text):
        """Minimal preprocessing: lowercase + whitespace normalization only."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train(self, texts, labels, validation_split=True):
        """
        Train the full pipeline.

        Returns training report dict with validation_metrics,
        cross_val_scores, feature_names, etc.
        """
        logger.info(f"Training on {len(texts)} samples")

        # Preprocess
        processed = [self.preprocess(t) for t in texts]

        # Config params
        max_features = self.config.get('max_features', 500)
        ngram_range = tuple(self.config.get('ngram_range', [1, 2]))
        min_df = self.config.get('min_df', 3)
        C = self.config.get('C', 1.0)
        class_weight = self.config.get('class_weight', 'balanced')
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)

        report = {}

        if validation_split and len(processed) > 20:
            X_train, X_val, y_train, y_val = train_test_split(
                processed, labels, test_size=test_size,
                random_state=random_state, stratify=labels
            )
        else:
            X_train, y_train = processed, labels
            X_val, y_val = None, None

        # Fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train model
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True

        # Training metrics
        train_pred = self.model.predict(X_train_vec)
        report['training_metrics'] = {
            'accuracy': float((np.array(y_train) == train_pred).mean()),
            'weighted_f1': float(f1_score(y_train, train_pred, average='weighted', zero_division=0)),
        }

        # Validation metrics
        if X_val is not None:
            X_val_vec = self.vectorizer.transform(X_val)
            val_pred = self.model.predict(X_val_vec)
            val_report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
            report['validation_metrics'] = {
                'accuracy': float(val_report['accuracy']),
                'weighted_f1': float(val_report['weighted avg']['f1-score']),
                'classification_report': val_report,
            }
            report['confusion_matrix'] = confusion_matrix(y_val, val_pred, labels=self.model.classes_).tolist()
            logger.info(f"Validation F1: {report['validation_metrics']['weighted_f1']:.3f}")

        # Cross-validation
        try:
            X_all_vec = self.vectorizer.transform(processed)
            cv_scores = cross_val_score(
                LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=random_state),
                X_all_vec, labels, cv=min(5, len(set(labels))), scoring='f1_weighted'
            )
            report['cross_val_scores'] = {
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std()),
                'scores': cv_scores.tolist(),
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            report['cross_val_scores'] = None

        # Store metadata
        self.metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(texts),
            'num_features': len(self.vectorizer.get_feature_names_out()),
            'classes': self.model.classes_.tolist(),
            'config': self.config,
            'metrics': report,
        }

        report['num_features'] = self.metadata['num_features']
        report['classes'] = self.metadata['classes']

        return report

    def predict(self, texts):
        """Predict sentiment for one or more texts."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if isinstance(texts, str):
            texts = [texts]

        processed = [self.preprocess(t) for t in texts]
        X = self.vectorizer.transform(processed)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for i, text in enumerate(texts):
            prob_dict = {cls: float(prob) for cls, prob in
                        zip(self.model.classes_, probabilities[i])}
            results.append({
                'text': text,
                'label': predictions[i],
                'confidence': float(max(probabilities[i])),
                'probabilities': prob_dict,
            })
        return results

    def predict_single(self, text):
        """Convenience method for single text prediction."""
        return self.predict([text])[0]

    def get_feature_importance(self, top_n=15):
        """Return top predictive features per class."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        feature_names = self.vectorizer.get_feature_names_out()
        result = {}

        for i, cls in enumerate(self.model.classes_):
            coefs = self.model.coef_[i]
            top_indices = np.argsort(coefs)[-top_n:][::-1]
            result[cls] = [(feature_names[j], float(coefs[j])) for j in top_indices]

        return result

    def error_analysis(self, texts, true_labels, predicted_labels):
        """Detailed analysis of misclassifications."""
        errors = []
        confidences = []

        if self.is_trained:
            preds = self.predict(texts)
            confidences = [p['confidence'] for p in preds]

        correct_confs = []
        incorrect_confs = []
        confusion_pairs = Counter()

        for i, (text, true_l, pred_l) in enumerate(zip(texts, true_labels, predicted_labels)):
            conf = confidences[i] if i < len(confidences) else None

            if true_l == pred_l:
                if conf is not None:
                    correct_confs.append(conf)
                continue

            if conf is not None:
                incorrect_confs.append(conf)

            confusion_pairs[(true_l, pred_l)] += 1

            # Categorize error
            if conf and conf < 0.4:
                category = 'model_uncertainty'
            elif true_l in ('meme', 'bullish') and pred_l in ('meme', 'bullish'):
                category = 'labeling_ambiguity'
            else:
                category = 'model_limitation'

            errors.append({
                'text': text[:200],
                'true_label': true_l,
                'predicted_label': pred_l,
                'confidence': conf,
                'error_category': category,
            })

        most_confused = confusion_pairs.most_common(1)[0] if confusion_pairs else None

        return {
            'errors': errors,
            'total_errors': len(errors),
            'error_rate': len(errors) / len(texts) if texts else 0,
            'most_confused_pair': most_confused,
            'avg_confidence_correct': float(np.mean(correct_confs)) if correct_confs else None,
            'avg_confidence_incorrect': float(np.mean(incorrect_confs)) if incorrect_confs else None,
        }

    def save(self, path="data/models"):
        """Save trained pipeline to disk."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.model, os.path.join(path, 'sentiment_model.pkl'))

        with open(os.path.join(path, 'model_metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    def load(self, path="data/models"):
        """Load trained pipeline from disk."""
        self.vectorizer = joblib.load(os.path.join(path, 'tfidf_vectorizer.pkl'))
        self.model = joblib.load(os.path.join(path, 'sentiment_model.pkl'))

        meta_path = os.path.join(path, 'model_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        self.is_trained = True
        logger.info(f"Model loaded from {path}")
