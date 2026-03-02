"""
Model Versioning — track and compare model versions.
"""

import os
import json
import glob
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelVersion:

    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _next_version_number(self):
        """Get next version number by scanning existing versions."""
        existing = glob.glob(os.path.join(self.model_dir, "v*"))
        if not existing:
            return 1
        nums = []
        for path in existing:
            dirname = os.path.basename(path)
            try:
                nums.append(int(dirname.split('_')[0][1:]))
            except (ValueError, IndexError):
                continue
        return max(nums) + 1 if nums else 1

    def save_version(self, pipeline, label_source, metrics, notes=""):
        """
        Save a model version with full metadata.
        Creates: data/models/v{N}_{source}_{timestamp}/
        """
        version_num = self._next_version_number()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dirname = f"v{version_num}_{label_source}_{timestamp}"
        version_path = os.path.join(self.model_dir, dirname)
        os.makedirs(version_path, exist_ok=True)

        # Save model artifacts
        pipeline.save(version_path)

        # Save version metadata
        version_meta = {
            'version': version_num,
            'label_source': label_source,
            'training_date': timestamp,
            'dataset_size': pipeline.metadata.get('dataset_size', 0),
            'metrics': metrics,
            'config': pipeline.config,
            'notes': notes,
        }
        with open(os.path.join(version_path, 'version_metadata.json'), 'w') as f:
            json.dump(version_meta, f, indent=2, default=str)

        logger.info(f"Saved model version {version_num} ({label_source}) to {version_path}")
        return version_num

    def list_versions(self):
        """List all saved model versions with summary metrics."""
        versions = []
        for path in sorted(glob.glob(os.path.join(self.model_dir, "v*"))):
            meta_path = os.path.join(path, 'version_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                versions.append({
                    'version': meta.get('version'),
                    'label_source': meta.get('label_source'),
                    'training_date': meta.get('training_date'),
                    'dataset_size': meta.get('dataset_size'),
                    'path': path,
                    'notes': meta.get('notes', ''),
                })
        return versions

    def compare_versions(self, version_ids=None):
        """Compare metrics across model versions. Returns list of dicts."""
        versions = self.list_versions()
        if version_ids:
            versions = [v for v in versions if v['version'] in version_ids]

        comparison = []
        for v in versions:
            meta_path = os.path.join(v['path'], 'version_metadata.json')
            with open(meta_path) as f:
                meta = json.load(f)
            metrics = meta.get('metrics', {})
            comparison.append({
                'version': v['version'],
                'label_source': v['label_source'],
                'weighted_f1': metrics.get('weighted_f1', metrics.get('validation_metrics', {}).get('weighted_f1')),
                'accuracy': metrics.get('accuracy', metrics.get('validation_metrics', {}).get('accuracy')),
                'dataset_size': v['dataset_size'],
            })
        return comparison

    def load_version(self, version_id):
        """Load a specific model version into a SentimentPipeline."""
        from src.models.pipeline import SentimentPipeline

        for path in glob.glob(os.path.join(self.model_dir, f"v{version_id}_*")):
            pipeline = SentimentPipeline()
            pipeline.load(path)
            return pipeline
        raise FileNotFoundError(f"Version {version_id} not found")
