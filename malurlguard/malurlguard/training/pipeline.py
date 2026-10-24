"""End-to-end training pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from malurlguard.features.extractor import FeatureExtractor
from malurlguard.models.ensemble import EnsemblePredictor
from malurlguard.models.random_forest import RandomForestUrlClassifier
from malurlguard.models.xgboost_model import XGBoostUrlClassifier
from malurlguard.training.dataset import UrlDataset
from malurlguard.training.evaluate import ModelEvaluator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates: data loading → feature extraction → model training → evaluation → persistence.

    Args:
        data_path: Path to CSV dataset.
        output_dir: Directory to save trained models and metrics.
        models: Which models to train ('rf', 'xgb', 'all').
    """

    MODEL_RF = "rf"
    MODEL_XGB = "xgb"
    ALL_MODELS = [MODEL_RF, MODEL_XGB]

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path = "models_pretrained",
        models: str = "all",
    ) -> None:
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if models == "all":
            self.model_keys = self.ALL_MODELS
        else:
            self.model_keys = [m.strip() for m in models.split(",")]

        self.extractor = FeatureExtractor(use_host_features=False)
        self.evaluator = ModelEvaluator(output_dir=self.output_dir / "metrics")
        self.trained_models: Dict[str, object] = {}

    def run(self) -> Dict[str, Dict]:
        """Execute the full pipeline and return metrics for each model."""
        dataset = UrlDataset(extractor=self.extractor)
        dataset.load(self.data_path).extract_features()

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.get_splits()
        feature_names = dataset.feature_names

        all_metrics: Dict[str, Dict] = {}

        for key in self.model_keys:
            logger.info("=== Training model: %s ===", key)
            model = self._build_model(key)
            model.train(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = self.evaluator.evaluate(y_test, y_pred, y_proba, model_name=key)

            # Latency benchmark
            latency = self.evaluator.benchmark_latency(model.predict_proba, X_test)
            metrics["latency"] = latency

            save_path = self.output_dir / f"{key}_model.joblib"
            model.save(save_path)

            self.trained_models[key] = model
            all_metrics[key] = metrics

        # Build + evaluate ensemble if multiple models trained
        if len(self.trained_models) > 1:
            models_list = [self.trained_models[k] for k in self.model_keys]
            ensemble = EnsemblePredictor(models_list)
            y_proba_ens = ensemble.predict_proba(X_test)
            y_pred_ens = np.argmax(y_proba_ens, axis=1)
            ens_metrics = self.evaluator.evaluate(y_test, y_pred_ens, y_proba_ens, "ensemble")
            all_metrics["ensemble"] = ens_metrics

        # Persist summary
        summary_path = self.output_dir / "metrics" / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Training summary saved to %s", summary_path)

        # Save feature names
        fn_path = self.output_dir / "feature_names.json"
        with open(fn_path, "w") as f:
            json.dump(feature_names, f)

        return all_metrics

    def _build_model(self, key: str):
        if key == self.MODEL_RF:
            return RandomForestUrlClassifier()
        elif key == self.MODEL_XGB:
            return XGBoostUrlClassifier()
        else:
            raise ValueError(f"Unknown model key: {key}. Choose from {self.ALL_MODELS}")
