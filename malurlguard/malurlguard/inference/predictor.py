"""URL prediction with cached model loading."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np

from malurlguard.features.extractor import FeatureExtractor
from malurlguard.models.base import LABEL_MAP
from malurlguard.models.ensemble import EnsemblePredictor
from malurlguard.models.random_forest import RandomForestUrlClassifier
from malurlguard.models.xgboost_model import XGBoostUrlClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models_pretrained"


class Predictor:
    """
    Load trained models once and serve predictions efficiently.

    Models are loaded on first use and cached in memory for the lifetime of
    the Predictor instance (suitable for long-running API servers).

    Args:
        model_dir: Directory containing .joblib model files.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.extractor = FeatureExtractor(use_host_features=False)
        self._ensemble: Optional[EnsemblePredictor] = None
        self._feature_names: List[str] = []
        self._loaded = False
        self._model_info: Dict[str, Any] = {}

    def _load_models(self) -> None:
        if self._loaded:
            return

        models = []
        weights = []

        # Try to load individual models
        rf_path = self.model_dir / "rf_model.joblib"
        xgb_path = self.model_dir / "xgb_model.joblib"

        if rf_path.exists():
            rf = RandomForestUrlClassifier()
            rf.load(rf_path)
            models.append(rf)
            weights.append(1.0)
            self._model_info["random_forest"] = str(rf_path)

        if xgb_path.exists():
            xgb = XGBoostUrlClassifier()
            xgb.load(xgb_path)
            models.append(xgb)
            weights.append(1.2)  # slight XGBoost preference
            self._model_info["xgboost"] = str(xgb_path)

        if not models:
            raise FileNotFoundError(
                f"No trained models found in {self.model_dir}. "
                "Run `malurlguard train` first."
            )

        self._ensemble = EnsemblePredictor(models, weights)

        # Load feature names if available
        fn_path = self.model_dir / "feature_names.json"
        if fn_path.exists():
            with open(fn_path) as f:
                self._feature_names = json.load(f)
        else:
            self._feature_names = self.extractor.get_feature_names()

        self._loaded = True
        logger.info("Predictor initialised with %d model(s).", len(models))

    def predict(self, url: str) -> Dict[str, Any]:
        """
        Classify a single URL.

        Args:
            url: URL string to classify.

        Returns:
            Dict with keys: label, label_id, confidence, probabilities, latency_ms.
        """
        self._load_models()
        t0 = time.perf_counter()

        features = self.extractor.extract(url)
        x = np.array([list(features.values())], dtype=np.float32)
        label, confidence, proba = self._ensemble.predict_single(x)  # type: ignore[union-attr]

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        label_id = int(np.argmax(proba))

        return {
            "label": label,
            "label_id": label_id,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)
            },
            "latency_ms": latency_ms,
        }

    def predict_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple URLs efficiently."""
        self._load_models()
        t0 = time.perf_counter()

        X = self.extractor.to_numpy(urls)
        results_raw = self._ensemble.predict_with_confidence(X)  # type: ignore[union-attr]
        proba_matrix = self._ensemble.predict_proba(X)  # type: ignore[union-attr]

        latency_total = (time.perf_counter() - t0) * 1000
        per_url_ms = round(latency_total / max(len(urls), 1), 3)

        results = []
        for i, (label_id, confidence) in enumerate(results_raw):
            proba = proba_matrix[i]
            results.append({
                "url": urls[i],
                "label": LABEL_MAP.get(label_id, "unknown"),
                "label_id": label_id,
                "confidence": round(float(confidence), 4),
                "probabilities": {
                    LABEL_MAP[j]: round(float(p), 4) for j, p in enumerate(proba)
                },
                "latency_ms": per_url_ms,
            })
        return results

    @property
    def model_info(self) -> Dict[str, Any]:
        self._load_models()
        return self._model_info
