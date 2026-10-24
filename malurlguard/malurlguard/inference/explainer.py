"""SHAP-based explanation of individual URL predictions."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from malurlguard.features.extractor import FeatureExtractor
from malurlguard.models.random_forest import RandomForestUrlClassifier
from malurlguard.models.xgboost_model import XGBoostUrlClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models_pretrained"


class ShapExplainer:
    """
    Generates SHAP feature-importance explanations for URL predictions.

    Uses TreeExplainer (fast, exact for tree-based models).
    Falls back gracefully if SHAP is unavailable.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.extractor = FeatureExtractor(use_host_features=False)
        self._feature_names = feature_names
        self._explainer = None
        self._base_model = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import shap  # noqa: F401
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")
            self._loaded = True
            return

        import shap

        if self._feature_names is None:
            self._feature_names = self.extractor.get_feature_names()

        # Prefer XGBoost (cleaner SHAP support)
        xgb_path = self.model_dir / "xgb_model.joblib"
        rf_path = self.model_dir / "rf_model.joblib"

        if xgb_path.exists():
            model = XGBoostUrlClassifier()
            model.load(xgb_path)
            self._base_model = model
            self._explainer = shap.TreeExplainer(model.model)
            logger.info("SHAP explainer using XGBoost.")
        elif rf_path.exists():
            model = RandomForestUrlClassifier()
            model.load(rf_path)
            self._base_model = model
            self._explainer = shap.TreeExplainer(model.model)
            logger.info("SHAP explainer using RandomForest.")
        else:
            logger.warning("No model found for SHAP explainer.")

        self._loaded = True

    def explain(self, url: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Compute top-N SHAP feature contributions for a single URL.

        Args:
            url: URL to explain.
            top_n: Number of top features to return.

        Returns:
            List of dicts with keys: feature, value, shap_value, direction.
            Sorted by absolute SHAP value descending.
        """
        self._load()

        features = self.extractor.extract(url)
        feature_names = list(features.keys())
        x = np.array([list(features.values())], dtype=np.float32)

        if self._explainer is None:
            return self._fallback_importance(features, top_n)

        try:
            shap_values = self._explainer.shap_values(x)
            # For multi-class, shap_values is a list; take class with highest probability
            if isinstance(shap_values, list):
                # Use class 1 (phishing) for binary or highest pred class
                pred_class = int(np.argmax(self._base_model.predict_proba(x)[0]))
                sv = shap_values[pred_class][0]
            else:
                sv = shap_values[0]

            contributions = sorted(
                zip(feature_names, x[0], sv),
                key=lambda t: abs(t[2]),
                reverse=True,
            )[:top_n]

            return [
                {
                    "feature": fname,
                    "value": round(float(fval), 4),
                    "shap_value": round(float(sval), 6),
                    "direction": "increases_risk" if sval > 0 else "decreases_risk",
                }
                for fname, fval, sval in contributions
            ]
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return self._fallback_importance(features, top_n)

    def _fallback_importance(
        self, features: Dict[str, float], top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Return top features by raw value magnitude when SHAP is unavailable.
        """
        sorted_features = sorted(
            features.items(), key=lambda t: abs(t[1]), reverse=True
        )[:top_n]
        return [
            {
                "feature": name,
                "value": round(float(val), 4),
                "shap_value": None,
                "direction": "unknown",
            }
            for name, val in sorted_features
        ]
