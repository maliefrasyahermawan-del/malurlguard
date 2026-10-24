"""Random Forest URL classifier."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import UrlClassifierBase

logger = logging.getLogger(__name__)


class RandomForestUrlClassifier(UrlClassifierBase):
    """Wraps scikit-learn RandomForestClassifier with the UrlClassifierBase interface."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training RandomForest on %d samples…", len(y))
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("RandomForest training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("RandomForest saved to %s", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info("RandomForest loaded from %s", path)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
