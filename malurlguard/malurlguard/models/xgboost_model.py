"""XGBoost URL classifier."""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from xgboost import XGBClassifier

from .base import UrlClassifierBase

logger = logging.getLogger(__name__)


class XGBoostUrlClassifier(UrlClassifierBase):
    """Wraps XGBClassifier with the UrlClassifierBase interface."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        n_jobs: int = -1,
        random_state: int = 42,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._scale_pos_weight = scale_pos_weight
        self._init_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
        )
        self.model = XGBClassifier(**self._init_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training XGBoost on %d samples…", len(y))
        # Auto-compute scale_pos_weight for binary imbalance
        if self._scale_pos_weight is None and len(np.unique(y)) == 2:
            neg = np.sum(y == 0)
            pos = np.sum(y == 1)
            spw = neg / max(pos, 1)
            self.model.set_params(scale_pos_weight=spw)
        elif self._scale_pos_weight is not None:
            self.model.set_params(scale_pos_weight=self._scale_pos_weight)

        self.model.fit(X, y)
        self.is_trained = True
        logger.info("XGBoost training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("XGBoost saved to %s", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info("XGBoost loaded from %s", path)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
