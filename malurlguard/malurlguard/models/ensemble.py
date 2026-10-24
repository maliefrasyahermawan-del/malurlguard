"""Ensemble predictor — soft voting over multiple base models."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .base import UrlClassifierBase, LABEL_MAP

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Soft-voting ensemble that combines probability outputs from multiple models.

    Args:
        models: List of trained UrlClassifierBase instances.
        weights: Optional per-model weight list (same length as models).
                 If None, equal weighting is used.
    """

    def __init__(
        self,
        models: List[UrlClassifierBase],
        weights: Optional[List[float]] = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model required.")
        self.models = models
        self.weights = weights or [1.0] * len(models)
        if len(self.weights) != len(self.models):
            raise ValueError("weights length must match models length.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted-average probability matrix (n_samples, n_classes)."""
        total_weight = sum(self.weights)
        proba_sum: Optional[np.ndarray] = None

        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            if proba_sum is None:
                proba_sum = proba * (weight / total_weight)
            else:
                # Align class columns (some models may see fewer classes)
                if proba.shape[1] != proba_sum.shape[1]:
                    n_classes = proba_sum.shape[1]
                    aligned = np.zeros((proba.shape[0], n_classes))
                    for i, cls in enumerate(model.model.classes_):
                        if cls < n_classes:
                            aligned[:, cls] = proba[:, i]
                    proba = aligned
                proba_sum += proba * (weight / total_weight)

        return proba_sum  # type: ignore[return-value]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices with highest ensemble probability."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_with_confidence(self, X: np.ndarray) -> List[Tuple[int, float]]:
        """
        Predict class and confidence for each sample.

        Returns:
            List of (class_index, confidence_score) tuples.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        confidences = proba[np.arange(len(proba)), indices]
        return list(zip(indices.tolist(), confidences.tolist()))

    def predict_single(self, x: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict for a single feature vector.

        Args:
            x: 1-D feature array or 2-D (1, n_features).

        Returns:
            (label_name, confidence, full_probability_vector)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        proba = self.predict_proba(x)[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = LABEL_MAP.get(class_idx, "unknown")
        return label, confidence, proba
