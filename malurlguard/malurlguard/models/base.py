"""Abstract base class for URL classifiers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "benign", 1: "phishing", 2: "malware", 3: "defacement"}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}


class UrlClassifierBase(ABC):
    """Base interface for all URL classifier models."""

    name: str = "base"

    def __init__(self) -> None:
        self.model: Any = None
        self.is_trained: bool = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on feature matrix X and labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for each sample."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix (n_samples, n_classes)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""

    def label_name(self, idx: int) -> str:
        """Convert numeric label index to human-readable name."""
        return LABEL_MAP.get(idx, "unknown")
