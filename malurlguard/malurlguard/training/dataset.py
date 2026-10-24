"""Dataset loading and splitting utilities."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from malurlguard.features.extractor import FeatureExtractor

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "benign", 1: "phishing", 2: "malware", 3: "defacement"}


class UrlDataset:
    """
    Loads URL CSV data, extracts features, and provides train/val/test splits.

    Expected CSV columns: url, label (int 0-3), source (str).
    """

    def __init__(self, extractor: FeatureExtractor | None = None) -> None:
        self.extractor = extractor or FeatureExtractor(use_host_features=False)
        self.df: pd.DataFrame | None = None
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def load(self, path: str | Path) -> "UrlDataset":
        """
        Load and validate a dataset CSV.

        Args:
            path: Path to CSV file with columns [url, label, source].

        Returns:
            self for chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.df = pd.read_csv(path)
        required_cols = {"url", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        # Drop rows with invalid labels or missing URLs
        self.df = self.df.dropna(subset=["url", "label"])
        self.df["label"] = self.df["label"].astype(int)
        valid_labels = set(LABEL_NAMES.keys())
        self.df = self.df[self.df["label"].isin(valid_labels)]
        self.df["url"] = self.df["url"].astype(str).str.strip()
        self.df = self.df[self.df["url"].str.len() > 5]

        logger.info(
            "Loaded %d samples from %s | label dist: %s",
            len(self.df),
            path.name,
            self.df["label"].value_counts().to_dict(),
        )
        return self

    def extract_features(self) -> "UrlDataset":
        """Run feature extraction over all loaded URLs."""
        if self.df is None:
            raise RuntimeError("Call .load() before .extract_features()")
        logger.info("Extracting features for %d URLs…", len(self.df))
        X_df = self.extractor.extract_batch(self.df["url"].tolist())
        self.X = X_df.values.astype(np.float32)
        self.y = self.df["label"].values.astype(np.int32)
        logger.info("Feature extraction complete. Shape: %s", self.X.shape)
        return self

    def get_splits(
        self,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Return stratified (train, val, test) splits as (X, y) pairs.

        Args:
            val_size: Fraction of data for validation.
            test_size: Fraction of data for test.
            random_state: Random seed.

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Call .extract_features() first.")

        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=random_state
        )
        # Then split val from remaining
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=random_state
        )
        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(y_train), len(y_val), len(y_test),
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @property
    def feature_names(self) -> list[str]:
        return self.extractor.get_feature_names()

    @property
    def n_classes(self) -> int:
        if self.y is None:
            return 0
        return int(np.max(self.y)) + 1
