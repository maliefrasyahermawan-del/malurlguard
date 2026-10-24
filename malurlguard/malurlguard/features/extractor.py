"""Feature extraction orchestrator."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from .lexical import extract_lexical_features, feature_names
from .host_based import extract_host_features

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Orchestrates lexical + optional host-based feature extraction.

    Args:
        use_host_features: If True, perform DNS/WHOIS lookups (requires network).
    """

    def __init__(self, use_host_features: bool = False) -> None:
        self.use_host_features = use_host_features
        self._feature_names: List[str] | None = None

    def extract(self, url: str) -> Dict[str, float]:
        """Extract all features for a single URL."""
        features = extract_lexical_features(url)
        if self.use_host_features:
            host_features = extract_host_features(url)
            features.update(host_features)
        return features

    def extract_batch(self, urls: List[str]) -> pd.DataFrame:
        """
        Extract features for a list of URLs and return as DataFrame.

        Args:
            urls: List of URL strings.

        Returns:
            DataFrame with one row per URL and one column per feature.
        """
        rows = []
        for i, url in enumerate(urls):
            try:
                row = self.extract(url)
            except Exception as exc:
                logger.warning("Feature extraction failed for URL[%d] %s: %s", i, url, exc)
                row = {name: 0.0 for name in self.get_feature_names()}
            rows.append(row)
        return pd.DataFrame(rows).fillna(0.0)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names produced by this extractor."""
        if self._feature_names is None:
            sample = self.extract("https://example.com/path?q=1")
            self._feature_names = list(sample.keys())
        return self._feature_names

    def to_numpy(self, urls: List[str]) -> np.ndarray:
        """Extract features and return as a 2-D NumPy array."""
        df = self.extract_batch(urls)
        return df.values.astype(np.float32)
