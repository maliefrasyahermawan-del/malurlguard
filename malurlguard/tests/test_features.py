"""Tests for feature extraction."""

import math
import pytest

from malurlguard.features.lexical import extract_lexical_features, feature_names
from malurlguard.features.extractor import FeatureExtractor


class TestLexicalFeatures:
    def test_basic_benign_url(self):
        f = extract_lexical_features("https://github.com/torvalds/linux")
        assert f["has_https"] == 1.0
        assert f["has_ip_as_hostname"] == 0.0
        assert f["has_at_symbol"] == 0.0
        assert f["url_length"] > 0

    def test_ip_hostname_detection(self):
        f = extract_lexical_features("http://192.168.1.1/admin")
        assert f["has_ip_as_hostname"] == 1.0

    def test_at_symbol_detection(self):
        f = extract_lexical_features("http://legit.com@evil.tk/phish")
        assert f["has_at_symbol"] == 1.0

    def test_suspicious_keyword(self):
        f = extract_lexical_features("http://paypal-secure-login.tk/verify")
        assert f["has_suspicious_keyword"] == 1.0
        assert f["suspicious_keyword_count"] >= 2

    def test_punycode_detection(self):
        f = extract_lexical_features("http://xn--pple-43d.com/login")
        assert f["has_punycode"] == 1.0

    def test_http_in_hostname(self):
        f = extract_lexical_features("http://https-paypal.com/login")
        assert f["has_https_in_hostname"] == 1.0

    def test_entropy_is_nonnegative(self):
        f = extract_lexical_features("https://example.com/path")
        assert f["shannon_entropy_hostname"] >= 0.0
        assert f["shannon_entropy_path"] >= 0.0

    def test_returns_all_feature_names(self):
        f = extract_lexical_features("https://example.com")
        names = feature_names()
        for name in names:
            assert name in f, f"Missing feature: {name}"

    def test_feature_values_are_finite(self):
        f = extract_lexical_features("http://192.0.2.1@evil.tk/login?id=1")
        for name, val in f.items():
            assert math.isfinite(val), f"Feature {name} is not finite: {val}"

    def test_empty_url_does_not_crash(self):
        f = extract_lexical_features("")
        assert isinstance(f, dict)
        assert f["url_length"] == 0

    def test_subdomain_count(self):
        f = extract_lexical_features("http://sub1.sub2.evil.tk/path")
        assert f["num_subdomains"] >= 2

    def test_digit_ratio(self):
        f = extract_lexical_features("http://1234567890.com/path")
        assert f["ratio_digits_to_total"] > 0


class TestFeatureExtractor:
    def test_batch_extraction(self):
        ext = FeatureExtractor()
        urls = [
            "https://github.com",
            "http://evil-phish.tk/login",
            "",
        ]
        df = ext.extract_batch(urls)
        assert len(df) == 3
        assert not df.isnull().any().any(), "DataFrame should not have NaN values"

    def test_to_numpy_shape(self):
        ext = FeatureExtractor()
        urls = ["https://example.com", "http://evil.tk"]
        arr = ext.to_numpy(urls)
        assert arr.ndim == 2
        assert arr.shape[0] == 2
        assert arr.shape[1] > 20  # at least 20 features
